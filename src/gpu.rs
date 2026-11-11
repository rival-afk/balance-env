//! GPU compute backend for the Balance environment (wgpu).
//!
//! The physics (pendulum ODE, RK4, rail clamp) runs in `pendulum.wgsl`,
//! one invocation per environment — identical equations to the CPU path.
//! Scoring, observation construction and rewards stay on the CPU and are
//! shared with the single-thread env (`env::ScoreState`, `env::verticality`),
//! so reward semantics are identical.

use crate::env::{ScoreState, Obs, CART_LIMIT, ASSIST_KP, ASSIST_KD};
use rand::prelude::*;
use rayon::prelude::*;

const WGSL: &str = include_str!("pendulum.wgsl");
const WORKGROUP: u32 = 256;

fn state_bytes(s: &[f32; 6]) -> [u8; 24] {
    let mut out = [0u8; 24];
    for (i, f) in s.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    out
}

fn bytes_state(b: &[u8]) -> [f32; 6] {
    let mut out = [0.0f32; 6];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&b[i * 4..i * 4 + 4]);
        *slot = f32::from_le_bytes(buf);
    }
    out
}

pub struct GpuCore {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    state_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    actions_buf: wgpu::Buffer,
    assist_buf: wgpu::Buffer,
    /// kept so the params buffer outlives the bind group
    #[allow(dead_code)]
    params_buf: wgpu::Buffer,
    pub n: usize,
}

impl GpuCore {
    fn build_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let mk = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("balance-env gpu bind group layout"),
            entries: &[
                mk(0, wgpu::BufferBindingType::Storage { read_only: false }),
                mk(1, wgpu::BufferBindingType::Storage { read_only: true }),
                mk(2, wgpu::BufferBindingType::Storage { read_only: true }),
                mk(3, wgpu::BufferBindingType::Uniform),
            ],
        })
    }

    pub fn new(n: usize, dt: f32) -> Result<Self, String> {
        if n == 0 {
            return Err("n must be >= 1".to_string());
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("balance-env gpu device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("failed to open GPU device: {e}"))?;

        let layout = Self::build_bind_group_layout(&device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("balance-env pendulum"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("balance-env gpu pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("balance-env gpu pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("balance-env gpu states"),
            size: (n * 24) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("balance-env gpu states readback"),
            size: (n * 24) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let actions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("balance-env gpu actions"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let assist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("balance-env gpu assists"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("balance-env gpu params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut params = [0u8; 16];
        params[0..4].copy_from_slice(&dt.to_le_bytes());
        params[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        queue.write_buffer(&params_buf, 0, &params);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("balance-env gpu bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &state_buf, offset: 0, size: None }) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &actions_buf, offset: 0, size: None }) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &assist_buf, offset: 0, size: None }) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &params_buf, offset: 0, size: None }) },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            state_buf,
            readback_buf,
            actions_buf,
            assist_buf,
            params_buf,
            n,
        })
    }

    /// Uploads the full state vector (spawn/reset).
    pub fn upload_states(&self, states: &[[f32; 6]]) {
        let mut bytes = Vec::with_capacity(states.len() * 24);
        for s in states {
            bytes.extend_from_slice(&state_bytes(s));
        }
        self.queue.write_buffer(&self.state_buf, 0, &bytes);
    }

    /// Uploads per-env assist (kp, kd).
    pub fn upload_assists(&self, kp_kd: &[[f32; 2]]) {
        let mut bytes = Vec::with_capacity(kp_kd.len() * 8);
        for pair in kp_kd {
            bytes.extend_from_slice(&pair[0].to_le_bytes());
            bytes.extend_from_slice(&pair[1].to_le_bytes());
        }
        self.queue.write_buffer(&self.assist_buf, 0, &bytes);
    }

    /// One physics step for all envs; returns the new state vector.
    pub fn step(&self, actions: &[i8]) -> Result<Vec<[f32; 6]>, String> {
        if actions.len() != self.n {
            return Err(format!("expected {} actions, got {}", self.n, actions.len()));
        }
        let mut act = Vec::with_capacity(actions.len() * 4);
        for &a in actions {
            act.extend_from_slice(&(a as i32).to_le_bytes());
        }
        self.queue.write_buffer(&self.actions_buf, 0, &act);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("balance-env gpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("balance-env gpu compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = self.n.div_ceil(WORKGROUP as usize) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.state_buf, 0, &self.readback_buf, 0, (self.n * 24) as u64);

        self.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        self.readback_buf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e:?}"))?;
        rx.recv()
            .map_err(|e| format!("GPU map callback dropped: {e}"))?
            .map_err(|e| format!("GPU buffer map failed: {e:?}"))?;

        let view = self
            .readback_buf
            .get_mapped_range(..)
            .map_err(|e| format!("GPU get_mapped_range failed: {e:?}"))?;
        let bytes: &[u8] = &view;
        let mut states = Vec::with_capacity(self.n);
        for i in 0..self.n {
            states.push(bytes_state(&bytes[i * 24..(i + 1) * 24]));
        }
        drop(view);
        self.readback_buf.unmap();
        Ok(states)
    }
}

/// GPU-backed batch environment: same obs/reward/done semantics as `Env`.
pub struct GpuEnv {
    gpu: GpuCore,
    rngs: Vec<StdRng>,
    scores: Vec<ScoreState>,
    states: Vec<[f32; 6]>,
    assist_levels: Vec<u8>,
    dt: f32,
}

impl GpuEnv {
    pub fn new(n: usize, seed: Option<u64>, assist_level: u8) -> Result<Self, String> {
        let gpu = GpuCore::new(n, 1.0 / 60.0)?;
        let assist_level = assist_level.min(4);
        let mut master = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed(rand::random::<[u8; 32]>()),
        };
        let mut rngs = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        let mut assist = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        for _ in 0..n {
            let env_seed = master.gen::<u64>();
            let mut rng = StdRng::seed_from_u64(env_seed);
            let th1 = std::f32::consts::PI + rng.gen_range(-0.05..0.05);
            let th2 = std::f32::consts::PI + rng.gen_range(-0.10..0.10);
            states.push([th1, th2, 0.0, 0.0, 0.0, 0.0]);
            assist.push([ASSIST_KP[assist_level as usize], ASSIST_KD[assist_level as usize]]);
            scores.push(ScoreState::new());
            rngs.push(rng);
        }
        gpu.upload_states(&states);
        gpu.upload_assists(&assist);
        Ok(Self {
            gpu,
            rngs,
            scores,
            states,
            assist_levels: vec![assist_level; n],
            dt: 1.0 / 60.0,
        })
    }

    /// True when a usable GPU adapter exists on this machine.
    pub fn is_available() -> bool {
        GpuCore::new(1, 1.0 / 60.0).is_ok()
    }

    pub fn n(&self) -> usize {
        self.gpu.n
    }

    pub fn set_assist(&mut self, level: u8) {
        let level = level.min(4);
        for a in self.assist_levels.iter_mut() {
            *a = level;
        }
        let assist: Vec<[f32; 2]> = self
            .assist_levels
            .iter()
            .map(|&l| [ASSIST_KP[l as usize], ASSIST_KD[l as usize]])
            .collect();
        self.gpu.upload_assists(&assist);
    }

    pub fn reset(&mut self) {
        for i in 0..self.rngs.len() {
            let th1 = std::f32::consts::PI + self.rngs[i].gen_range(-0.05..0.05);
            let th2 = std::f32::consts::PI + self.rngs[i].gen_range(-0.10..0.10);
            self.states[i] = [th1, th2, 0.0, 0.0, 0.0, 0.0];
            self.scores[i].reset();
        }
        self.gpu.upload_states(&self.states);
    }

    /// Physics on the GPU, scoring/obs/rewards on the CPU (rayon).
    pub fn step(&mut self, actions: &[i8]) -> Result<(Vec<Obs>, Vec<f32>, Vec<bool>), String> {
        self.states = self.gpu.step(actions)?;
        let dt = self.dt;

        let mut obs = Vec::with_capacity(self.states.len());
        let mut rewards = Vec::with_capacity(self.states.len());
        let mut dones = Vec::with_capacity(self.states.len());

        let updated: Vec<(Obs, f32, bool)> = self
            .states
            .par_iter()
            .zip(self.scores.par_iter_mut())
            .map(|(s, score)| {
                let (reward, done, vert) = score.update(s[0], s[1], dt);
                let o = Obs {
                    t1: s[0],
                    t2: s[1],
                    w1: s[2],
                    w2: s[3],
                    cart_x: s[4] / CART_LIMIT,
                    cart_v: s[5],
                    verticality: vert,
                    multiplier: score.multiplier,
                    score: score.score,
                };
                (o, reward, done)
            })
            .collect();

        for (o, r, d) in updated {
            obs.push(o);
            rewards.push(r);
            dones.push(d);
        }
        Ok((obs, rewards, dones))
    }

}
