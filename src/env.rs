//! Balance physics — 1:1 port of the Godot game, both sharing
//! the Lagrangian-derived system matrix M*a = b, sympy-verified.
//! SI units throughout: kg, m, N, m/s. The Godot lane draws with 120 px/m.

use rand::prelude::*;

pub const GRAVITY: f32 = 9.81;
pub const M_CART: f32 = 12.0;
pub const M1: f32 = 1.0;
pub const M2: f32 = 1.0;
pub const L1: f32 = 1.0;
pub const L2: f32 = 1.0;
/// Cart drag (rail-locked) — from the original game's linear_damp=3.0.
pub const CART_DAMP: f32 = 3.0;
/// Angular damping on the arms — from the original CPU integrator (0.10 /s).
pub const DAMPING: f32 = 0.10;
/// Peak drive force, N (≈5 m/s² on the 14 kg cart+arms system).
pub const CART_ACCEL_RAW: f32 = 70.0;
/// Rail extent in metres (±2.2 m = ±264 px; the platform stops just short of the walls at ±3 m).
pub const CART_LIMIT: f32 = 2.2;
/// Fixed physics sub-step. plat.txt recommends 4–16 sub-steps per 60 Hz frame;
/// RK4 at 240 Hz keeps this pendulum's energy drift at ~2e-6 per 10 s.
pub const PHYS_DT: f32 = 1.0 / 240.0;

pub const ASSIST_KP: [f32; 5] = [0.0, 3.0, 15.0, 60.0, 200.0];
pub const ASSIST_KD: [f32; 5] = [0.0, 1.2, 6.0, 25.0, 80.0];

const RATE: f32 = 12.0;
const PERFECT_VERT: f32 = 0.90;
const DANGER_VERT: f32 = 0.35;
const MULT_GAIN: f32 = 0.2;
const MAX_MULT: f32 = 4.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct State {
    pub th1: f32,
    pub th2: f32,
    pub w1: f32,
    pub w2: f32,
    pub x: f32,
    pub v: f32,
}

/// Scoring/episode bookkeeping — shared by the CPU env and the GPU batch env.
/// Pure function of (th1, th2, dt) plus its own accumulated state.
#[derive(Clone, Copy, Debug)]
pub struct ScoreState {
    pub score: f32,
    pub multiplier: f32,
    pub perfect_streak: f32,
    pub total_perfect: f32,
    pub balance_time: f32,
    pub fault_count: u32,
    pub fault_time: f32,
    pub prev_fault: bool,
}

impl Default for ScoreState {
    fn default() -> Self {
        Self {
            score: 0.0,
            multiplier: 1.0,
            perfect_streak: 0.0,
            total_perfect: 0.0,
            balance_time: 0.0,
            fault_count: 0,
            fault_time: 0.0,
            prev_fault: false,
        }
    }
}

impl ScoreState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Updates all scoring from the post-step angles.
    /// Returns (reward, done, verticality).
    pub fn update(&mut self, th1: f32, th2: f32, dt: f32) -> (f32, bool, f32) {
        let v = verticality(th1, th2);
        let perfect = v > PERFECT_VERT;
        let up_avg = (-th1.cos() - th2.cos()) * 0.5;
        let fault = up_avg > -0.2 && v < DANGER_VERT;

        self.balance_time += dt;
        if perfect {
            self.perfect_streak += dt;
            self.total_perfect += dt;
        } else {
            self.perfect_streak = 0.0;
        }
        self.multiplier = 1.0 + (self.perfect_streak * MULT_GAIN).min(MAX_MULT);
        self.score += v * v * RATE * self.multiplier * dt;

        if fault {
            self.fault_time += dt;
            if !self.prev_fault {
                self.fault_count += 1;
            }
        }
        self.prev_fault = fault;

        let reward = v * v * self.multiplier;
        let done = self.balance_time > 30.0 || v < 0.02;
        (reward, done, v)
    }
}

/// Verticality — fraction of "up" the pair of arms still has (1 = full upright).
#[inline]
pub(crate) fn verticality(t1: f32, t2: f32) -> f32 {
    let h1 = -t1.cos();
    let h2 = -t2.cos();
    ((h1 + h2) * 0.5).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug)]
pub struct Obs {
    pub t1: f32, pub t2: f32, pub w1: f32, pub w2: f32,
    pub cart_x: f32, pub cart_v: f32,
    pub verticality: f32, pub multiplier: f32, pub score: f32,
}

impl Obs {
    pub fn as_array(&self) -> [f32; 9] {
        [self.t1, self.t2, self.w1, self.w2, self.cart_x, self.cart_v,
         self.verticality, self.multiplier, self.score]
    }
}

pub struct Env {
    rng: StdRng,
    pub s: State,
    pub assist_level: u8,
    pub score: ScoreState,
}

impl Env {
    pub fn new(seed: Option<u64>, assist_level: u8) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed(rand::random::<[u8; 32]>()),
        };
        let th1 = std::f32::consts::PI + rng.gen_range(-0.05..0.05);
        let th2 = std::f32::consts::PI + rng.gen_range(-0.10..0.10);
        Self {
            rng,
            s: State { th1, th2, w1: 0.0, w2: 0.0, x: 0.0, v: 0.0 },
            assist_level: assist_level.min(4),
            score: ScoreState::new(),
        }
    }

    pub fn reset(&mut self) {
        let th1 = std::f32::consts::PI + self.rng.gen_range(-0.05..0.05);
        let th2 = std::f32::consts::PI + self.rng.gen_range(-0.10..0.10);
        self.s = State { th1, th2, w1: 0.0, w2: 0.0, x: 0.0, v: 0.0 };
        self.score.reset();
    }

    pub fn set_assist(&mut self, level: u8) {
        self.assist_level = level.min(4);
    }

    pub fn obs(&self) -> Obs {
        let v = verticality(self.s.th1, self.s.th2);
        Obs {
            t1: self.s.th1, t2: self.s.th2, w1: self.s.w1, w2: self.s.w2,
            cart_x: self.s.x / CART_LIMIT, cart_v: self.s.v,
            verticality: v, multiplier: self.score.multiplier, score: self.score.score,
        }
    }

    pub fn step(&mut self, action: i8, dt: f32) -> (Obs, f32, bool) {
        let dir = (action as f32).clamp(-1.0, 1.0);
        let force = dir * CART_ACCEL_RAW;

        // plat.txt §4: integrate with small sub-steps (500–2000 Hz is the usual
        // advice; 240 Hz already gives ~2e-6 energy drift per 10 s here).
        let n = ((dt / PHYS_DT).round() as u32).max(1);
        let h = dt / n as f32;
        for _ in 0..n {
            self.integrate(force, h);
        }

        let (reward, done, _v) = self.score.update(self.s.th1, self.s.th2, dt);
        (self.obs(), reward, done)
    }

    /// One RK4 sub-step of the 3-DOF cart + double pendulum (plat.txt §3).
    fn integrate(&mut self, force: f32, dt: f32) {
        // RK4 — same structure as Godot _step
        let k1 = deriv(self.s, force, self.assist_level);
        let s2 = State {
            th1: self.s.th1 + 0.5 * dt * k1[0],
            th2: self.s.th2 + 0.5 * dt * k1[1],
            w1:  self.s.w1  + 0.5 * dt * k1[2],
            w2:  self.s.w2  + 0.5 * dt * k1[3],
            x:   self.s.x   + 0.5 * dt * k1[4],
            v:   self.s.v   + 0.5 * dt * k1[5],
        };
        let k2 = deriv(s2, force, self.assist_level);
        let s3 = State {
            th1: self.s.th1 + 0.5 * dt * k2[0],
            th2: self.s.th2 + 0.5 * dt * k2[1],
            w1:  self.s.w1  + 0.5 * dt * k2[2],
            w2:  self.s.w2  + 0.5 * dt * k2[3],
            x:   self.s.x   + 0.5 * dt * k2[4],
            v:   self.s.v   + 0.5 * dt * k2[5],
        };
        let k3 = deriv(s3, force, self.assist_level);
        let s4 = State {
            th1: self.s.th1 + dt * k3[0],
            th2: self.s.th2 + dt * k3[1],
            w1:  self.s.w1  + dt * k3[2],
            w2:  self.s.w2  + dt * k3[3],
            x:   self.s.x   + dt * k3[4],
            v:   self.s.v   + dt * k3[5],
        };
        let k4 = deriv(s4, force, self.assist_level);

        self.s.th1 = self.s.th1 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0;
        self.s.th2 = self.s.th2 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0;
        self.s.w1  = self.s.w1  + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0;
        self.s.w2  = self.s.w2  + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0;
        self.s.x   = self.s.x   + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0;
        self.s.v   = self.s.v   + dt * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0;

        // rail clamp (Godot-side uses same cart_limit but in pixels)
        if self.s.x > CART_LIMIT {
            self.s.x = CART_LIMIT;
            if self.s.v > 0.0 { self.s.v = 0.0; }
        } else if self.s.x < -CART_LIMIT {
            self.s.x = -CART_LIMIT;
            if self.s.v < 0.0 { self.s.v = 0.0; }
        }
    }
}

/// Builds the plat.txt §3 system A(q)·a = r and solves it with Cramer's rule.
/// Joint friction and the assist torque are part of r (r2, r3). Identical construction to Godot.
fn deriv(s: State, force: f32, assist: u8) -> [f32; 6] {
    let (t1, t2, w1, w2) = (s.th1, s.th2, s.w1, s.w2);
    let d = t1 - t2;
    let cd = d.cos();
    let c1 = t1.cos();
    let c2 = t2.cos();
    let s1 = t1.sin();
    let s2 = t2.sin();

    // M(q) * a = b — same generalized mass matrix as Godot/WGSL.
    let m00 = M_CART + M1 + M2;
    let m01 = (M1 + M2) * L1 * c1;
    let m02 = M2 * L2 * c2;
    let m10 = m01;
    let m11 = (M1 + M2) * L1 * L1;
    let m12 = M2 * L1 * L2 * cd;
    let m20 = m02;
    let m21 = m12;
    let m22 = M2 * L2 * L2;

    let b0 = force - CART_DAMP * s.v
        + (M1 + M2) * L1 * s1 * w1 * w1
        + M2 * L2 * s2 * w2 * w2;
    // plat.txt §3: joint friction and the assist torque live in the RHS (r2, r3),
    // they are not added to the solved accelerations.
    let kp = ASSIST_KP[assist as usize];
    let kd = ASSIST_KD[assist as usize];
    let b1 = -DAMPING * w1
        - M2 * L1 * L2 * d.sin() * w2 * w2
        - (M1 + M2) * GRAVITY * L1 * s1
        + kp * s1 - kd * w1;
    let b2 = -DAMPING * w2
        + M2 * L1 * L2 * d.sin() * w1 * w1
        - M2 * GRAVITY * L2 * s2
        + kp * s2 - kd * w2;

    // Cramer's rule — exactly the same as Godot.
    let det = m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20);

    let ac = (b0 * (m11 * m22 - m12 * m21) - m01 * (b1 * m22 - m12 * b2) + m02 * (b1 * m21 - m11 * b2)) / det;
    let a1 = (m00 * (b1 * m22 - m12 * b2) - b0 * (m10 * m22 - m12 * m20) + m02 * (m10 * b2 - b1 * m20)) / det;
    let a2 = (m00 * (m11 * b2 - b1 * m21) - m01 * (m10 * b2 - b1 * m20) + b0 * (m10 * m21 - m11 * m20)) / det;

    [s.w1, s.w2, a1, a2, s.v, ac]
}
