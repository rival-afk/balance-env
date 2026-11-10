//! Balance physics — 1:1 port of the Godot game, both sharing
//! the Lagrangian-derived system matrix M*a = b, sympy-verified.
//! Constants and pathways match exactly: M=40, m1=m2=L1=L2=1, g=9.81.

use rand::prelude::*;

pub const GRAVITY: f32 = 9.81;
pub const M_CART: f32 = 40.0;
pub const CART_ACCEL_RAW: f32 = 900.0;
/// Same rail extent as the Godot lane (±360 px) — physics stays identical at the ends.
pub const CART_LIMIT: f32 = 360.0;

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
    pub fn vec(&self) -> Vec<f32> {
        self.as_array().to_vec()
    }
}

pub struct Env {
    rng: StdRng,
    pub s: State,
    pub assist_level: u8,
    pub score: f32,
    pub multiplier: f32,
    pub perfect_streak: f32,
    pub total_perfect: f32,
    pub balance_time: f32,
    pub fault_count: u32,
    pub fault_time: f32,
    pub prev_fault: bool,
}

impl Env {
    pub fn new(seed: Option<u64>, assist_level: u8) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed(rand::random::<[u8; 32]>()),
        };
        let th1 = std::f32::consts::PI + rng.gen_range(-0.10..0.10);
        let th2 = std::f32::consts::PI + rng.gen_range(-0.20..0.20);
        Self {
            rng,
            s: State { th1, th2, w1: 0.0, w2: 0.0, x: 0.0, v: 0.0 },
            assist_level: assist_level.min(4),
            score: 0.0, multiplier: 1.0,
            perfect_streak: 0.0, total_perfect: 0.0,
            balance_time: 0.0, fault_count: 0, fault_time: 0.0,
            prev_fault: false,
        }
    }

    pub fn reset(&mut self) {
        let th1 = std::f32::consts::PI + self.rng.gen_range(-0.10..0.10);
        let th2 = std::f32::consts::PI + self.rng.gen_range(-0.20..0.20);
        self.s = State { th1, th2, w1: 0.0, w2: 0.0, x: 0.0, v: 0.0 };
        self.score = 0.0; self.multiplier = 1.0;
        self.perfect_streak = 0.0; self.total_perfect = 0.0;
        self.balance_time = 0.0; self.fault_count = 0; self.fault_time = 0.0;
        self.prev_fault = false;
    }

    pub fn set_assist(&mut self, level: u8) {
        self.assist_level = level.min(4);
    }

    #[inline]
    fn verticality(t1: f32, t2: f32) -> f32 {
        let h1 = -t1.cos();
        let h2 = -t2.cos();
        ((h1 + h2) * 0.5).clamp(0.0, 1.0)
    }

    pub fn obs(&self) -> Obs {
        let v = Self::verticality(self.s.th1, self.s.th2);
        Obs {
            t1: self.s.th1, t2: self.s.th2, w1: self.s.w1, w2: self.s.w2,
            cart_x: self.s.x / CART_LIMIT, cart_v: self.s.v,
            verticality: v, multiplier: self.multiplier, score: self.score,
        }
    }

    pub fn step(&mut self, action: i8, dt: f32) -> (Obs, f32, bool) {
        let dir = (action as f32).clamp(-1.0, 1.0);
        let force = dir * CART_ACCEL_RAW;

        // RK4 — same structure as Godot _physics_process
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

        let v = Self::verticality(self.s.th1, self.s.th2);
        let perfect = v > PERFECT_VERT;
        let up_avg = (-self.s.th1.cos() - self.s.th2.cos()) * 0.5;
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
            if !self.prev_fault { self.fault_count += 1; }
        }
        self.prev_fault = fault;

        let reward = v * v * self.multiplier;
        let done = self.balance_time > 30.0 || v < 0.02;
        (self.obs(), reward, done)
    }
}

/// Partially vectorised derivative combo — builds matrix + rhs,
/// solves for raw accelerations, adds assist. Identical construction to Godot.
fn deriv(s: State, force: f32, assist: u8) -> [f32; 6] {
    let (t1, t2, w1, w2) = (s.th1, s.th2, s.w1, s.w2);
    let d = t1 - t2;
    let cd = d.cos();
    let c1 = t1.cos();
    let c2 = t2.cos();

    let m00 = M_CART + 2.0; let m01 = 2.0 * c1; let m02 = c2;
    let m10 = 2.0 * c1; let m11 = 2.0;    let m12 = cd;
    let m20 = c2;      let m21 = cd;     let m22 = 1.0;

    let s1 = t1.sin(); let s2 = t2.sin();
    let b0 = force + 2.0 * s1 * w1 * w1 + s2 * w2 * w2;
    let b1 = -(d.sin() * w2 * w2 + GRAVITY * 2.0 * s1);
    let b2 = d.sin() * w1 * w1 - GRAVITY * s2;

    // Cramer's rule — exactly the same as Godot.
    let det = m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20);

    let ac = (b0 * (m11 * m22 - m12 * m21) - m01 * (b1 * m22 - m12 * b2) + m02 * (b1 * m21 - m11 * b2)) / det;
    let mut a1 = (m00 * (b1 * m22 - m12 * b2) - b0 * (m10 * m22 - m12 * m20) + m02 * (m10 * b2 - b1 * m20)) / det;
    let mut a2 = (m00 * (m11 * b2 - b1 * m21) - m01 * (m10 * b2 - b1 * m20) + b0 * (m10 * m21 - m11 * m20)) / det;

    // assist — identical formulation to Godot
    let kp = ASSIST_KP[assist as usize];
    let kd = ASSIST_KD[assist as usize];
    a1 += kp * t1.sin() - kd * w1;
    a2 += kp * t2.sin() - kd * w2;

    [s.w1, s.w2, a1, a2, s.v, ac]
}
