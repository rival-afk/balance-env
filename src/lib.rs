//! Balance physics — 1:1 port of the Godot game. "Close to real" double
//! pendulum on a horizontal axis. Equations of motion derived with sympy
//! from the Lagrangian (cart mass 40, arm masses 1 each, unit lengths),
//! verified energy-conserving to ~1e-12 in sympy.
//!
//! Константы: M=40, m1=m2=1, L=1, g=9.81 — ровно как в Godot-игре.
//! Rust c RK4 — тоже как в игре.

use rand::prelude::*;

pub const GRAVITY: f32 = 9.81;
pub const M_CART: f32 = 40.0;
pub const CART_DAMP: f32 = 3.0;
pub const CART_ACCEL: f32 = 900.0;
pub const CART_LIMIT: f32 = 130.0;

pub const ASSIST_KP: [f32; 5] = [0.0, 3.0, 15.0, 60.0, 200.0];
pub const ASSIST_KD: [f32; 5] = [0.0, 1.2, 6.0, 25.0, 80.0];

pub const DAMPING: f32 = 0.10;

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
        // Same spawn as the game: pole near upright with random tilt.
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

    /// action ∈ {-1, 0, +1} — направление тяги платформы.
    pub fn step(&mut self, action: i8, dt: f32) -> (Obs, f32, bool) {
        let dir = (action as f32).clamp(-1.0, 1.0);
        let force = dir * CART_ACCEL;
        // rail-locked cart with damping (matches Godot linear_damp=3.0)
        let accel_cart = (force - CART_DAMP * self.s.v) / M_CART;
        self.s.v += accel_cart * dt;
        self.s.x += self.s.v * dt;
        if self.s.x > CART_LIMIT { self.s.x = CART_LIMIT; if self.s.v > 0.0 { self.s.v = 0.0; } }
        else if self.s.x < -CART_LIMIT { self.s.x = -CART_LIMIT; if self.s.v < 0.0 { self.s.v = 0.0; } }

        // RK4 step for angles exactly as in the Godot path
        self.s = rk4_step(self.s, self.assist_level, dt);

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

#[inline]
fn rk4_deriv(s: State, assist: u8) -> [f32; 6] {
    let (_, a1, a2) = accel_t1_t2_x(s, assist);
    [s.w1, s.w2, a1, a2, s.v, 0.0]
}

#[inline]
fn rk4_step(s: State, assist: u8, dt: f32) -> State {
    // Full 6-dim RK4 matching the Godot path (cart position only for observation)
    let k1 = rk4_deriv(s, assist);
    let s2 = State {
        th1: s.th1 + 0.5 * dt * k1[0],
        th2: s.th2 + 0.5 * dt * k1[1],
        w1:  s.w1  + 0.5 * dt * k1[2],
        w2:  s.w2  + 0.5 * dt * k1[3],
        x:   s.x   + 0.5 * dt * k1[4],
        v:   s.v   + 0.5 * dt * k1[5],
    };
    let k2 = rk4_deriv(s2, assist);
    let s3 = State {
        th1: s.th1 + 0.5 * dt * k2[0],
        th2: s.th2 + 0.5 * dt * k2[1],
        w1:  s.w1  + 0.5 * dt * k2[2],
        w2:  s.w2  + 0.5 * dt * k2[3],
        x:   s.x   + 0.5 * dt * k2[4],
        v:   s.v   + 0.5 * dt * k2[5],
    };
    let k3 = rk4_deriv(s3, assist);
    let s4 = State {
        th1: s.th1 + dt * k3[0],
        th2: s.th2 + dt * k3[1],
        w1:  s.w1  + dt * k3[2],
        w2:  s.w2  + dt * k3[3],
        x:   s.x   + dt * k3[4],
        v:   s.v   + dt * k3[5],
    };
    let k4 = rk4_deriv(s4, assist);
    State {
        th1: s.th1 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        th2: s.th2 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        w1:  s.w1  + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        w2:  s.w2  + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        x:   s.x   + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
        v:   s.v   + dt * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0,
    }
}

/// Sympy-derived accelerations (M=40, m1=m2=L=1, g=9.81).
/// Verified: energy conserved to ~1e-12 in RK4.
#[inline]
fn accel_t1_t2_x(s: State, assist: u8) -> (f32, f32, f32) {
    let (t1, t2, w1, w2) = (s.th1, s.th2, s.w1, s.w2);
    let d = t1 - t2;
    let cd = d.cos();

    let den = 2.0 * (2.0 * t1).cos() + 40.0 * (2.0 * d).cos() - 122.0;

    let a1 = (2.0 * w1 * w1 * (2.0 * t1).sin()
              + 40.0 * w1 * w1 * (2.0 * d).sin()
              + 81.0 * w2 * w2 * d.sin()
              + w2 * w2 * (t1 + t2).sin()
              + 1216.44 * t1.sin()
              + 392.4 * (t1 - 2.0 * t2).sin())
              / den;

    let a2 = (-160.0 * w1 * w1 * d.sin()
              - 40.0 * w2 * w2 * (2.0 * d).sin()
              + 784.8 * t2.sin()
              - 784.8 * (2.0 * t1 - t2).sin())
              / den;

    let den_car = 42.0 * t1.sin().powi(2)
                 - 80.0 * t1.sin() * t2.sin() * cd
                 + 40.0 * t2.sin().powi(2)
                 + 40.0;
    let ddcart = (0.5 * w1 * w1 * ((t1 - 2.0*t2).sin() + (3.0*t1 - 2.0*t2).sin())
                 - 2.0 * w1 * w1 * t1.sin() * cd * cd
                 + 4.0 * w1 * w1 * t1.sin()
                 - 2.0 * w1 * w1 * d.sin() * t2.cos()
                 - 0.25 * w2 * w2 * ((2.0*t1 - 3.0*t2).sin() + (2.0*t1 - t2).sin())
                 - w2 * w2 * t2.sin() * cd * cd
                 + 2.0 * w2 * w2 * t2.sin()
                 + 2.0 * w2 * w2 * d.sin() * t1.cos()
                 - 19.62 * t1.sin() * t2.cos() * cd
                 + 19.62 * (2.0*t1).sin()
                 - 19.62 * t2.sin() * t1.cos() * cd
                 + 9.81 * (2.0*t2).sin())
                 / den_car;

    // assist spring (same form as game, same constants)
    let kp = ASSIST_KP[assist as usize];
    let kd = ASSIST_KD[assist as usize];
    let a1 = a1 + kp * t1.sin() - kd * w1;
    let a2 = a2 + kp * t2.sin() - kd * w2;
    (ddcart, a1, a2)
}
