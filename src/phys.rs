//! Balance physics — same equations of motion as the Godot game.
//! Angles measured from vertical *up* (PI = upright pole), like the GDScript.

use rand::prelude::*;

// Physical constants match the Godot game exactly:
const GRAVITY: f32 = 9.81;
const DAMPING: f32 = 0.10;
const ARM_LEN: f32 = 1.0;      // normalised (1.0 = one unit)
const ARM_LEN2: f32 = 1.0;
const MASS: f32 = 1.0;
const CART_MASS: f32 = 40.0;   // platform mass (Godot)
const CART_DAMP: f32 = 3.0;    // platform linear_damp (Godot)
const CART_ACCEL: f32 = 900.0; // platform ACCEL (px/s^2 Godot)
const CART_LIMIT: f32 = 120.0; // normalised clamp for x
const RATE: f32 = 12.0;        // score rate at perfect verticality
const PERFECT_VERT: f32 = 0.90;
const DANGER_VERT: f32 = 0.35;
const MAX_MULT: f32 = 4.0;
const MULT_GAIN: f32 = 0.2;    // +0.2 per second of perfect streak

pub const ASSIST_KP: [f32; 5] = [0.0, 3.0, 15.0, 60.0, 200.0];
pub const ASSIST_KD: [f32; 5] = [0.0, 1.2, 6.0, 25.0, 80.0];

#[derive(Clone, Copy, Debug)]
pub struct Obs {
    pub t1: f32,
    pub t2: f32,
    pub w1: f32,
    pub w2: f32,
    pub cart_x: f32,
    pub cart_v: f32,
    pub verticality: f32,
    pub multiplier: f32,
    pub score: f32,
}

impl Obs {
    pub fn as_array(&self) -> [f32; 9] {
        [self.t1, self.t2, self.w1, self.w2, self.cart_x, self.cart_v,
         self.verticality, self.multiplier, self.score]
    }
}

pub struct Env {
    rng: StdRng,
    pub t1: f32,
    pub t2: f32,
    pub w1: f32,
    pub w2: f32,
    pub cart_x: f32,
    pub cart_v: f32,
    pub assist_level: u8,
    pub score: f32,
    pub multiplier: f32,
    pub perfect_streak: f32,
    pub total_perfect: f32,
    pub balance_time: f32,
    pub fault_count: u32,
    pub fault_time: f32,
    pub prev_fault: bool,
    pub prev_vx: f32,
}

impl Env {
    pub fn new(seed: Option<u64>, assist_level: u8) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed(rand::random::<[u8; 32]>()),
        };
        // same spawn as the game: vertical + tiny random tilt
        let t1 = std::f32::consts::PI + rng.gen_range(-0.10..0.10);
        let t2 = std::f32::consts::PI + rng.gen_range(-0.20..0.20);
        Self {
            rng,
            t1, t2,
            w1: 0.0, w2: 0.0,
            cart_x: 0.0,
            cart_v: 0.0,
            assist_level: assist_level.min(4),
            score: 0.0,
            multiplier: 1.0,
            perfect_streak: 0.0,
            total_perfect: 0.0,
            balance_time: 0.0,
            fault_count: 0,
            fault_time: 0.0,
            prev_fault: false,
            prev_vx: 0.0,
        }
    }

    pub fn assist(&self) -> u8 {
        self.assist_level
    }

    pub fn set_assist(&mut self, level: u8) {
        self.assist_level = level.min(4);
    }

    pub fn reset(&mut self) {
        let t1 = std::f32::consts::PI + self.rng.gen_range(-0.10..0.10);
        let t2 = std::f32::consts::PI + self.rng.gen_range(-0.20..0.20);
        self.t1 = t1;
        self.t2 = t2;
        self.w1 = 0.0;
        self.w2 = 0.0;
        self.cart_x = 0.0;
        self.cart_v = 0.0;
        self.score = 0.0;
        self.multiplier = 1.0;
        self.perfect_streak = 0.0;
        self.total_perfect = 0.0;
        self.balance_time = 0.0;
        self.fault_count = 0;
        self.fault_time = 0.0;
        self.prev_fault = false;
        self.prev_vx = 0.0;
    }

    /// Физика одного шага. action ∈ {-1, 0, +1}: направление тяги платформы.
    pub fn step(&mut self, action: i8, dt: f32) -> (Obs, f32, bool) {
        let dir = (action as f32).clamp(-1.0, 1.0);

        // --- cart update (rail-locked) ---
        let force = dir * CART_ACCEL * CART_MASS;
        let accel = (force / CART_MASS) - CART_DAMP * self.cart_v;
        self.cart_v += accel * dt;
        self.cart_x += self.cart_v * dt;
        if self.cart_x > CART_LIMIT {
            self.cart_x = CART_LIMIT;
            self.cart_v = self.cart_v.clamp(-100.0, 0.0).min(500.0) * 0.0; // жёсткий стоп
        } else if self.cart_x < -CART_LIMIT {
            self.cart_x = -CART_LIMIT;
            self.cart_v = 0.0;
        }
        let pivot_a = (self.cart_v - self.prev_vx) / dt.max(1e-6);
        self.prev_vx = self.cart_v;

        // --- double pendulum with pivot acceleration (RK4 sub-steps) ---
        let sub = 4u32;
        let h = dt / sub as f32;
        for _ in 0..sub {
            let (t1n, t2n, w1n, w2n) =
                rk4(self.t1, self.t2, self.w1, self.w2, pivot_a, self.assist_level, h);
            self.t1 = t1n;
            self.t2 = t2n;
            self.w1 = w1n;
            self.w2 = w2n;
        }

        // --- verticality / scoring (same as the game) ---
        let h1 = -self.t1.cos();
        let h2 = -self.t2.cos();
        let up_avg = (h1 + h2) * 0.5;
        let verticality = up_avg.clamp(0.0, 1.0);
        let tilt_total = up_avg.clamp(-1.0, 1.0).acos();

        let perfect = verticality > PERFECT_VERT;
        let fault = up_avg > -0.2 && verticality < DANGER_VERT;

        self.balance_time += dt;
        if perfect {
            self.perfect_streak += dt;
            self.total_perfect += dt;
        } else {
            self.perfect_streak = 0.0;
        }
        self.multiplier = 1.0 + (self.perfect_streak * MULT_GAIN).min(MAX_MULT);
        self.score += verticality * verticality * RATE * self.multiplier * dt;

        if fault {
            self.fault_time += dt;
            if !self.prev_fault {
                self.fault_count += 1;
            }
        }
        self.prev_fault = fault;

        // --- reward: dense shaped reward (verticality^2 × multiplier) ---
        let reward = verticality * verticality * self.multiplier;

        let done = self.balance_time > 30.0 || verticality < 0.02;
        (obs(self), reward, done)
    }

    pub fn obs(&self) -> Obs {
        let h1 = -self.t1.cos();
        let h2 = -self.t2.cos();
        let verticality = ((h1 + h2) * 0.5).clamp(0.0, 1.0);
        Obs {
            t1: self.t1, t2: self.t2, w1: self.w1, w2: self.w2,
            cart_x: self.cart_x / CART_LIMIT,
            cart_v: self.cart_v,
            verticality,
            multiplier: self.multiplier,
            score: self.score,
        }
    }
}

fn obs(e: &Env) -> Obs {
    e.obs()
}

fn rk4(t1: f32, t2: f32, w1: f32, w2: f32, pivot_a: f32, assist: u8, dt: f32) -> (f32, f32, f32, f32) {
    let (k1t1, k1t2, k1w1, k1w2) = deriv(t1, t2, w1, w2, pivot_a, assist);
    let (k2t1, k2t2, k2w1, k2w2) = deriv(t1 + 0.5*dt*k1t1, t2 + 0.5*dt*k1t2,
                                         w1 + 0.5*dt*k1w1, w2 + 0.5*dt*k1w2, pivot_a, assist);
    let (k3t1, k3t2, k3w1, k3w2) = deriv(t1 + 0.5*dt*k2t1, t2 + 0.5*dt*k2t2,
                                         w1 + 0.5*dt*k2w1, w2 + 0.5*dt*k2w2, pivot_a, assist);
    let (k4t1, k4t2, k4w1, k4w2) = deriv(t1 + dt*k3t1, t2 + dt*k3t2,
                                         w1 + dt*k3w1, w2 + dt*k3w2, pivot_a, assist);
    (
        t1 + dt/6.0 * (k1t1 + 2.0*k2t1 + 2.0*k3t1 + k4t1),
        t2 + dt/6.0 * (k1t2 + 2.0*k2t2 + 2.0*k3t2 + k4t2),
        w1 + dt/6.0 * (k1w1 + 2.0*k2w1 + 2.0*k3w1 + k4w1),
        w2 + dt/6.0 * (k1w2 + 2.0*k2w2 + 2.0*k3w2 + k4w2),
    )
}

fn deriv(t1: f32, t2: f32, w1: f32, w2: f32, pivot_a: f32, assist: u8) -> (f32, f32, f32, f32) {
    let d = t1 - t2;
    let sind = d.sin();
    let cosd = d.cos();
    let den = 2.0 - cosd * cosd;
    let mut a1 = (-3.0 * GRAVITY * t1.sin()
                  - GRAVITY * (t1 - 2.0 * t2).sin()
                  - 2.0 * sind * (w2 * w2 + w1 * w1 * cosd)) / den;
    let mut a2 = (sind * (2.0 * w1 * w1 + 2.0 * GRAVITY * t1.cos() + w2 * w2 * cosd)) / den;
    // Pivot acceleration coupling (cart pushes pendulum) — same as game's GPU ODE
    a1 -= pivot_a * t1.cos() / 60.0;
    a2 -= pivot_a * t2.cos() / 60.0;
    // Assist spring (matches game)
    let kp = ASSIST_KP[assist as usize];
    let kd = ASSIST_KD[assist as usize];
    if kp != 0.0 || kd != 0.0 {
        a1 += kp * t1.sin() - kd * w1;
        a2 += kp * t2.sin() - kd * w2;
    }
    // damping
    a1 -= DAMPING * w1;
    a2 -= DAMPING * w2;
    (w1, w2, a1, a2)
}
