// Balance — GPU compute shader for the batched double-pendulum-on-cart.
// 1:1 port of the Rust/Godot ODE (Lagrangian, matrix solve + Cramer).
// One invocation per environment; all state fits in 24 bytes.

struct Params {
	dt: f32,
	n: u32,
}

struct EnvState {
	th1: f32,
	th2: f32,
	w1: f32,
	w2: f32,
	x: f32,
	v: f32,
}

struct Assist {
	kp: f32,
	kd: f32,
}

@group(0) @binding(0) var<storage, read_write> states: array<EnvState>;
@group(0) @binding(1) var<storage, read> actions: array<i32>;
@group(0) @binding(2) var<storage, read> assists: array<Assist>;
@group(0) @binding(3) var<uniform> params: Params;

const GRAVITY: f32 = 9.81;
const M_CART: f32 = 12.0;
const M1: f32 = 1.0;
const M2: f32 = 1.0;
const L1: f32 = 1.0;
const L2: f32 = 1.0;
const ANG_DAMP: f32 = 0.1;      // angular damping on the arms (matches Rust/Godot)
const CART_DAMP: f32 = 3.0;     // cart drag (matches Rust/Godot)
const DRIVE_FORCE: f32 = 70.0;  // N, peak drive force (matches Rust/Godot)
const CART_LIMIT: f32 = 2.2;    // m, same as the game (264 px at 120 px/m)

// Returns (a1, a2, a_cart) with damping and the assist spring applied.
// `cart_v` is the cart velocity of this RK4 stage (used for cart drag).
fn accel(th1: f32, th2: f32, w1: f32, w2: f32, cart_v: f32, force: f32, kp: f32, kd: f32) -> vec3<f32> {
	let d = th1 - th2;
	let cd = cos(d);
	let sd = sin(d);
	let s1 = sin(th1);
	let s2 = sin(th2);
	let c1 = cos(th1);
	let c2 = cos(th2);

	// M(q) * a = b — same Lagrangian system matrix as Rust/Godot.
	let b0 = force - CART_DAMP * cart_v
		+ (M1 + M2) * L1 * s1 * w1 * w1
		+ M2 * L2 * s2 * w2 * w2;
	let b1 = -(M2 * L1 * L2 * sd * w2 * w2 + (M1 + M2) * GRAVITY * L1 * s1);
	let b2 = M2 * L1 * L2 * sd * w1 * w1 - M2 * GRAVITY * L2 * s2;

	let m00 = M_CART + M1 + M2;
	let m01 = (M1 + M2) * L1 * c1;
	let m02 = M2 * L2 * c2;
	let m10 = m01;
	let m11 = (M1 + M2) * L1 * L1;
	let m12 = M2 * L1 * L2 * cd;
	let m20 = m02;
	let m21 = m12;
	let m22 = M2 * L2 * L2;

	let det = m00 * (m11 * m22 - m12 * m21)
			- m01 * (m10 * m22 - m12 * m20)
			+ m02 * (m10 * m21 - m11 * m20);

	let ac = (b0 * (m11 * m22 - m12 * m21) - m01 * (b1 * m22 - m12 * b2) + m02 * (b1 * m21 - m11 * b2)) / det;
	var a1 = (m00 * (b1 * m22 - m12 * b2) - b0 * (m10 * m22 - m12 * m20) + m02 * (m10 * b2 - b1 * m20)) / det;
	var a2 = (m00 * (m11 * b2 - b1 * m21) - m01 * (m10 * b2 - b1 * m20) + b0 * (m10 * m21 - m11 * m20)) / det;

	// angular damping on the arms (same as Rust/Godot)
	a1 -= ANG_DAMP * w1;
	a2 -= ANG_DAMP * w2;

	// assist spring toward vertical (same formula as the CPU path)
	a1 += kp * s1 - kd * w1;
	a2 += kp * s2 - kd * w2;

	return vec3<f32>(a1, a2, ac);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let i = gid.x;
	if (i >= params.n) {
		return;
	}
	let st = states[i];
	let kp = assists[i].kp;
	let kd = assists[i].kd;
	let force = clamp(f32(actions[i]), -1.0, 1.0) * DRIVE_FORCE;
	let dt = params.dt;

	// RK4, full 6-dim — mirrors the CPU integrator step for step.
	let a1k = accel(st.th1, st.th2, st.w1, st.w2, st.v, force, kp, kd);
	let k10 = st.w1;  let k11 = st.w2;  let k12 = a1k.x;  let k13 = a1k.y;  let k14 = st.v;  let k15 = a1k.z;

	let s2_th1 = st.th1 + 0.5 * dt * k10; let s2_th2 = st.th2 + 0.5 * dt * k11;
	let s2_w1  = st.w1  + 0.5 * dt * k12; let s2_w2  = st.w2  + 0.5 * dt * k13;
	let s2_v   = st.v   + 0.5 * dt * k15;
	let a2k = accel(s2_th1, s2_th2, s2_w1, s2_w2, s2_v, force, kp, kd);
	let k20 = s2_w1;  let k21 = s2_w2;  let k22 = a2k.x;  let k23 = a2k.y;  let k24 = s2_v;  let k25 = a2k.z;

	let s3_th1 = st.th1 + 0.5 * dt * k20; let s3_th2 = st.th2 + 0.5 * dt * k21;
	let s3_w1  = st.w1  + 0.5 * dt * k22; let s3_w2  = st.w2  + 0.5 * dt * k23;
	let s3_v   = st.v   + 0.5 * dt * k25;
	let a3k = accel(s3_th1, s3_th2, s3_w1, s3_w2, s3_v, force, kp, kd);
	let k30 = s3_w1;  let k31 = s3_w2;  let k32 = a3k.x;  let k33 = a3k.y;  let k34 = s3_v;  let k35 = a3k.z;

	let s4_th1 = st.th1 + dt * k30; let s4_th2 = st.th2 + dt * k31;
	let s4_w1  = st.w1  + dt * k32; let s4_w2  = st.w2  + dt * k33;
	let s4_v   = st.v   + dt * k35;
	let a4k = accel(s4_th1, s4_th2, s4_w1, s4_w2, s4_v, force, kp, kd);
	let k40 = s4_w1;  let k41 = s4_w2;  let k42 = a4k.x;  let k43 = a4k.y;  let k44 = s4_v;  let k45 = a4k.z;

	var out: EnvState;
	out.th1 = st.th1 + dt * (k10 + 2.0 * k20 + 2.0 * k30 + k40) / 6.0;
	out.th2 = st.th2 + dt * (k11 + 2.0 * k21 + 2.0 * k31 + k41) / 6.0;
	out.w1  = st.w1  + dt * (k12 + 2.0 * k22 + 2.0 * k32 + k42) / 6.0;
	out.w2  = st.w2  + dt * (k13 + 2.0 * k23 + 2.0 * k33 + k43) / 6.0;
	out.x   = st.x   + dt * (k14 + 2.0 * k24 + 2.0 * k34 + k44) / 6.0;
	out.v   = st.v   + dt * (k15 + 2.0 * k25 + 2.0 * k35 + k45) / 6.0;

	// rail clamp (matches the game and the CPU path)
	if (out.x > CART_LIMIT) {
		out.x = CART_LIMIT;
		if (out.v > 0.0) { out.v = 0.0; }
	} else if (out.x < -CART_LIMIT) {
		out.x = -CART_LIMIT;
		if (out.v < 0.0) { out.v = 0.0; }
	}

	states[i] = out;
}
