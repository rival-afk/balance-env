# balance-env

Headless Balance environment for neural network training, implemented as a Python extension module in Rust. Exact port of the Godot game's double-pendulum-on-cart physics (sympy-derived Lagrangian, energy-conserving to ~1e-12).

## Install

```bash
pip install .
```

Or build as a wheel:

```bash
python -m maturin build --release
pip install target/wheels/balance_env-*-cp310-abi3-*.whl
```

Requires Python >= 3.10.

## Quick start

```python
import balance_env

env = balance_env.Env(seed=42, assist_level=1)
obs, reward, done = env.step(1)   # push right
obs, reward, done = env.step(0)   # coast
print(obs)                         # [t1, t2, w1, w2, cart_x, cart_v, verticality, multiplier, score]
```

## BatchEnv for RL training

```python
import balance_env

envs = balance_env.BatchEnv(n=64, seed=None, assist_level=0)
obs, reward, done = envs.step([1] * 64)
```

Each `step` returns three parallel lists: observations, rewards, done flags.
CPU stepping is parallelised with rayon (all cores). Expect ~1.8M env-steps/s at large batch sizes.

## GpuBatchEnv — compute on the GPU

```python
import balance_env

if balance_env.GpuBatchEnv.available():
    envs = balance_env.GpuBatchEnv(n=4096, seed=None, assist_level=0)
    obs, reward, done = envs.step([1] * 4096)
```

The physics (Lagrangian ODE, RK4, rail clamp) runs in a WGSL compute shader via wgpu — one invocation per environment. Scoring/observations/rewards stay on the CPU (rayon) and are shared with the single-thread env, so reward semantics are identical to `Env`/`BatchEnv`. Same-seed batches step bit-identically on this machine.

Falls back to a clear `RuntimeError` if no suitable GPU adapter is found (e.g. in CI); `available()` reports support ahead of time. Combine with `BatchEnv` for CPU-only machines.

## Physics

- Double pendulum on a horizontal cart (rail-locked, `cart_x` limited to ±2.2 m, damped cart `damper=3.0`).
- SI units: cart mass 12 kg, uniform 1 m arms of 1 kg each, `g = 9.81 m/s²`, peak drive force 70 N (≈5 m/s²).
- Same numbers drive the Godot game, which draws 1 m as 120 px (±2.2 m = ±264 px on screen).
- Exact sympy-derived Lagrangian equations of motion (plat.txt §3): `A(q)·q̈ = r`, with joint friction and the assist torque inside `r`.
- RK4 at a fixed 240 Hz (4 sub-steps per 60 Hz action step); energy drift ≈2e-6 per 10 s.
- Same ODE in both backends: CPU (Rust + rayon) and GPU (WGSL compute via wgpu).
- Spawn: pendulum upright with random initial tilt; assist levels `0..4` mirror the in-game difficulty.

## Observation vector

| idx | name | meaning |
|-----|------|---------|
| 0 | `t1` | angle of arm 1, rad (PI = upright) |
| 1 | `t2` | angle of arm 2, rad |
| 2 | `w1` | angular velocity, arm 1 |
| 3 | `w2` | angular velocity, arm 2 |
| 4 | `cart_x` | cart position (normalised to ±1) |
| 5 | `cart_v` | cart velocity, m/s |
| 6 | `verticality` | 0 (fallen) … 1 (full upright) |
| 7 | `multiplier` | score multiplier (1.0 … 5.0) |
| 8 | `score` | episode reward accumulator |

## Assistance levels

| level | name | behaviour |
|-------|------|-----------|
| 0 | Нет | pure physics, no assist |
| 1 | Лёгкий | light spring |
| 2 | Средний | moderate assist |
| 3 | Сильный | strong assist |
| 4 | Максимум | pole essentially never falls |

## License

MIT
