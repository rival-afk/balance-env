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

## Physics

- Double pendulum on a horizontal cart (rail-locked, `cart_x` limited to ±130 px, damped cart `damper=3.0`).
- Cart mass 40, unit arms (L=1), unit masses, `g = 9.81 m/s²`.
- Exact sympy-derived Lagrangian equations of motion — match the Godot game 1:1.
- Spawn: pendulum upright with random initial tilt; assist levels `0..4` mirror the in-game difficulty.

## Observation vector

| idx | name | meaning |
|-----|------|---------|
| 0 | `t1` | angle of arm 1, rad (PI = upright) |
| 1 | `t2` | angle of arm 2, rad |
| 2 | `w1` | angular velocity, arm 1 |
| 3 | `w2` | angular velocity, arm 2 |
| 4 | `cart_x` | cart position (normalised to ±1) |
| 5 | `cart_v` | cart velocity, px/s |
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
