//! PyO3-модуль Balance Env — headless окружение для обучения нейросетей.
//! 1:1 match с игровой физикой (тот же Лагранжиан, те же константы).

use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

mod env;

#[pyclass]
pub struct Env {
    inner: env::Env,
}

#[pymethods]
impl Env {
    #[new]
    #[pyo3(signature = (seed = None, assist_level = 1))]
    fn new(seed: Option<u64>, assist_level: u8) -> Self {
        Env { inner: env::Env::new(seed, assist_level) }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn set_assist(&mut self, level: u8) {
        self.inner.set_assist(level);
    }

    /// action ∈ {-1, 0, +1}: -1 = двигать платформу влево, +1 = вправо.
    /// Возвращает (obs, reward, done).
    fn step(&mut self, action: i8) -> PyResult<(PyObject, f32, bool)> {
        let dt = 1.0 / 60.0;
        let (obs, reward, done) = self.inner.step(action, dt);
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::new(py, obs.as_array().iter().map(|&x| x))?;
            Ok((list.into(), reward, done))
        })
    }

    fn obs(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::new(py, self.inner.obs().as_array().iter().map(|&x| x))?;
            Ok(list.into())
        })
    }

    fn __repr__(&self) -> String {
        format!("BalanceEnv(assist={}, score={:.3})", self.inner.assist_level, self.inner.score.score)
    }
}

#[pyclass]
pub struct BatchEnv {
    envs: Vec<env::Env>,
}

#[pymethods]
impl BatchEnv {
    #[new]
    #[pyo3(signature = (n, seed = None, assist_level = 1))]
    fn new(n: usize, seed: Option<u64>, assist_level: u8) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed(rand::random::<[u8; 32]>()),
        };
        let envs = (0..n)
            .map(|_| env::Env::new(Some(rng.gen()), assist_level))
            .collect();
        BatchEnv { envs }
    }

    /// actions: список действий (-1, 0, +1) длиной n.
    /// Возвращает (obs_list, rewards_list, dones_list) — батч-шаг,
    /// векторизованный rayon across env's (максимум CPU).
    fn step(&mut self, actions: Vec<i8>) -> PyResult<(PyObject, PyObject, PyObject)> {
        if actions.len() != self.envs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("expected {} actions, got {}", self.envs.len(), actions.len())));
        }
        let dt = 1.0 / 60.0;
        let results: Vec<([f32; 9], f32, bool)> = self.envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(e, &a)| e.step(a, dt))
            .map(|(o, r, d)| (o.as_array(), r, d))
            .collect();
        let mut obs_list = Vec::with_capacity(results.len());
        let mut rew_list = Vec::with_capacity(results.len());
        let mut done_list = Vec::with_capacity(results.len());
        for (o, r, d) in results {
            obs_list.push(o.to_vec());
            rew_list.push(r);
            done_list.push(d);
        }
        Python::with_gil(|py| {
            let obs_list_obj = pyo3::types::PyList::new(
                py,
                obs_list.iter().map(|o| o.to_vec()))?;
            let rew_list_obj = pyo3::types::PyList::new(
                py,
                rew_list.into_iter().map(|r| r))?;
            let done_list_obj = pyo3::types::PyList::new(
                py,
                done_list.into_iter().map(|d| d))?;
            Ok((obs_list_obj.into(), rew_list_obj.into(), done_list_obj.into()))
        })
    }

    fn reset(&mut self) {
        self.envs.par_iter_mut().for_each(|e| e.reset());
    }

    fn __len__(&self) -> usize {
        self.envs.len()
    }
}

#[pymodule]
fn balance_env(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<BatchEnv>()?;
    Ok(())
}
