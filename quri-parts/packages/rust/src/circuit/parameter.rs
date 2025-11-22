use pyo3::prelude::*;

#[pyclass(eq, get_all, ord, module = "quri_parts.rust.circuit.parameter")]
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Parameter {
    name: String,
    id: u64,
}

impl core::fmt::Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

#[pymethods]
impl Parameter {
    #[new]
    #[pyo3(signature = (name = String::new()))]
    pub fn new(name: String) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::OnceLock;

        // The `Parameter` may be sustained beyond processes with (un-)pickling. To prevent
        // overlapping, here we should use random number unique for processes.
        static RANDOM: OnceLock<u64> = OnceLock::new();

        let random = RANDOM.get_or_init(|| {
            use std::hash::{BuildHasher, Hasher};
            std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish()
        });

        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let count = COUNTER
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| Some(x + 1))
            .unwrap();

        Parameter {
            name: name.into(),
            id: random ^ count,
        }
    }

    #[getter]
    pub(crate) fn get_name(&self) -> String {
        self.name.clone().into()
    }

    #[setter]
    fn set_name(&mut self, s: String) {
        self.name = s.into();
    }

    #[pyo3(name = "__repr__")]
    fn py_repr(&self) -> String {
        format!("Parameter(name={})", &self.name)
    }

    #[pyo3(name = "__hash__")]
    fn py_hash(slf: Bound<'_, Self>) -> u64 {
        slf.borrow().id
    }

    #[pyo3(name = "__getstate__")]
    fn py_getstate(slf: Bound<'_, Self>) -> (String, u64) {
        let borrowed = slf.borrow();
        (borrowed.name.clone(), borrowed.id)
    }

    #[pyo3(name = "__setstate__")]
    fn py_setstate(slf: Bound<'_, Self>, state: (String, u64)) {
        let mut borrowed = slf.borrow_mut();
        borrowed.name = state.0;
        borrowed.id = state.1;
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "parameter")?;
    m.add_class::<Parameter>()?;
    Ok(m)
}
