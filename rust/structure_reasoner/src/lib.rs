use pyo3::prelude::*;

#[pyfunction]
fn placeholder() -> PyResult<&'static str> {
    Ok("structure_reasoner placeholder")
}

#[pymodule]
fn structure_reasoner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(placeholder, m)?)?;
    Ok(())
}
