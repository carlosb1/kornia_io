pub mod dlpack;
pub mod io;
pub mod tensor;
pub mod viz;

use pyo3::prelude::*;

#[pymodule]
pub fn kornia_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(io::read_image_rs, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack_py::read_image_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(viz::show_image_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(viz::show_image_from_tensor, m)?)?;
    m.add_class::<viz::VizManager>()?;
    m.add_class::<tensor::cv::Tensor>()?;
    Ok(())
}
