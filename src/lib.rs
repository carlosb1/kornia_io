#![feature(test)]
extern crate test;

pub mod dlpack;
pub mod dlpack_py;
pub mod io;
pub mod tensor;
pub mod viz;

use crate::dlpack_py::__pyo3_get_function_read_image_dlpack;
use crate::io::__pyo3_get_function_read_image_jpeg;
use crate::io::__pyo3_get_function_read_image_rs;
use crate::viz::__pyo3_get_function_show_image_from_file;
use crate::viz::__pyo3_get_function_show_image_from_tensor;

use pyo3::prelude::*;

#[pymodule]
pub fn kornia_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_image_rs, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_tensor, m)?)?;
    m.add_class::<viz::VizManager>()?;
    m.add_class::<tensor::cv::Tensor>()?;
    Ok(())
}
