use pyo3::prelude::*;
use image::*;
use std::path::Path;

#[pyfunction]
fn my_fcn() -> () {
    println!("hello world");
}

#[pyfunction]
fn read_image(file_path: String) -> (Vec<u8>, Vec<usize>) {
    let img: image::DynamicImage = image::open(Path::new(&file_path)).unwrap();
    let new_shape: Vec<usize> = Vec::from([1, 3, img.height() as usize, img.width() as usize]);
    let new_data: Vec<u8> = img.to_rgb8().to_vec();
    (new_data, new_shape)
}

#[pymodule]
pub fn kornia_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(my_fcn, m)?)?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    Ok(())
}
