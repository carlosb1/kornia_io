#![feature(test)]
extern crate test;

use image::*;
use pyo3::prelude::*;

#[pyfunction]
fn my_fcn() -> () {
    println!("hello world");
}

#[pyfunction]
fn read_image(file_path: String) -> (Vec<u8>, Vec<usize>) {
    let img: image::DynamicImage = image::open(file_path).unwrap();
    let new_shape: Vec<usize> = Vec::from([img.height() as usize, img.width() as usize, 3]);
    let new_data: Vec<u8> = img.to_rgb8().to_vec();
    (new_data, new_shape)
}

#[pymodule]
pub fn kornia_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(my_fcn, m)?)?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::SystemTime;
    use test::Bencher;

    #[test]
    fn load() {
        let PATH: PathBuf = [env!("CARGO_MANIFEST_DIR"), "clients", "test.jpg"]
            .iter()
            .collect();

        let str_path = PATH.into_os_string().into_string().unwrap();
        let start = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("get millis error");
        let info = read_image(str_path.clone());
        let end = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("get millis error");
        println!("{}", str_path);
        println!("{:?}", info.1);
        println!(
            "time {:?} secs",
            (end.as_millis() - start.as_millis()) as f64 / 1000.,
        );
    }

    #[bench]
    fn bench(b: &mut Bencher) {
        let PATH: PathBuf = [env!("CARGO_MANIFEST_DIR"), "clients", "test.jpg"]
            .iter()
            .collect();
        let str_path = PATH.into_os_string().into_string().unwrap();
        b.iter(|| {
            let info = read_image(str_path.clone());
        });
    }
}
