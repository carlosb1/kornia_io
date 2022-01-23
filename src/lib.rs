//#![feature(test)]
//extern crate test;

use image::*;
use pyo3::prelude::*;

#[pyfunction]
pub fn read_image(file_path: String) -> (Vec<u8>, Vec<usize>) {
    let img: image::DynamicImage = image::open(file_path).unwrap();
    let new_shape: Vec<usize> = Vec::from([img.height() as usize, img.width() as usize, 3]);
    let new_data: Vec<u8> = img.to_rgb8().to_vec();
    (new_data, new_shape)
}

#[pyfunction]
pub fn show_image_from_file(file_path: String) {
    vviz::app::spawn(move |mut manager: vviz::manager::Manager| {
        let img: image::DynamicImage = image::open(file_path.clone()).unwrap();
        manager.add_widget2("img".to_string(), img);
        manager.sync_with_gui();
    });
}

#[pyfunction]
pub fn show_image_from_raw(data: Vec<u8>, shape: Vec<usize>) {
    vviz::app::spawn(move |mut manager: vviz::manager::Manager| {
        let height = shape[0] as u32;
        let width = shape[1] as u32;
        let buf: RgbImage = image::ImageBuffer::from_raw(width, height, data).unwrap();
        let img = image::DynamicImage::from(image::DynamicImage::ImageRgb8(buf));
        manager.add_widget2("img".to_string(), img);
        manager.sync_with_gui();
    });
}

#[pymodule]
pub fn kornia_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_raw, m)?)?;
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
