#![feature(test)]
extern crate test;

use image::*;
use pyo3::prelude::*;
use std::ffi::{c_void, CStr, CString};

// for libjpeg-turbo
use turbojpeg::{Decompressor, Image, PixelFormat};

// internal lib
mod dlpack;
mod tensor;
use dlpack::{DLManagedTensor};

#[pyfunction]
pub fn read_image(file_path: String) -> (Vec<u8>, Vec<usize>) {
    let img: image::DynamicImage = image::open(file_path).unwrap();
    let new_shape: Vec<usize> = Vec::from([img.height() as usize, img.width() as usize, 3]);
    // TODO: check the line below since it copies and we might want
    // to just pass a pointer to the data.
    let new_data: Vec<u8> = img.to_rgb8().to_vec();
    // NOTE: are this two things the same ? We should benchmark.
    //let buf_data: &[u8] = img.as_bytes();
    //let new_data: Vec<u8> = (*buf_data).iter().cloned().collect();
    (new_data, new_shape)
}

fn _read_image_jpeg(file_path: String) -> Result<(), Box<dyn std::error::Error>> {
    // get the JPEG data
    let jpeg_data = std::fs::read(file_path)?;

    // initialize a Decompressor   
    let mut decompressor = Decompressor::new()?;

    // read the JPEG header with image size
    let header = decompressor.read_header(&jpeg_data)?;
    let (width, height) = (header.width, header.height);

    // prepare a storage for the raw pixel data
    let mut pixels = vec![0; 3*width*height];
    let image = Image {
        pixels: pixels.as_mut_slice(),
        width: width,
        pitch: 3 * width, // we use no padding between rows
        height: height,
        format: PixelFormat::RGB,
    };

    // decompress the JPEG data 
    decompressor.decompress_to_slice(&jpeg_data, image)?;

    // use the raw pixel data
    println!("{:?}", &pixels[0..9]);
    Ok(())
}

#[pyfunction]
pub fn read_image_jpeg(file_path: String) {
    let out = _read_image_jpeg(file_path);
}

// desctructor function for the python capsule
unsafe extern "C" fn destructor(o: *mut pyo3::ffi::PyObject) {
    println!("PyCapsule destructor");

    let name = CString::new("dltensor").unwrap();

    let ptr = pyo3::ffi::PyCapsule_GetName(o);
    let current_name = CStr::from_ptr(ptr);
    println!("Expected Name: {:?}", name);
    println!("Current Name: {:?}", current_name);

    if current_name != name.as_c_str() {
        return;
    }

    let ptr = pyo3::ffi::PyCapsule_GetPointer(
        o, name.as_ptr()) as *mut DLManagedTensor;
    (*ptr).deleter.unwrap()(ptr);

    println!("Delete by Python");
}

#[pyfunction]
pub fn read_image_dlpack(file_path: String) -> PyResult<*mut pyo3::ffi::PyObject> {
    // decode image
    let (data, shape) = read_image(file_path);
    // create kornia tensor
    let mut img_t = tensor::cv::Tensor {
        shape: shape,
        data: data,
    };
    // create dlpack managed tensor
    let dlm_tensor = img_t.to_dlpack();
    let name = CString::new("dltensor").unwrap();
    // create python capsule
    let ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            Box::into_raw(dlm_tensor) as *mut c_void,
            name.as_ptr(),
            Some(destructor as pyo3::ffi::PyCapsule_Destructor),
        )
    };
    Ok(ptr)
}

#[pyfunction]
pub fn show_image_from_file(file_path: String) {
    vviz::app::spawn(vviz::app::VVizMode::Local, | mut manager: vviz::manager::Manager| {
        let img: image::DynamicImage = image::open(file_path.clone()).unwrap();
        manager.add_widget2("img".to_string(), img);
        manager.sync_with_gui();
    });
}

#[pyfunction]
pub fn show_image_from_raw(data: Vec<u8>, shape: Vec<usize>) {
    vviz::app::spawn(vviz::app::VVizMode::Local, | mut manager: vviz::manager::Manager| {
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
    m.add_function(wrap_pyfunction!(read_image_jpeg, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(show_image_from_raw, m)?)?;
    m.add_class::<tensor::cv::Tensor>()?;
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
