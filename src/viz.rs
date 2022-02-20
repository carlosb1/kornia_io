use pyo3::prelude::*;
use image;//TODO: import what you use
// internal libs
use tensor;


#[pyclass(unsendable)]
pub struct VizManager {
    pub manager: vviz::manager::Manager,
}

#[pymethods]
impl VizManager {
    #[new]
    pub fn new() -> Self {
        VizManager {
            manager: vviz::manager::Manager::new_remote()
        }
    }

    // TODO: support later our tensor
    // TODO: tensor_to_image
    //pub fn add_image(&mut self, window_name: String, data: Vec<u8>, shape: Vec<usize>) {
    pub fn add_image(&mut self, window_name: String, image: cv::Tensor) {
        let (data, shape) = (image.data, image.shape);
        let (_w, _h, _ch) = (shape[0], shape[1], shape[2]);
        let buf: image::RgbImage = image::ImageBuffer::from_raw(
            _w as u32, _h as u32, data).unwrap();
        let img = image::DynamicImage::from(
            image::DynamicImage::ImageRgb8(buf));
        self.manager.add_widget2(window_name, img.into_rgba8());
    }

    pub fn show(&mut self) {
        loop {
            self.manager.sync_with_gui();
        }
    }

}

#[pyfunction]
pub fn show_image_from_file(file_path: String) {
    vviz::app::spawn(vviz::app::VVizMode::Local, move | mut manager: vviz::manager::Manager| {
        let img: image::DynamicImage = image::open(file_path.clone()).unwrap();
        manager.add_widget2("img".to_string(), img.into_rgba8());
        manager.sync_with_gui();
    });
}

#[pyfunction]
pub fn show_image_from_tensor(image: cv::Tensor) {
    vviz::app::spawn(vviz::app::VVizMode::Local, move | mut manager: vviz::manager::Manager| {
        let (data, shape) = (image.data, image.shape);
        let (_w, _h, _ch) = (shape[0], shape[1], shape[2]);
        let buf: RgbImage = image::ImageBuffer::from_raw(
            _w as u32, _h as u32, data).unwrap();
        let img = image::DynamicImage::from(
            image::DynamicImage::ImageRgb8(buf));
        manager.add_widget2("img".to_string(), img.into_rgba8());
        manager.sync_with_gui();
    });
}