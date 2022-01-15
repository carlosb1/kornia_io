pub mod cv {
    use image::GenericImageView;
    use std::path::Path;

    #[derive(Debug, Clone, PartialEq)]
    pub struct Tensor {
        pub shape: Vec<usize>,
        pub data: Vec<u8>,
    }

    impl Tensor {
        pub fn new(shape: Vec<usize>, data: Vec<u8>) -> Self {
            Tensor {
                shape: shape,
                data: data,
            }
        }

        pub fn dims(&self) -> usize {
            self.data.len()
        }

        pub fn get(&self, i0: usize, i1: usize, i2: usize, i3: usize) -> u8 {
            let i = i0 * self.shape[1] * self.shape[2] * self.shape[3];
            let j = i1 * self.shape[2] * self.shape[3];
            let k = i2 * self.shape[3];
            self.data[i + j + k + i3]
        }
        pub fn add(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] += other.data[i];
            }
            Tensor {
                shape: self.shape.clone(),
                data: data,
            }
        }

        pub fn mul(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] *= other.data[i];
            }
            Tensor {
                shape: self.shape.clone(),
                data: data,
            }
        }

        pub fn subs(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] -= other.data[i];
            }
            Tensor {
                shape: self.shape.clone(),
                data: data,
            }
        }

        pub fn div(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] /= other.data[i];
            }
            Tensor {
                shape: self.shape.clone(),
                data: data,
            }
        }

        pub fn from_file(file_path: &str) -> Tensor {
            let img: image::DynamicImage = image::open(&Path::new(file_path)).unwrap();
            let new_shape = Vec::from([1, 3, img.height() as usize, img.width() as usize]);
            let new_data: Vec<u8> = img.to_rgb8().to_vec();
            Tensor {
                shape: new_shape,
                data: new_data,
            }
        }

        pub fn print(&self) -> () {
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    for k in 0..self.shape[2] {
                        for l in 0..self.shape[3] {
                            println!("Index: ({}, {}, {}, {}):", i, j, k, l);
                            println!("Val: {:?}", self.get(i, j, k, l));
                            println!("---------");
                        }
                    }
                }
            }
        }
    }

    pub fn cumsum(data: &Vec<usize>) -> usize {
        let mut acc: usize = 0;
        for x in data {
            acc += x;
        }
        return acc;
    }

    pub fn cumprod(data: &Vec<usize>) -> usize {
        let mut acc: usize = 1;
        for x in data {
            acc *= x;
        }
        return acc;
    }
}

use pyo3::prelude::*;
use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn my_fcn() -> () {
    println!("hello world");
}

#[pymodule]
pub fn kornia_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(my_fcn, m)?)?;
    Ok(())
}

/*
#[pyclass]
#[derive(Clone)]
pub struct WrapperTensor {
    _in: cv::Tensor,
}
//TODO Add work via reference
#[pymethods]
impl WrapperTensor {
    #[new]
    fn new(shape: Vec<usize>, data: Vec<u8>) -> Self {
        WrapperTensor {
            _in: cv::Tensor::new(shape, data),
        }
    }

    fn add(&self, other: WrapperTensor) -> PyResult<WrapperTensor> {
        Ok(WrapperTensor {
            _in: self._in.add(other._in),
        })
    }
}
*/

#[cfg(test)]
mod tests {
    use crate::cv;

    #[test]
    fn add() {
        let shape: Vec<usize> = vec![1, 1, 2, 2];
        let data: Vec<u8> = (0..cv::cumprod(&shape)).map(|x| x as u8).collect();
        let t1 = cv::Tensor::new(shape.clone(), data);
        let t2 = t1.clone();
        let t3 = t1.add(t2.clone());
        let to_compare = cv::Tensor::new(shape.clone(), vec![0, 2, 4, 6]);
        assert_eq!(t3, to_compare);
    }
}
