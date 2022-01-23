use image::GenericImageView;
use std::path::Path;

use pyo3::prelude::*;

use std::os::raw::c_void;

// implement the dlpack data structure 
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

#[pyclass]
enum DLDeviceType {
  kDLCPU,
  kDLCUDA,
  kDLCUDAHost,
  kDLOpenCL,
  kDLVulkan,
  kDLMetal,
  kDLVPI,
  kDLROCM,
  kDLROCMHost,
  kDLExtDev,
  kDLCUDAManaged,
  kDLOneAPI,
  kDLWebGPU,
  kDLHexagon,
}

#[pyclass]
pub struct DLDevice {
  pub device_type: DLDeviceType,
  pub device_id: i32,
}

#[pyclass]
enum DLDataTypeCode {
  kDLInt,
  kDLUInt,
  kDLFloat,
  kDLOpaqueHandle,
  kDLBfloat,
  kDLComplex,
}

#[pyclass]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

enum void_ptr {}

#[pyclass]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: u32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,

}

#[pyclass]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub data: Vec<u8>,
}

#[pymethods]
impl Tensor {
    #[new]
    pub fn new(shape: Vec<usize>, data: Vec<u8>) -> Self {
        Tensor {
            shape: shape,
            data: data,
        }
    }
    #[getter]
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
