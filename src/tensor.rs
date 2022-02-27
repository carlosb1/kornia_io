pub mod cv {

    use crate::dlpack::{
        DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor,
    };
    use pyo3::prelude::*;
    use std::ffi::c_void;

    //unsafe extern "C" fn deleter(x: *mut DLManagedTensor) {
    //    println!("DLManagedTensor deleter");

    //    let ctx = (*x).manager_ctx as *mut Tensor;
    //    ctx.drop_in_place();
    //    //(*x).dl_tensor.shape.drop_in_place();
    //    //(*x).dl_tensor.strides.drop_in_place();
    //    x.drop_in_place();
    //}

    #[pyclass]
    #[derive(Debug, Clone, PartialEq)]
    pub struct Tensor {
        #[pyo3(get)]
        pub shape: Vec<i64>,
        #[pyo3(get)]
        pub data: Vec<u8>,
        #[pyo3(get)]
        pub strides: Vec<i64>,
    }

    #[pymethods]
    impl Tensor {
        #[new]
        pub fn new(shape: Vec<i64>, data: Vec<u8>) -> Self {
            let num_strides = shape.len() as i64;
            Tensor {
                shape: shape,
                data: data,
                strides: vec![0, num_strides],
            }
        }
    }

    impl Tensor {
        // TODO: something is wrong with the context
        //pub fn to_dlpack(&mut self) -> Box<DLManagedTensor> {
        //    let tensor_bx = Box::new(self);
        //    let dl_tensor = DLTensor {
        //        data: tensor_bx.data.as_mut_ptr() as *mut c_void,
        //        device: DLDevice {
        //            device_type: DLDeviceType::kDLCPU,
        //            device_id: 0,
        //        },
        //        ndim: tensor_bx.shape.len() as u32,
        //        dtype: DLDataType {
        //            code: DLDataTypeCode::kDLFloat as u8,
        //            bits: 32,
        //            lanes: 1,
        //        },
        //        shape: tensor_bx.shape.iter().map(|&x| x as i64).collect(),
        //        strides: vec![0, tensor_bx.shape.len() as i64],
        //        byte_offset: 0,
        //    };

        //    let dlm_tensor = DLManagedTensor {
        //        dl_tensor,
        //        manager_ctx: Box::into_raw(tensor_bx) as *mut c_void,
        //        deleter: Some(deleter),
        //    };
        //    let ptr = Box::new(dlm_tensor);
        //    ptr
        //}

        //pub fn dims(&self) -> usize {
        //    self.data.len()
        //}

        //pub fn get(&self, i0: usize, i1: usize, i2: usize, i3: usize) -> u8 {
        //    let i = i0 * self.shape[1] * self.shape[2] * self.shape[3];
        //    let j = i1 * self.shape[2] * self.shape[3];
        //    let k = i2 * self.shape[3];
        //    self.data[i + j + k + i3]
        //}
        //pub fn add(&self, other: Tensor) -> Tensor {
        //    let mut data: Vec<u8> = self.data.clone();
        //    for i in 0..data.len() {
        //        data[i] += other.data[i];
        //    }
        //    Tensor {
        //        shape: self.shape.clone(),
        //        data: data,
        //    }
        //}

        //pub fn mul(&self, other: Tensor) -> Tensor {
        //    let mut data: Vec<u8> = self.data.clone();
        //    for i in 0..data.len() {
        //        data[i] *= other.data[i];
        //    }
        //    Tensor {
        //        shape: self.shape.clone(),
        //        data: data,
        //    }
        //}

        //pub fn subs(&self, other: Tensor) -> Tensor {
        //    let mut data: Vec<u8> = self.data.clone();
        //    for i in 0..data.len() {
        //        data[i] -= other.data[i];
        //    }
        //    Tensor {
        //        shape: self.shape.clone(),
        //        data: data,
        //    }
        //}

        //pub fn div(&self, other: Tensor) -> Tensor {
        //    let mut data: Vec<u8> = self.data.clone();
        //    for i in 0..data.len() {
        //        data[i] /= other.data[i];
        //    }
        //    Tensor {
        //        shape: self.shape.clone(),
        //        data: data,
        //    }
        //}

        //pub fn print(&self) -> () {
        //    for i in 0..self.shape[0] {
        //        for j in 0..self.shape[1] {
        //            for k in 0..self.shape[2] {
        //                for l in 0..self.shape[3] {
        //                    println!("Index: ({}, {}, {}, {}):", i, j, k, l);
        //                    println!("Val: {:?}", self.get(i, j, k, l));
        //                    println!("---------");
        //                }
        //            }
        //        }
        //    }
        //}
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
} // namespace cv

#[cfg(test)]
mod tests {
    use super::*;

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
