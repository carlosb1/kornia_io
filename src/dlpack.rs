// implement the dlpack data structure
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
pub mod dlpack {

    use pyo3::prelude::*;
    use std::os::raw::c_void;

    pub enum DLDeviceType {
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

    pub struct DLDevice {
        pub device_type: DLDeviceType,
        pub device_id: i32,
    }

    pub enum DLDataTypeCode {
        kDLInt,
        kDLUInt,
        kDLFloat,
        kDLOpaqueHandle,
        kDLBfloat,
        kDLComplex,
    }

    pub struct DLDataType {
        pub code: u8,
        pub bits: u8,
        pub lanes: u16,
    }

    pub struct DLTensor {
        pub data: Vec<u8>,
        pub device: DLDevice,
        pub ndim: u32,
        pub dtype: DLDataType,
        pub shape: Vec<i64>,
        pub strides: Vec<i64>,
        pub byte_offset: u64,
    }
    impl DLTensor {
        pub fn new() -> Self {
            DLTensor {
                data: Vec::new(),
                device: DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 1,
                },
                ndim: 1,
                dtype: DLDataType {
                    code: 1,
                    bits: 1,
                    lanes: 1,
                },
                shape: Vec::new(),
                strides: Vec::new(),
                byte_offset: 1,
            }
        }
    }

    #[pyclass]
    pub struct DLManagedTensor {
        pub dl_tensor: DLTensor,
        //pub manager_ctx: *mut c_void,
    }

    impl DLManagedTensor {
        pub fn new() -> Self {
            DLManagedTensor {
                dl_tensor: DLTensor::new(),
            }
        }
    }
} // namespace dlpack
