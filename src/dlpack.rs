// implement the dlpack data structure
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

// check this
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/DLConvertor.cpp
use std::ffi::c_void;

pub enum DLDeviceType {
    kDLCPU = 1,
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
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: u32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(self_: *mut DLManagedTensor)>,
}
