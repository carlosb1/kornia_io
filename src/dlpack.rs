// implement the dlpack data structure
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
pub mod dlpack {

    use std::os::raw::c_void;

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

    pub struct DLDevice {
        pub device_type: DLDeviceType,
        pub device_id: i32,
    }

    enum DLDataTypeCode {
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

    #[pyclass]
    pub struct DLManagedTensor {
        pub dl_tensor: DLTensor,
        pub manager_ctx: *mut c_void,
    }
} // namespace dlpack
