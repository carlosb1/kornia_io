use crate::dlpack::*;
use crate::tensor::cv;
use pyo3::prelude::*;
use std::ffi::{c_void, CStr, CString};

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

    let ptr = pyo3::ffi::PyCapsule_GetPointer(o, name.as_ptr()) as *mut DLManagedTensor;
    (*ptr).deleter.unwrap()(ptr);

    println!("Delete by Python");
}

unsafe extern "C" fn deleter(x: *mut DLManagedTensor) {
    println!("DLManagedTensor deleter");

    let ctx = (*x).manager_ctx as *mut cv::Tensor;
    ctx.drop_in_place();
    //(*x).dl_tensor.shape.drop_in_place();
    //(*x).dl_tensor.strides.drop_in_place();
    x.drop_in_place();
}

#[pyfunction]
pub fn cvtensor_to_dlpack(tensor: cv::Tensor) -> PyResult<*mut pyo3::ffi::PyObject> {
    let mut tensor_bx = Box::new(tensor);
    // create dlpack managed tensor
    let dl_tensor = DLTensor {
        data: tensor_bx.data.as_mut_ptr() as *mut c_void,
        device: DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: tensor_bx.shape.len() as u32,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLFloat as u8,
            bits: 32,
            lanes: 1,
        },
        shape: tensor_bx.shape.iter().map(|&x| x as i64).collect(),
        strides: vec![0, tensor_bx.shape.len() as i64],
        byte_offset: 0,
    };
    let dlm_tensor = DLManagedTensor {
        dl_tensor,
        manager_ctx: Box::into_raw(tensor_bx) as *mut c_void,
        deleter: Some(deleter),
    };
    let dlm_tensor_bx = Box::new(dlm_tensor);
    let name = CString::new("dltensor").unwrap();
    // create python capsule
    let ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            Box::into_raw(dlm_tensor_bx) as *mut c_void,
            name.as_ptr(),
            Some(destructor as pyo3::ffi::PyCapsule_Destructor),
        )
    };
    Ok(ptr)
}