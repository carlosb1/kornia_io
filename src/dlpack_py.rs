use crate::dlpack::DLManagedTensor;
use crate::io;
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

#[pyfunction]
pub fn read_image_dlpack(file_path: String) -> PyResult<*mut pyo3::ffi::PyObject> {
    // decode image
    let mut img_t: cv::Tensor = io::read_image_jpeg(file_path);
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
