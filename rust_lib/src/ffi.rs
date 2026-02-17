//! C FFI interface for LanceDB operations.

use std::ffi::{CStr, c_char, c_void};
use std::slice;

use crate::lance_manager::LanceIndex;

pub type LanceHandlePtr = *mut c_void;

#[repr(C)]
pub struct LanceBytes {
    pub data: *mut u8,
    pub len: usize,
}

// ========================================
// Helpers
// ========================================

unsafe fn write_err(err_buf: *mut c_char, err_buf_len: i32, msg: &str) {
    if err_buf.is_null() || err_buf_len <= 0 {
        return;
    }
    let max = (err_buf_len - 1) as usize;
    let bytes = msg.as_bytes();
    let copy_len = bytes.len().min(max);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), err_buf as *mut u8, copy_len);
    *err_buf.add(copy_len) = 0;
}

unsafe fn c_str_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

// ========================================
// Create / Free
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_create_detached(
    db_path: *const c_char,
    dimension: i32,
    metric: *const c_char,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> LanceHandlePtr {
    let db_path_str = c_str_to_string(db_path);
    let metric_str = c_str_to_string(metric);

    match LanceIndex::create(&db_path_str, dimension as usize, &metric_str) {
        Ok(index) => Box::into_raw(Box::new(index)) as LanceHandlePtr,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("create failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_detached(handle: LanceHandlePtr) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut LanceIndex));
    }
}

// ========================================
// Add vectors
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_detached_add(
    handle: LanceHandlePtr,
    vector: *const f32,
    dimension: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i64 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    let vec_slice = slice::from_raw_parts(vector, dimension as usize);

    match h.add_vector(vec_slice) {
        Ok(label) => label,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("add failed: {}", e));
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_detached_add_batch(
    handle: LanceHandlePtr,
    vectors: *const f32,
    num: i32,
    dim: i32,
    out_labels: *mut i64,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    let total = (num as usize) * (dim as usize);
    let vec_data = slice::from_raw_parts(vectors, total);

    match h.add_batch(vec_data, num as usize) {
        Ok(labels) => {
            for (i, label) in labels.iter().enumerate() {
                *out_labels.add(i) = *label;
            }
            labels.len() as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("add_batch failed: {}", e));
            -1
        }
    }
}

// ========================================
// Search
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_detached_search(
    handle: LanceHandlePtr,
    query: *const f32,
    dim: i32,
    k: i32,
    nprobes: i32,
    refine_factor: i32,
    out_labels: *mut i64,
    out_distances: *mut f32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    let query_slice = slice::from_raw_parts(query, dim as usize);

    match h.search(query_slice, k as usize, nprobes as usize, refine_factor as usize) {
        Ok(results) => {
            let n = results.len();
            for (i, (label, dist)) in results.iter().enumerate() {
                *out_labels.add(i) = *label;
                *out_distances.add(i) = *dist;
            }
            n as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("search failed: {}", e));
            -1
        }
    }
}

// ========================================
// Count / Delete
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_detached_count(
    handle: LanceHandlePtr,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i64 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    match h.count() {
        Ok(n) => n as i64,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("count failed: {}", e));
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_detached_delete(
    handle: LanceHandlePtr,
    label: i64,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    match h.delete(label) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("delete failed: {}", e));
            -1
        }
    }
}

// ========================================
// ANN Index / Compact
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_detached_create_index(
    handle: LanceHandlePtr,
    num_partitions: i32,
    num_sub_vectors: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    match h.create_ann_index(num_partitions as u32, num_sub_vectors as u32) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("create_index failed: {}", e));
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_detached_compact(
    handle: LanceHandlePtr,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    match h.compact() {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("compact failed: {}", e));
            -1
        }
    }
}

// ========================================
// Get vector
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_detached_get_vector(
    handle: LanceHandlePtr,
    label: i64,
    out_vec: *mut f32,
    capacity: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    match h.get_vector(label) {
        Ok(vec) => {
            let dim = vec.len();
            if dim > capacity as usize {
                write_err(err_buf, err_buf_len, "output buffer too small");
                return -1;
            }
            std::ptr::copy_nonoverlapping(vec.as_ptr(), out_vec, dim);
            dim as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("get_vector failed: {}", e));
            -1
        }
    }
}

// ========================================
// Serialize / Deserialize (metadata only)
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_detached_serialize_meta(
    handle: LanceHandlePtr,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> LanceBytes {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return LanceBytes {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let h = &*(handle as *mut LanceIndex);
    match h.serialize_meta() {
        Ok(mut blob) => {
            blob.shrink_to_fit();
            let data = blob.as_mut_ptr();
            let len = blob.len();
            std::mem::forget(blob);
            LanceBytes { data, len }
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("serialize_meta failed: {}", e));
            LanceBytes {
                data: std::ptr::null_mut(),
                len: 0,
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_detached_deserialize_meta(
    db_path: *const c_char,
    data: *const u8,
    len: usize,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> LanceHandlePtr {
    if data.is_null() || len == 0 {
        write_err(err_buf, err_buf_len, "null or empty data");
        return std::ptr::null_mut();
    }
    let db_path_str = c_str_to_string(db_path);
    let slice = slice::from_raw_parts(data, len);

    match LanceIndex::deserialize_meta(&db_path_str, slice) {
        Ok(index) => Box::into_raw(Box::new(index)) as LanceHandlePtr,
        Err(e) => {
            write_err(
                err_buf,
                err_buf_len,
                &format!("deserialize_meta failed: {}", e),
            );
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_bytes(bytes: LanceBytes) {
    if !bytes.data.is_null() && bytes.len > 0 {
        drop(Vec::from_raw_parts(bytes.data, bytes.len, bytes.len));
    }
}
