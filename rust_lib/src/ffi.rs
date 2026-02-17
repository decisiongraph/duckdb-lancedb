//! C FFI interface for LanceDB operations.

use std::ffi::{CStr, c_char, c_void};
use std::slice;

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use crate::lance_manager::LanceIndex;

pub type LanceHandlePtr = *mut c_void;

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
// Create / Open / Free
// ========================================

#[no_mangle]
pub unsafe extern "C" fn lance_create_detached(
    db_path: *const c_char,
    dimension: i32,
    metric: *const c_char,
    table_name: *const c_char,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> LanceHandlePtr {
    let db_path_str = c_str_to_string(db_path);
    let metric_str = c_str_to_string(metric);
    let table_name_str = c_str_to_string(table_name);

    match LanceIndex::create(&db_path_str, dimension as usize, &metric_str, &table_name_str) {
        Ok(index) => Box::into_raw(Box::new(index)) as LanceHandlePtr,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("create failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Create a Lance dataset from an Arrow schema (multi-column, zero-copy).
/// `arrow_schema` is a pointer to an ArrowSchema struct describing the data columns
/// (vector + extras). A label column is prepended automatically.
#[no_mangle]
pub unsafe extern "C" fn lance_create_detached_from_arrow(
    db_path: *const c_char,
    arrow_schema: *mut c_void,
    metric: *const c_char,
    table_name: *const c_char,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> LanceHandlePtr {
    let db_path_str = c_str_to_string(db_path);
    let metric_str = c_str_to_string(metric);
    let table_name_str = c_str_to_string(table_name);

    if arrow_schema.is_null() {
        write_err(err_buf, err_buf_len, "null arrow schema");
        return std::ptr::null_mut();
    }

    let schema_ptr = arrow_schema as *mut FFI_ArrowSchema;

    match LanceIndex::create_from_arrow(&db_path_str, schema_ptr, &metric_str, &table_name_str) {
        Ok(index) => Box::into_raw(Box::new(index)) as LanceHandlePtr,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("create_from_arrow failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Open an existing Lance dataset, deriving schema from the table.
#[no_mangle]
pub unsafe extern "C" fn lance_open_detached(
    db_path: *const c_char,
    table_name: *const c_char,
    metric: *const c_char,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> LanceHandlePtr {
    let db_path_str = c_str_to_string(db_path);
    let table_name_str = c_str_to_string(table_name);
    let metric_str = c_str_to_string(metric);

    match LanceIndex::open(&db_path_str, &table_name_str, &metric_str) {
        Ok(index) => Box::into_raw(Box::new(index)) as LanceHandlePtr,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("open failed: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Check if the index has extra columns beyond label + vector.
#[no_mangle]
pub unsafe extern "C" fn lance_detached_has_extra_columns(
    handle: LanceHandlePtr,
) -> i32 {
    if handle.is_null() {
        return 0;
    }
    let h = &*(handle as *mut LanceIndex);
    if h.has_extra_columns() { 1 } else { 0 }
}

/// Get the dimension of vectors in the index.
#[no_mangle]
pub unsafe extern "C" fn lance_detached_dimension(
    handle: LanceHandlePtr,
) -> i32 {
    if handle.is_null() {
        return 0;
    }
    let h = &*(handle as *mut LanceIndex);
    h.dimension() as i32
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_detached(handle: LanceHandlePtr) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut LanceIndex));
    }
}

/// Add a batch of rows via Arrow C Data Interface.
/// `arrow_schema` and `arrow_array` are pointers to ArrowSchema/ArrowArray structs.
/// Fills `out_labels` with assigned labels. Returns count or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn lance_detached_add_batch_arrow(
    handle: LanceHandlePtr,
    arrow_schema: *mut c_void,
    arrow_array: *mut c_void,
    out_labels: *mut i64,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    if arrow_schema.is_null() || arrow_array.is_null() {
        write_err(err_buf, err_buf_len, "null arrow schema/array");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    let schema_ptr = arrow_schema as *mut FFI_ArrowSchema;
    let array_ptr = arrow_array as *mut FFI_ArrowArray;

    match h.add_batch_arrow(schema_ptr, array_ptr) {
        Ok(labels) => {
            for (i, label) in labels.iter().enumerate() {
                *out_labels.add(i) = *label;
            }
            labels.len() as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("add_batch_arrow failed: {}", e));
            -1
        }
    }
}

/// Merge live rows from source index into target index (all in Rust).
/// `live_source_labels` are the labels in source that are not tombstoned.
/// Fills `out_old_labels` and `out_new_labels` with the mapping.
/// Returns count of merged rows or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn lance_detached_merge(
    target_handle: LanceHandlePtr,
    source_handle: LanceHandlePtr,
    live_source_labels: *const i64,
    live_count: i32,
    out_old_labels: *mut i64,
    out_new_labels: *mut i64,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if target_handle.is_null() || source_handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let target = &*(target_handle as *mut LanceIndex);
    let source = &*(source_handle as *mut LanceIndex);
    let live_labels = if live_count > 0 && !live_source_labels.is_null() {
        slice::from_raw_parts(live_source_labels, live_count as usize)
    } else {
        &[]
    };

    match target.merge_from(source, live_labels) {
        Ok(mapping) => {
            for (i, (old_label, new_label)) in mapping.iter().enumerate() {
                *out_old_labels.add(i) = *old_label;
                *out_new_labels.add(i) = *new_label;
            }
            mapping.len() as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("merge failed: {}", e));
            -1
        }
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
    debug_assert_eq!(
        dim as usize,
        h.dimension(),
        "C++ passed dim={} but LanceIndex has dimension={}",
        dim,
        h.dimension()
    );
    let total_floats = num as usize * h.dimension();
    let vec_slice = slice::from_raw_parts(vectors, total_floats);

    match h.add_batch(vec_slice, num as usize) {
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

#[no_mangle]
pub unsafe extern "C" fn lance_detached_delete_batch(
    handle: LanceHandlePtr,
    labels: *const i64,
    count: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    let label_slice = slice::from_raw_parts(labels, count as usize);
    match h.delete_batch(label_slice) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("delete_batch failed: {}", e));
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
pub unsafe extern "C" fn lance_detached_create_hnsw_index(
    handle: LanceHandlePtr,
    m: i32,
    ef_construction: i32,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    match h.create_hnsw_index(m as u32, ef_construction as u32) {
        Ok(()) => 0,
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("create_hnsw_index failed: {}", e));
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
// Bulk vector export
// ========================================

/// Export all vectors from a detached index.
/// Returns count of vectors. Fills out_labels and out_vectors (caller-allocated).
/// out_vectors must hold count * dimension floats.
/// Call with null out_labels/out_vectors to get the count first.
#[no_mangle]
pub unsafe extern "C" fn lance_detached_get_all_vectors(
    handle: LanceHandlePtr,
    out_labels: *mut i64,
    out_vectors: *mut f32,
    out_count: *mut i64,
    err_buf: *mut c_char,
    err_buf_len: i32,
) -> i32 {
    if handle.is_null() {
        write_err(err_buf, err_buf_len, "null handle");
        return -1;
    }
    let h = &*(handle as *mut LanceIndex);
    match h.get_all_vectors() {
        Ok((labels, vectors)) => {
            let count = labels.len();
            if !out_count.is_null() {
                *out_count = count as i64;
            }
            if !out_labels.is_null() && !out_vectors.is_null() {
                std::ptr::copy_nonoverlapping(labels.as_ptr(), out_labels, count);
                std::ptr::copy_nonoverlapping(vectors.as_ptr(), out_vectors, vectors.len());
            }
            count as i32
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &format!("get_all_vectors failed: {}", e));
            -1
        }
    }
}
