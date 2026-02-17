// Rust LanceDB FFI wrapper for DuckDB extension

#include "rust_ffi.hpp"
#include <stdexcept>
#include <string>

extern "C" {

struct LanceBytesFFI {
	uint8_t *data;
	size_t len;
};

void *lance_create_detached(const char *db_path, int32_t dimension, const char *metric, char *err_buf,
                            int err_buf_len);
void lance_free_detached(void *handle);
int64_t lance_detached_add(void *handle, const float *vector, int32_t dimension, char *err_buf, int err_buf_len);
int32_t lance_detached_add_batch(void *handle, const float *vectors, int32_t num, int32_t dim, int64_t *out_labels,
                                 char *err_buf, int err_buf_len);
int32_t lance_detached_search(void *handle, const float *query, int32_t dim, int32_t k, int32_t nprobes,
                              int32_t refine_factor, int64_t *out_labels, float *out_distances, char *err_buf,
                              int err_buf_len);
int64_t lance_detached_count(void *handle, char *err_buf, int err_buf_len);
int32_t lance_detached_delete(void *handle, int64_t label, char *err_buf, int err_buf_len);
int32_t lance_detached_create_index(void *handle, int32_t num_partitions, int32_t num_sub_vectors, char *err_buf,
                                    int err_buf_len);
int32_t lance_detached_compact(void *handle, char *err_buf, int err_buf_len);
int32_t lance_detached_get_vector(void *handle, int64_t label, float *out_vec, int32_t capacity, char *err_buf,
                                  int err_buf_len);
LanceBytesFFI lance_detached_serialize_meta(void *handle, char *err_buf, int err_buf_len);
void *lance_detached_deserialize_meta(const char *db_path, const uint8_t *data, size_t len, char *err_buf,
                                      int err_buf_len);
void lance_free_bytes(LanceBytesFFI bytes);
}

namespace duckdb {

constexpr int ERR_BUF_LEN = 512;

LanceHandle LanceCreateDetached(const std::string &db_path, int32_t dimension, const std::string &metric) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto handle = lance_create_detached(db_path.c_str(), dimension, metric.c_str(), err_buf, ERR_BUF_LEN);
	if (!handle) {
		throw std::runtime_error("Lance create: " + std::string(err_buf));
	}
	return handle;
}

void LanceFreeDetached(LanceHandle handle) {
	lance_free_detached(handle);
}

int64_t LanceDetachedAdd(LanceHandle handle, const float *vector, int32_t dimension) {
	char err_buf[ERR_BUF_LEN] = {0};
	int64_t label = lance_detached_add(handle, vector, dimension, err_buf, ERR_BUF_LEN);
	if (label < 0) {
		throw std::runtime_error("Lance add: " + std::string(err_buf));
	}
	return label;
}

int32_t LanceDetachedAddBatch(LanceHandle handle, const float *vectors, int32_t num, int32_t dim,
                              int64_t *out_labels) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = lance_detached_add_batch(handle, vectors, num, dim, out_labels, err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw std::runtime_error("Lance add_batch: " + std::string(err_buf));
	}
	return n;
}

int32_t LanceDetachedSearch(LanceHandle handle, const float *query, int32_t dim, int32_t k, int32_t nprobes,
                            int32_t refine_factor, int64_t *out_labels, float *out_distances) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = lance_detached_search(handle, query, dim, k, nprobes, refine_factor, out_labels, out_distances,
	                                  err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw std::runtime_error("Lance search: " + std::string(err_buf));
	}
	return n;
}

int64_t LanceDetachedCount(LanceHandle handle) {
	char err_buf[ERR_BUF_LEN] = {0};
	int64_t n = lance_detached_count(handle, err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw std::runtime_error("Lance count: " + std::string(err_buf));
	}
	return n;
}

void LanceDetachedDelete(LanceHandle handle, int64_t label) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_delete(handle, label, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw std::runtime_error("Lance delete: " + std::string(err_buf));
	}
}

void LanceDetachedCreateIndex(LanceHandle handle, int32_t num_partitions, int32_t num_sub_vectors) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_create_index(handle, num_partitions, num_sub_vectors, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw std::runtime_error("Lance create_index: " + std::string(err_buf));
	}
}

void LanceDetachedCompact(LanceHandle handle) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_compact(handle, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw std::runtime_error("Lance compact: " + std::string(err_buf));
	}
}

int32_t LanceDetachedGetVector(LanceHandle handle, int64_t label, float *out_vec, int32_t capacity) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t dim = lance_detached_get_vector(handle, label, out_vec, capacity, err_buf, ERR_BUF_LEN);
	if (dim < 0) {
		throw std::runtime_error("Lance get_vector: " + std::string(err_buf));
	}
	return dim;
}

LanceSerializedMeta LanceDetachedSerializeMeta(LanceHandle handle) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto result = lance_detached_serialize_meta(handle, err_buf, ERR_BUF_LEN);
	if (!result.data) {
		throw std::runtime_error("Lance serialize_meta: " + std::string(err_buf));
	}
	return {result.data, result.len};
}

LanceHandle LanceDetachedDeserializeMeta(const std::string &db_path, const uint8_t *data, size_t len) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto handle = lance_detached_deserialize_meta(db_path.c_str(), data, len, err_buf, ERR_BUF_LEN);
	if (!handle) {
		throw std::runtime_error("Lance deserialize_meta: " + std::string(err_buf));
	}
	return handle;
}

void LanceFreeBytes(LanceSerializedMeta bytes) {
	LanceBytesFFI raw = {bytes.data, bytes.len};
	lance_free_bytes(raw);
}

} // namespace duckdb
