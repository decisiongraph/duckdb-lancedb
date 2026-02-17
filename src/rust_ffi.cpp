// Rust LanceDB FFI wrapper for DuckDB extension

#include "rust_ffi.hpp"
#include "duckdb/common/exception.hpp"
#include <string>

extern "C" {

void *lance_create_detached(const char *db_path, int32_t dimension, const char *metric, const char *table_name,
                            char *err_buf, int err_buf_len);
void *lance_create_detached_from_arrow(const char *db_path, void *arrow_schema, const char *metric,
                                       const char *table_name, char *err_buf, int err_buf_len);
void *lance_open_detached(const char *db_path, const char *table_name, const char *metric, char *err_buf,
                          int err_buf_len);
void lance_free_detached(void *handle);
int32_t lance_detached_has_extra_columns(void *handle);
int32_t lance_detached_dimension(void *handle);
int64_t lance_detached_add(void *handle, const float *vector, int32_t dimension, char *err_buf, int err_buf_len);
int32_t lance_detached_add_batch(void *handle, const float *vectors, int32_t num, int32_t dim, int64_t *out_labels,
                                 char *err_buf, int err_buf_len);
int32_t lance_detached_add_batch_arrow(void *handle, void *arrow_schema, void *arrow_array, int64_t *out_labels,
                                       char *err_buf, int err_buf_len);
int32_t lance_detached_merge(void *target_handle, void *source_handle, const int64_t *live_source_labels,
                             int32_t live_count, int64_t *out_old_labels, int64_t *out_new_labels, char *err_buf,
                             int err_buf_len);
int32_t lance_detached_search(void *handle, const float *query, int32_t dim, int32_t k, int32_t nprobes,
                              int32_t refine_factor, int64_t *out_labels, float *out_distances, char *err_buf,
                              int err_buf_len);
int64_t lance_detached_count(void *handle, char *err_buf, int err_buf_len);
int32_t lance_detached_delete(void *handle, int64_t label, char *err_buf, int err_buf_len);
int32_t lance_detached_delete_batch(void *handle, const int64_t *labels, int32_t count, char *err_buf,
                                    int err_buf_len);
int32_t lance_detached_create_index(void *handle, int32_t num_partitions, int32_t num_sub_vectors, char *err_buf,
                                    int err_buf_len);
int32_t lance_detached_create_hnsw_index(void *handle, int32_t m, int32_t ef_construction, char *err_buf,
                                          int err_buf_len);
int32_t lance_detached_compact(void *handle, char *err_buf, int err_buf_len);
int32_t lance_detached_get_vector(void *handle, int64_t label, float *out_vec, int32_t capacity, char *err_buf,
                                  int err_buf_len);
int32_t lance_detached_get_all_vectors(void *handle, int64_t *out_labels, float *out_vectors, int64_t *out_count,
                                       char *err_buf, int err_buf_len);
}

namespace duckdb {

constexpr int ERR_BUF_LEN = 2048;

LanceHandle LanceCreateDetached(const std::string &db_path, int32_t dimension, const std::string &metric,
                                const std::string &table_name) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto handle = lance_create_detached(db_path.c_str(), dimension, metric.c_str(), table_name.c_str(), err_buf,
	                                    ERR_BUF_LEN);
	if (!handle) {
		throw IOException("Lance create: " + std::string(err_buf));
	}
	return handle;
}

LanceHandle LanceCreateDetachedFromArrow(const std::string &db_path, void *arrow_schema, const std::string &metric,
                                         const std::string &table_name) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto handle = lance_create_detached_from_arrow(db_path.c_str(), arrow_schema, metric.c_str(), table_name.c_str(),
	                                               err_buf, ERR_BUF_LEN);
	if (!handle) {
		throw IOException("Lance create_from_arrow: " + std::string(err_buf));
	}
	return handle;
}

LanceHandle LanceOpenDetached(const std::string &db_path, const std::string &table_name, const std::string &metric) {
	char err_buf[ERR_BUF_LEN] = {0};
	auto handle = lance_open_detached(db_path.c_str(), table_name.c_str(), metric.c_str(), err_buf, ERR_BUF_LEN);
	if (!handle) {
		throw IOException("Lance open: " + std::string(err_buf));
	}
	return handle;
}

void LanceFreeDetached(LanceHandle handle) {
	lance_free_detached(handle);
}

bool LanceDetachedHasExtraColumns(LanceHandle handle) {
	return lance_detached_has_extra_columns(handle) != 0;
}

int32_t LanceDetachedDimension(LanceHandle handle) {
	return lance_detached_dimension(handle);
}

int64_t LanceDetachedAdd(LanceHandle handle, const float *vector, int32_t dimension) {
	char err_buf[ERR_BUF_LEN] = {0};
	int64_t label = lance_detached_add(handle, vector, dimension, err_buf, ERR_BUF_LEN);
	if (label < 0) {
		throw IOException("Lance add: " + std::string(err_buf));
	}
	return label;
}

int32_t LanceDetachedAddBatch(LanceHandle handle, const float *vectors, int32_t num, int32_t dim,
                              int64_t *out_labels) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = lance_detached_add_batch(handle, vectors, num, dim, out_labels, err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw IOException("Lance add_batch: " + std::string(err_buf));
	}
	return n;
}

int32_t LanceDetachedAddBatchArrow(LanceHandle handle, void *arrow_schema, void *arrow_array, int64_t *out_labels) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = lance_detached_add_batch_arrow(handle, arrow_schema, arrow_array, out_labels, err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw IOException("Lance add_batch_arrow: " + std::string(err_buf));
	}
	return n;
}

int32_t LanceDetachedMerge(LanceHandle target, LanceHandle source, const int64_t *live_source_labels,
                           int32_t live_count, int64_t *out_old_labels, int64_t *out_new_labels) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = lance_detached_merge(target, source, live_source_labels, live_count, out_old_labels, out_new_labels,
	                                 err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw IOException("Lance merge: " + std::string(err_buf));
	}
	return n;
}

int32_t LanceDetachedSearch(LanceHandle handle, const float *query, int32_t dim, int32_t k, int32_t nprobes,
                            int32_t refine_factor, int64_t *out_labels, float *out_distances) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = lance_detached_search(handle, query, dim, k, nprobes, refine_factor, out_labels, out_distances,
	                                  err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw IOException("Lance search: " + std::string(err_buf));
	}
	return n;
}

int64_t LanceDetachedCount(LanceHandle handle) {
	char err_buf[ERR_BUF_LEN] = {0};
	int64_t n = lance_detached_count(handle, err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw IOException("Lance count: " + std::string(err_buf));
	}
	return n;
}

void LanceDetachedDelete(LanceHandle handle, int64_t label) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_delete(handle, label, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw IOException("Lance delete: " + std::string(err_buf));
	}
}

void LanceDetachedDeleteBatch(LanceHandle handle, const int64_t *labels, int32_t count) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_delete_batch(handle, labels, count, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw IOException("Lance delete_batch: " + std::string(err_buf));
	}
}

void LanceDetachedCreateIndex(LanceHandle handle, int32_t num_partitions, int32_t num_sub_vectors) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_create_index(handle, num_partitions, num_sub_vectors, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw IOException("Lance create_index: " + std::string(err_buf));
	}
}

void LanceDetachedCreateHnswIndex(LanceHandle handle, int32_t m, int32_t ef_construction) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_create_hnsw_index(handle, m, ef_construction, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw IOException("Lance create_hnsw_index: " + std::string(err_buf));
	}
}

void LanceDetachedCompact(LanceHandle handle) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t rc = lance_detached_compact(handle, err_buf, ERR_BUF_LEN);
	if (rc != 0) {
		throw IOException("Lance compact: " + std::string(err_buf));
	}
}

int32_t LanceDetachedGetVector(LanceHandle handle, int64_t label, float *out_vec, int32_t capacity) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t dim = lance_detached_get_vector(handle, label, out_vec, capacity, err_buf, ERR_BUF_LEN);
	if (dim < 0) {
		throw IOException("Lance get_vector: " + std::string(err_buf));
	}
	return dim;
}

int32_t LanceDetachedGetAllVectors(LanceHandle handle, int64_t *out_labels, float *out_vectors, int64_t *out_count) {
	char err_buf[ERR_BUF_LEN] = {0};
	int32_t n = lance_detached_get_all_vectors(handle, out_labels, out_vectors, out_count, err_buf, ERR_BUF_LEN);
	if (n < 0) {
		throw IOException("Lance get_all_vectors: " + std::string(err_buf));
	}
	return n;
}

} // namespace duckdb
