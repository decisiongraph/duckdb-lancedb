#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {

typedef void *LanceHandle;

// Create a Lance dataset at db_path. table_name identifies the Lance table within the dataset.
LanceHandle LanceCreateDetached(const std::string &db_path, int32_t dimension, const std::string &metric,
                                const std::string &table_name);
// Create from Arrow schema (multi-column, zero-copy). arrow_schema is an ArrowSchema*.
LanceHandle LanceCreateDetachedFromArrow(const std::string &db_path, void *arrow_schema, const std::string &metric,
                                         const std::string &table_name);
// Open existing Lance dataset, deriving schema from the table.
LanceHandle LanceOpenDetached(const std::string &db_path, const std::string &table_name, const std::string &metric);
void LanceFreeDetached(LanceHandle handle);

// Check if index has extra columns beyond label + vector.
bool LanceDetachedHasExtraColumns(LanceHandle handle);
// Get dimension from the Rust handle.
int32_t LanceDetachedDimension(LanceHandle handle);

// Add single vector. Returns label.
int64_t LanceDetachedAdd(LanceHandle handle, const float *vector, int32_t dimension);

// Add batch of vectors. Returns count. Fills out_labels.
int32_t LanceDetachedAddBatch(LanceHandle handle, const float *vectors, int32_t num, int32_t dim,
                              int64_t *out_labels);

// Add batch via Arrow C Data Interface (multi-column). Returns count. Fills out_labels.
// Takes ownership of arrow_array (sets release to null); caller must release arrow_schema.
int32_t LanceDetachedAddBatchArrow(LanceHandle handle, void *arrow_schema, void *arrow_array, int64_t *out_labels);

// Merge live rows from source into target (all in Rust). Returns count of merged rows.
// Fills out_old_labels and out_new_labels with the mapping.
int32_t LanceDetachedMerge(LanceHandle target, LanceHandle source, const int64_t *live_source_labels,
                           int32_t live_count, int64_t *out_old_labels, int64_t *out_new_labels);

// Search. Returns count. Fills out_labels, out_distances.
int32_t LanceDetachedSearch(LanceHandle handle, const float *query, int32_t dim, int32_t k, int32_t nprobes,
                            int32_t refine_factor, int64_t *out_labels, float *out_distances);

int64_t LanceDetachedCount(LanceHandle handle);
void LanceDetachedDelete(LanceHandle handle, int64_t label);
void LanceDetachedDeleteBatch(LanceHandle handle, const int64_t *labels, int32_t count);

void LanceDetachedCreateIndex(LanceHandle handle, int32_t num_partitions, int32_t num_sub_vectors);
void LanceDetachedCreateHnswIndex(LanceHandle handle, int32_t m, int32_t ef_construction);
void LanceDetachedCompact(LanceHandle handle);

int32_t LanceDetachedGetVector(LanceHandle handle, int64_t label, float *out_vec, int32_t capacity);

// Bulk vector export — returns count, fills out_labels and out_vectors.
// Pass nullptr for out_labels/out_vectors to get count first (via out_count).
int32_t LanceDetachedGetAllVectors(LanceHandle handle, int64_t *out_labels, float *out_vectors, int64_t *out_count);

} // namespace duckdb
