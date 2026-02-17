#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {

typedef void *LanceHandle;

// Create a Lance dataset at db_path. Throws on error.
LanceHandle LanceCreateDetached(const std::string &db_path, int32_t dimension, const std::string &metric);
void LanceFreeDetached(LanceHandle handle);

// Add single vector. Returns label.
int64_t LanceDetachedAdd(LanceHandle handle, const float *vector, int32_t dimension);

// Add batch of vectors. Returns count. Fills out_labels.
int32_t LanceDetachedAddBatch(LanceHandle handle, const float *vectors, int32_t num, int32_t dim,
                              int64_t *out_labels);

// Search. Returns count. Fills out_labels, out_distances.
int32_t LanceDetachedSearch(LanceHandle handle, const float *query, int32_t dim, int32_t k, int32_t nprobes,
                            int32_t refine_factor, int64_t *out_labels, float *out_distances);

int64_t LanceDetachedCount(LanceHandle handle);
void LanceDetachedDelete(LanceHandle handle, int64_t label);

void LanceDetachedCreateIndex(LanceHandle handle, int32_t num_partitions, int32_t num_sub_vectors);
void LanceDetachedCompact(LanceHandle handle);

int32_t LanceDetachedGetVector(LanceHandle handle, int64_t label, float *out_vec, int32_t capacity);

// Metadata serialization (Lance handles vector data on disk)
struct LanceSerializedMeta {
	uint8_t *data;
	size_t len;
};
LanceSerializedMeta LanceDetachedSerializeMeta(LanceHandle handle);
LanceHandle LanceDetachedDeserializeMeta(const std::string &db_path, const uint8_t *data, size_t len);
void LanceFreeBytes(LanceSerializedMeta bytes);

} // namespace duckdb
