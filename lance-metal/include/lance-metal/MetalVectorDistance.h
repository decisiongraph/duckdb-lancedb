#pragma once

#include <cstdint>
#include <string>

namespace lance_metal {

/// Initialize Metal context. Returns false if Metal unavailable.
bool MetalInit();

/// Check if Metal is available and initialized.
bool MetalIsAvailable();

/// Compute pairwise distances between query and database vectors on GPU.
///
/// queries:  nq * dim float array
/// vectors:  nv * dim float array
/// metric:   "l2", "cosine", or "ip"
/// out_distances: nq * nv float array (row-major, pre-allocated by caller)
void MetalComputeDistances(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
                           const std::string &metric, float *out_distances);

} // namespace lance_metal
