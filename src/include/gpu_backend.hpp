#pragma once

#include <cstdint>
#include <string>

namespace duckdb {

class LanceGpuBackend {
public:
	virtual ~LanceGpuBackend() = default;

	virtual bool IsAvailable() const = 0;
	virtual std::string Name() const = 0;

	/// Compute distances between nq query vectors and nv database vectors.
	/// out_distances: nq * nv matrix (row-major).
	virtual void ComputeDistances(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
	                              const std::string &metric, float *out_distances) = 0;
};

/// Get the Metal GPU backend (macOS only). Returns nullptr if unavailable.
LanceGpuBackend *GetMetalBackend();

/// Get the CPU fallback backend.
LanceGpuBackend *GetCpuBackend();

/// Get the best available backend.
LanceGpuBackend *GetBestBackend();

} // namespace duckdb
