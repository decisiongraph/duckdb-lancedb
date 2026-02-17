#include "gpu_backend.hpp"
#include <cmath>
#include <cstring>

namespace duckdb {

class CpuBackend : public LanceGpuBackend {
public:
	bool IsAvailable() const override {
		return true;
	}

	std::string Name() const override {
		return "CPU";
	}

	void ComputeDistances(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
	                      const std::string &metric, float *out_distances) override {
		if (metric == "cosine") {
			ComputeCosine(queries, nq, vectors, nv, dim, out_distances);
		} else if (metric == "dot" || metric == "ip" || metric == "inner_product") {
			ComputeIP(queries, nq, vectors, nv, dim, out_distances);
		} else {
			ComputeL2(queries, nq, vectors, nv, dim, out_distances);
		}
	}

private:
	void ComputeL2(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
	               float *out_distances) {
		for (int64_t q = 0; q < nq; q++) {
			for (int64_t v = 0; v < nv; v++) {
				float dist = 0;
				for (int64_t d = 0; d < dim; d++) {
					float diff = queries[q * dim + d] - vectors[v * dim + d];
					dist += diff * diff;
				}
				out_distances[q * nv + v] = dist;
			}
		}
	}

	void ComputeCosine(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
	                   float *out_distances) {
		for (int64_t q = 0; q < nq; q++) {
			float q_norm = 0;
			for (int64_t d = 0; d < dim; d++) {
				q_norm += queries[q * dim + d] * queries[q * dim + d];
			}
			q_norm = std::sqrt(q_norm);

			for (int64_t v = 0; v < nv; v++) {
				float dot = 0, v_norm = 0;
				for (int64_t d = 0; d < dim; d++) {
					dot += queries[q * dim + d] * vectors[v * dim + d];
					v_norm += vectors[v * dim + d] * vectors[v * dim + d];
				}
				v_norm = std::sqrt(v_norm);
				float sim = (q_norm > 0 && v_norm > 0) ? dot / (q_norm * v_norm) : 0;
				out_distances[q * nv + v] = 1.0f - sim;
			}
		}
	}

	void ComputeIP(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
	               float *out_distances) {
		for (int64_t q = 0; q < nq; q++) {
			for (int64_t v = 0; v < nv; v++) {
				float dot = 0;
				for (int64_t d = 0; d < dim; d++) {
					dot += queries[q * dim + d] * vectors[v * dim + d];
				}
				out_distances[q * nv + v] = -dot;
			}
		}
	}
};

static CpuBackend g_cpu_backend;

LanceGpuBackend *GetCpuBackend() {
	return &g_cpu_backend;
}

#ifndef LANCE_METAL_AVAILABLE
LanceGpuBackend *GetMetalBackend() {
	return nullptr;
}
#endif

LanceGpuBackend *GetBestBackend() {
	auto *metal = GetMetalBackend();
	if (metal && metal->IsAvailable()) {
		return metal;
	}
	return GetCpuBackend();
}

} // namespace duckdb
