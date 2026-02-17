#include "gpu_backend.hpp"

#ifdef LANCE_METAL_AVAILABLE
#include "lance-metal/MetalVectorDistance.h"
#endif

namespace duckdb {

#ifdef LANCE_METAL_AVAILABLE

class MetalBackend : public LanceGpuBackend {
public:
    MetalBackend() {
        lance_metal::MetalInit();
    }

    bool IsAvailable() const override {
        return lance_metal::MetalIsAvailable();
    }

    std::string Name() const override {
        return "Metal";
    }

    void ComputeDistances(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
                          const std::string &metric, float *out_distances) override {
        lance_metal::MetalComputeDistances(queries, nq, vectors, nv, dim, metric, out_distances);
    }
};

static MetalBackend g_metal_backend;

LanceGpuBackend *GetMetalBackend() {
    return &g_metal_backend;
}

#endif // LANCE_METAL_AVAILABLE

} // namespace duckdb
