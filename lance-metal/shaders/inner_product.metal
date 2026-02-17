#include <metal_stdlib>
using namespace metal;

/// Compute negative inner product distance (for maximum inner product search).
kernel void inner_product(
    device const float *queries   [[buffer(0)]],
    device const float *vectors   [[buffer(1)]],
    device float *distances       [[buffer(2)]],
    constant uint &dim            [[buffer(3)]],
    constant uint &nv             [[buffer(4)]],
    uint2 gid                     [[thread_position_in_grid]])
{
    uint q_idx = gid.y;
    uint v_idx = gid.x;

    float dot = 0.0f;

    uint q_offset = q_idx * dim;
    uint v_offset = v_idx * dim;

    for (uint d = 0; d < dim; d++) {
        dot += queries[q_offset + d] * vectors[v_offset + d];
    }

    // Negative for distance ordering (smaller = more similar)
    distances[q_idx * nv + v_idx] = -dot;
}
