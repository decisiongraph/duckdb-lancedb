#include <metal_stdlib>
using namespace metal;

/// Compute L2 squared distance between query vectors and database vectors.
/// Each thread computes one (query, vector) pair.
///
/// queries:  [nq * dim] flat
/// vectors:  [nv * dim] flat
/// distances: [nq * nv] output
kernel void l2_distance(
    device const float *queries   [[buffer(0)]],
    device const float *vectors   [[buffer(1)]],
    device float *distances       [[buffer(2)]],
    constant uint &dim            [[buffer(3)]],
    constant uint &nv             [[buffer(4)]],
    uint2 gid                     [[thread_position_in_grid]])
{
    uint q_idx = gid.y;  // query index
    uint v_idx = gid.x;  // vector index

    float dist = 0.0f;
    uint q_offset = q_idx * dim;
    uint v_offset = v_idx * dim;

    // Use simdgroup reduction for better performance on large dimensions
    for (uint d = 0; d < dim; d++) {
        float diff = queries[q_offset + d] - vectors[v_offset + d];
        dist += diff * diff;
    }

    distances[q_idx * nv + v_idx] = dist;
}
