#include <metal_stdlib>
using namespace metal;

/// Compute cosine distance (1 - cosine_similarity) between query and database vectors.
kernel void cosine_distance(
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
    float q_norm = 0.0f;
    float v_norm = 0.0f;

    uint q_offset = q_idx * dim;
    uint v_offset = v_idx * dim;

    for (uint d = 0; d < dim; d++) {
        float qv = queries[q_offset + d];
        float vv = vectors[v_offset + d];
        dot += qv * vv;
        q_norm += qv * qv;
        v_norm += vv * vv;
    }

    q_norm = sqrt(q_norm);
    v_norm = sqrt(v_norm);

    float sim = (q_norm > 0.0f && v_norm > 0.0f) ? dot / (q_norm * v_norm) : 0.0f;
    distances[q_idx * nv + v_idx] = 1.0f - sim;
}
