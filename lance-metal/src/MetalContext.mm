// Metal context and compute distance — single compilation unit to share static state.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "lance-metal/MetalVectorDistance.h"
#include <cstring>

namespace lance_metal {

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static bool g_initialized = false;

bool MetalInit() {
    if (g_initialized) {
        return g_device != nil;
    }
    g_initialized = true;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return false;
        }

        NSError *error = nil;
        NSString *libPath = @LANCE_METALLIB_PATH;
        NSURL *libURL = [NSURL fileURLWithPath:libPath];
        g_library = [g_device newLibraryWithURL:libURL error:&error];
        if (!g_library) {
            NSLog(@"Failed to load metallib: %@", error);
            g_device = nil;
            return false;
        }

        g_commandQueue = [g_device newCommandQueue];
        if (!g_commandQueue) {
            g_device = nil;
            g_library = nil;
            return false;
        }
    }

    return true;
}

bool MetalIsAvailable() {
    if (!g_initialized) {
        MetalInit();
    }
    return g_device != nil;
}

static id<MTLComputePipelineState> GetPipeline(const std::string &kernelName) {
    @autoreleasepool {
        NSString *name = [NSString stringWithUTF8String:kernelName.c_str()];
        id<MTLFunction> func = [g_library newFunctionWithName:name];
        if (!func) {
            return nil;
        }

        NSError *error = nil;
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            NSLog(@"Failed to create pipeline for %@: %@", name, error);
        }
        return pipeline;
    }
}

void MetalComputeDistances(const float *queries, int64_t nq, const float *vectors, int64_t nv, int64_t dim,
                           const std::string &metric, float *out_distances) {
    if (!MetalIsAvailable() || nq == 0 || nv == 0) {
        return;
    }

    @autoreleasepool {
        std::string kernelName;
        if (metric == "cosine") {
            kernelName = "cosine_distance";
        } else if (metric == "dot" || metric == "ip" || metric == "inner_product") {
            kernelName = "inner_product";
        } else {
            kernelName = "l2_distance";
        }

        id<MTLComputePipelineState> pipeline = GetPipeline(kernelName);
        if (!pipeline) {
            return;
        }

        NSUInteger querySize = nq * dim * sizeof(float);
        NSUInteger vectorSize = nv * dim * sizeof(float);
        NSUInteger distSize = nq * nv * sizeof(float);

        id<MTLBuffer> queryBuf = [g_device newBufferWithBytes:queries length:querySize options:MTLResourceStorageModeShared];
        id<MTLBuffer> vectorBuf = [g_device newBufferWithBytes:vectors length:vectorSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> distBuf = [g_device newBufferWithLength:distSize options:MTLResourceStorageModeShared];

        uint32_t dimU32 = static_cast<uint32_t>(dim);
        uint32_t nvU32 = static_cast<uint32_t>(nv);

        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:queryBuf offset:0 atIndex:0];
        [encoder setBuffer:vectorBuf offset:0 atIndex:1];
        [encoder setBuffer:distBuf offset:0 atIndex:2];
        [encoder setBytes:&dimU32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&nvU32 length:sizeof(uint32_t) atIndex:4];

        MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(nv), static_cast<NSUInteger>(nq), 1);
        NSUInteger threadGroupWidth = MIN(static_cast<NSUInteger>(nv), pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(out_distances, [distBuf contents], distSize);
    }
}

} // namespace lance_metal
