#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "NeuralRadianceCacheConstants.h"

namespace openpgl
{

enum class LayerActivation
{
    None,
    ReLU,
};

struct NeuralRadianceCacheSample 
{
    float3 pos;
    float3 dir; 
    float3 radianceEstimate;
};

struct KernelParams 
{
    NeuralRadianceCacheSample *samples;
    uint32_t numSamples;
    uint32_t ringOffset;
    uint32_t ringSize;

    const __half *hashTable;
    float *hashTableGradients;
    const uint32_t *levelOffsets;
    uint32_t hashTableSize;
    uint32_t baseResolution;
    float levelScale;

    __half *weightsLayer0;
    __half *weightsLayer1;
    __half *weightsLayer2;
    __half *weightsOutputLayer;

    float *weightsGradientsLayer0;
    float *weightsGradientsLayer1;
    float *weightsGradientsLayer2;
    float *weightsGradientsOutputLayer;
};

cudaError_t LaunchInitializeHalfBufferKernel(__half *values, uint32_t numValues, float scale, uint32_t seed, cudaStream_t stream);

cudaError_t LaunchTrainingKernel(const KernelParams &params, uint32_t numBlocks, cudaStream_t stream);

cudaError_t LaunchAdamWNetworkWeightsKernel(__half *values,
                                            float *gradients,
                                            float *firstMoments,
                                            float *secondMoments,
                                            uint32_t numValues,
                                            float learningRate,
                                            float beta1,
                                            float beta2,
                                            float beta1Power,
                                            float beta2Power,
                                            float epsilon,
                                            float weightDecay,
                                            float invBatchSize,
                                            cudaStream_t stream);

cudaError_t LaunchAdamWHashFeaturesKernel(__half *values,
                                          float *gradients,
                                          float *firstMoments,
                                          float *secondMoments,
                                          uint32_t numValues,
                                          float learningRate,
                                          float beta1,
                                          float beta2,
                                          float beta1Power,
                                          float beta2Power,
                                          float epsilon,
                                          float invBatchSize,
                                          cudaStream_t stream);

}
