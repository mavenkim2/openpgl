#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tiny-cuda-nn/mma.h>

#include "NeuralRadianceCacheHelpers.h"
#include "NeuralRadianceCacheKernels.cuh"

namespace openpgl
{
template <uint32_t value, uint32_t alignment>
__host__ __device__ static constexpr uint32_t AlignUp()
{
    return ((value + alignment - 1) / alignment) * alignment;
}

template <int inputSize, int outputSize, LayerActivation activation = LayerActivation::ReLU>
OPENPGL_DEVICE inline tcnn::mma_vec<AlignUp<outputSize, 16>()> ForwardPassGeneric(const tcnn::mma_vec<AlignUp<inputSize, 16>()> &inputsMatrix, const __half *weightsLayer)
{
    constexpr int weightsRows = AlignUp<inputSize, 16>();
    constexpr int weightsColumns = AlignUp<outputSize, 16>();

    using weightsMatrix = tcnn::mma_mat<weightsRows, weightsColumns, tcnn::CM>;

    weightsMatrix weights = weightsMatrix::from_linear_memory(weightsLayer);

    auto outputMatrix = inputsMatrix * weights;

    if constexpr (activation == LayerActivation::ReLU)
    {
        outputMatrix.template activate<tcnn::Activation::ReLU>();
    }
    return outputMatrix;
}

template <uint32_t inputSize>
OPENPGL_DEVICE inline float3 ForwardPass(const __half *sampledFeatures, const __half *weightsLayer0, const __half *weightsLayer1, const __half *weightsLayer2,
                                         const __half *weightsOutputLayer, tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> *activatedHiddenLayer0 = nullptr,
                                         tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> *activatedHiddenLayer1 = nullptr,
                                         tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> *activatedHiddenLayer2 = nullptr)
{
    static_assert(OPENPGL_NEURAL_RADIANCE_CACHE_NUM_HIDDEN_LAYERS == 3, "ForwardPass assumes exactly 3 hidden layers.");

    tcnn::hvec<inputSize> inputsVector(sampledFeatures);
    tcnn::mma_vec<AlignUp<inputSize, 16>()> inputsMatrix(inputsVector);

    auto outputHiddenLayer0 = ForwardPassGeneric<inputSize, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>(inputsMatrix, weightsLayer0);

    if (activatedHiddenLayer0)
    {
        *activatedHiddenLayer0 = outputHiddenLayer0.template vec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>();
    }

    auto outputHiddenLayer1 =
        ForwardPassGeneric<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>(outputHiddenLayer0, weightsLayer1);

    if (activatedHiddenLayer1)
    {
        *activatedHiddenLayer1 = outputHiddenLayer1.template vec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>();
    }

    auto outputHiddenLayer2 =
        ForwardPassGeneric<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>(outputHiddenLayer1, weightsLayer2);

    if (activatedHiddenLayer2)
    {
        *activatedHiddenLayer2 = outputHiddenLayer2.template vec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>();
    }

    auto finalOutput = ForwardPassGeneric<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, OPENPGL_NEURAL_RADIANCE_CACHE_OUTPUT_SIZE, LayerActivation::None>(outputHiddenLayer2,
                                                                                                                                                             weightsOutputLayer);

    auto finalOutputVector = finalOutput.template vec<OPENPGL_NEURAL_RADIANCE_CACHE_OUTPUT_SIZE>();
    return make_float3(__half2float(finalOutputVector[0]), __half2float(finalOutputVector[1]), __half2float(finalOutputVector[2]));
}

template <uint32_t N_THREADS, uint32_t M, uint32_t N, tcnn::MatrixLayout LAYOUT>
OPENPGL_DEVICE inline void SumIntoLinearGlobalMemoryHierarchicalFloat(const tcnn::mma_mat<M, N, LAYOUT> &matrix, float *sharedMemory, float *globalWeightGrad)
{
    static_assert(N_THREADS % 32 == 0, "N_THREADS must be divisible by warp size.");

    using mat_t = tcnn::mma_mat<M, N, LAYOUT>;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;
    constexpr uint32_t numWarps = N_THREADS / 32;

    float accum[mat_t::N_REGS * 2];
#pragma unroll
    for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
    {
        const __half2 reg = matrix.regs[i];
        accum[2 * i + 0] = __half2float(__low2half(reg));
        accum[2 * i + 1] = __half2float(__high2half(reg));
    }

    if constexpr (numWarps > 1)
    {
#pragma unroll
        for (uint32_t j = 2; j <= numWarps; j <<= 1)
        {
            const uint32_t sharedBase = (warpId / j) * mat_t::N_ELEMS;

            if (warpId % j == j / 2)
            {
#pragma unroll
                for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
                {
                    const uint32_t linearIndex = mat_t::to_linear(lane, i);
                    sharedMemory[sharedBase + linearIndex + 0] = accum[2 * i + 0];
                    sharedMemory[sharedBase + linearIndex + 1] = accum[2 * i + 1];
                }
            }

            __syncthreads();

            if (warpId % j == 0)
            {
#pragma unroll
                for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
                {
                    const uint32_t linearIndex = mat_t::to_linear(lane, i);
                    accum[2 * i + 0] += sharedMemory[sharedBase + linearIndex + 0];
                    accum[2 * i + 1] += sharedMemory[sharedBase + linearIndex + 1];
                }
            }

            __syncthreads();
        }
    }

    if (warpId == 0)
    {
#pragma unroll
        for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
        {
            const uint32_t linearIndex = mat_t::to_linear(lane, i);
            atomicAdd(&globalWeightGrad[linearIndex + 0], accum[2 * i + 0]);
            atomicAdd(&globalWeightGrad[linearIndex + 1], accum[2 * i + 1]);
        }
    }
}

template <uint32_t numThreads, uint32_t inputSize, uint32_t outputSize, LayerActivation activation = LayerActivation::ReLU>
OPENPGL_DEVICE inline tcnn::mma_vec<AlignUp<inputSize, 16>()> BackwardPassGeneric(tcnn::mma_vec<AlignUp<outputSize, 16>()> &outputGradientMatrix,
                                                                                const tcnn::hvec<inputSize> &layerInput, const __half *weights, float *layerWeightGradients,
                                                                                float *shmem)
{
    constexpr int weightsRows = AlignUp<inputSize, 16>();
    constexpr int weightsColumns = AlignUp<outputSize, 16>();
    using weightsMatrix = tcnn::mma_mat<weightsRows, weightsColumns, tcnn::CM>;

    weightsMatrix weightsOutput = weightsMatrix::from_linear_memory(weights);
    auto inputGradientMatrix = outputGradientMatrix * weightsOutput.transpose();

    tcnn::mma_vec<weightsRows> layerMatrixInput(layerInput);
    if constexpr (activation == LayerActivation::ReLU)
    {
        inputGradientMatrix.template activate_bwd<tcnn::Activation::ReLU>(layerMatrixInput);
    }

    auto outputWeightGradientMatrix = tcnn::outer_product(layerMatrixInput, outputGradientMatrix).flip_layout();

    // Write to memory
    SumIntoLinearGlobalMemoryHierarchicalFloat<numThreads>(outputWeightGradientMatrix, shmem, layerWeightGradients);

    return inputGradientMatrix;
}

template <uint32_t numThreads>
OPENPGL_DEVICE inline tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE> BackwardPass(
    const float3 &lossGradientVector, const tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> &activatedHiddenLayer0,
    const tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> &activatedHiddenLayer1,
    const tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> &activatedHiddenLayer2, const tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE> &networkInput,
    const __half *weightsHidden0, const __half *weightsHidden1, const __half *weightsHidden2, const __half *weightsOutputLayer, float *hidden0WeightGradients,
    float *hidden1WeightGradients, float *hidden2WeightGradients, float *outputWeightGradients)
{
    constexpr uint32_t numWarps = numThreads / 32;

    constexpr uint32_t shmemFloats =
        (numWarps > 1
             ? (numWarps / 2) * tcnn::mma_mat<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, tcnn::CM>::N_ELEMS
             : 0);
    __shared__ float shmem[shmemFloats > 0 ? shmemFloats : 1];

    constexpr uint32_t alignedOutputSize = AlignUp<3, 16>();
    tcnn::hvec<3> lossGradient;
    lossGradient[0] = __float2half(lossGradientVector.x);
    lossGradient[1] = __float2half(lossGradientVector.y);
    lossGradient[2] = __float2half(lossGradientVector.z);

    tcnn::mma_vec<alignedOutputSize> outputGradientMatrix(lossGradient);
    tcnn::mma_vec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> hiddenGradientMatrix2 =
        BackwardPassGeneric<numThreads, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, alignedOutputSize>(outputGradientMatrix, activatedHiddenLayer2, weightsOutputLayer,
                                                                                                            outputWeightGradients, shmem);

    tcnn::mma_vec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> hiddenGradientMatrix1 =
        BackwardPassGeneric<numThreads, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>(
            hiddenGradientMatrix2, activatedHiddenLayer1, weightsHidden2, hidden2WeightGradients, shmem);

    tcnn::mma_vec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> hiddenGradientMatrix0 =
        BackwardPassGeneric<numThreads, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE>(
            hiddenGradientMatrix1, activatedHiddenLayer0, weightsHidden1, hidden1WeightGradients, shmem);

    tcnn::mma_vec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> inputGradientMatrix =
        BackwardPassGeneric<numThreads, OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE, OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE, LayerActivation::None>(
            hiddenGradientMatrix0, networkInput, weightsHidden0, hidden0WeightGradients, shmem);

    return inputGradientMatrix.template vec<OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE>();
}

// template <uint32_t numThreads>
__global__ void TrainingKernel(KernelParams params)
{
    uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t idx = threadIndex; idx < params.numSamples; idx += gridDim.x * blockDim.x)
    {
        uint32_t sampleIndex = (idx + params.ringOffset) % params.ringSize;
        __half featureVector[OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE];
        const NeuralRadianceCacheSample &sample = params.samples[sampleIndex];
        HashGridForward<__half>(featureVector, sample.pos, sample.dir, params.hashTable, params.levelOffsets, params.baseResolution, params.levelScale);

        tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE> activatedHiddenLayers[OPENPGL_NEURAL_RADIANCE_CACHE_NUM_HIDDEN_LAYERS];

        float3 radiancePredicted = ForwardPass<OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE>(featureVector, params.weightsLayer0, params.weightsLayer1, params.weightsLayer2,
                                                                                                  params.weightsOutputLayer, &activatedHiddenLayers[0], &activatedHiddenLayers[1],
                                                                                                  &activatedHiddenLayers[2]);

        float3 radianceEstimateSqr = sample.radianceEstimate * sample.radianceEstimate;
        float3 mseTemp = (radiancePredicted - sample.radianceEstimate) * (radiancePredicted - sample.radianceEstimate);
        mseTemp /= radianceEstimateSqr;
        float relMse = (mseTemp.x + mseTemp.y + mseTemp.z) / 3.f;

        float3 lossGradient = (2.f / 3.f) * (radiancePredicted - sample.radianceEstimate) / radianceEstimateSqr;

        tcnn::hvec<OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE> inputsVector(featureVector);
        auto featureGradientVector = BackwardPass<128>(lossGradient, activatedHiddenLayers[0], activatedHiddenLayers[1], activatedHiddenLayers[2], inputsVector,
                                                       params.weightsLayer0, params.weightsLayer1, params.weightsLayer2, params.weightsOutputLayer,
                                                       params.weightsGradientsLayer0, params.weightsGradientsLayer1, params.weightsGradientsLayer2,
                                                       params.weightsGradientsOutputLayer);
        HashGridBackward(featureGradientVector, sample.pos, params.hashTableGradients, params.levelOffsets, params.baseResolution, params.levelScale);
    }
}

OPENPGL_DEVICE inline float RandomSignedFloat(uint32_t index, uint32_t seed)
{
    uint32_t x = index ^ seed;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    const float unit = float(x & 0x00ffffffU) / float(0x01000000U);
    return 2.0f * unit - 1.0f;
}

__global__ void InitializeHalfBufferKernel(__half *values, uint32_t numValues, float scale, uint32_t seed)
{
    const uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t index = threadIndex; index < numValues; index += stride)
    {
        values[index] = __float2half(RandomSignedFloat(index, seed) * scale);
    }
}

OPENPGL_DEVICE inline void AdamWUpdate(__half *values,
                                       float *gradients,
                                       float *firstMoments,
                                       float *secondMoments,
                                       uint32_t index,
                                       float learningRate,
                                       float beta1,
                                       float beta2,
                                       float beta1Power,
                                       float beta2Power,
                                       float epsilon,
                                       float weightDecay,
                                       float invBatchSize)
{
    float value = __half2float(values[index]);
    const float gradient = gradients[index] * invBatchSize;

    const float firstMoment = beta1 * firstMoments[index] + (1.0f - beta1) * gradient;
    const float secondMoment = beta2 * secondMoments[index] + (1.0f - beta2) * gradient * gradient;
    firstMoments[index] = firstMoment;
    secondMoments[index] = secondMoment;

    const float firstUnbiased = firstMoment / fmaxf(1.0f - beta1Power, 1.0e-20f);
    const float secondUnbiased = secondMoment / fmaxf(1.0f - beta2Power, 1.0e-20f);

    if (weightDecay != 0.0f)
    {
        value *= 1.0f - learningRate * weightDecay;
    }

    value -= learningRate * firstUnbiased / (sqrtf(secondUnbiased) + epsilon);
    values[index] = __float2half(value);
}

__global__ void AdamWNetworkWeightsKernel(__half *values,
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
                                          float invBatchSize)
{
    const uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t index = threadIndex; index < numValues; index += stride)
    {
        AdamWUpdate(values, gradients, firstMoments, secondMoments, index, learningRate, beta1, beta2, beta1Power, beta2Power, epsilon, weightDecay, invBatchSize);
    }
}

__global__ void AdamWHashFeaturesKernel(__half *values,
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
                                        float invBatchSize)
{
    const uint32_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t index = threadIndex; index < numValues; index += stride)
    {
        AdamWUpdate(values, gradients, firstMoments, secondMoments, index, learningRate, beta1, beta2, beta1Power, beta2Power, epsilon, 0.0f, invBatchSize);
    }
}

cudaError_t LaunchInitializeHalfBufferKernel(__half *values, uint32_t numValues, float scale, uint32_t seed, cudaStream_t stream)
{
    if (numValues == 0)
    {
        return cudaSuccess;
    }

    constexpr uint32_t blockSize = 256;
    const uint32_t numBlocks = (numValues + blockSize - 1) / blockSize;
    InitializeHalfBufferKernel<<<numBlocks, blockSize, 0, stream>>>(values, numValues, scale, seed);
    return cudaGetLastError();
}

cudaError_t LaunchTrainingKernel(const KernelParams &params, uint32_t numBlocks, cudaStream_t stream)
{
    if (params.numSamples == 0 || numBlocks == 0)
    {
        return cudaSuccess;
    }

    TrainingKernel<<<numBlocks, 128, 0, stream>>>(params);
    return cudaGetLastError();
}

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
                                            cudaStream_t stream)
{
    if (numValues == 0)
    {
        return cudaSuccess;
    }

    constexpr uint32_t blockSize = 256;
    const uint32_t numBlocks = (numValues + blockSize - 1) / blockSize;
    AdamWNetworkWeightsKernel<<<numBlocks, blockSize, 0, stream>>>(values, gradients, firstMoments, secondMoments, numValues, learningRate, beta1, beta2, beta1Power,
                                                                  beta2Power, epsilon, weightDecay, invBatchSize);
    return cudaGetLastError();
}

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
                                          cudaStream_t stream)
{
    if (numValues == 0)
    {
        return cudaSuccess;
    }

    constexpr uint32_t blockSize = 256;
    const uint32_t numBlocks = (numValues + blockSize - 1) / blockSize;
    AdamWHashFeaturesKernel<<<numBlocks, blockSize, 0, stream>>>(values, gradients, firstMoments, secondMoments, numValues, learningRate, beta1, beta2, beta1Power, beta2Power,
                                                                epsilon, invBatchSize);
    return cudaGetLastError();
}

}  // namespace openpgl
