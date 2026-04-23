#pragma once

#include "../../data/SampleContainerInternal.h"
#include "NeuralRadianceCacheHelpers.h"
#include "NeuralRadianceCacheKernels.cuh"
#include "../../openpgl_common.h"

#include <cuda_runtime.h>
#include <embreeSrc/common/math/vec3.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tbb/task_group.h>

#ifdef USE_EMBREE_PARALLEL
#define TASKING_TBB
#include <embreeSrc/common/algorithms/parallel_for.h>
#else
#include <tbb/parallel_for.h>
#endif

#ifndef CUDA_ASSERT
#define CUDA_ASSERT(statement)                                                                                                              \
    do                                                                                                                                      \
    {                                                                                                                                       \
        cudaError_t result = (statement);                                                                                                   \
        if (result != cudaSuccess)                                                                                                          \
        {                                                                                                                                   \
            fprintf(stderr, "CUDA Error (%s): %s in %s (%s:%d)\n", cudaGetErrorName(result), cudaGetErrorString(result), #statement,       \
                    __FILE__, __LINE__);                                                                                                    \
            std::exit(EXIT_FAILURE);                                                                                                        \
        }                                                                                                                                   \
    } while (0)
#endif

namespace openpgl
{

struct OptimizedHalfBuffer
{
    __half *values{nullptr};
    float *gradients{nullptr};
    float *firstMoments{nullptr};
    float *secondMoments{nullptr};
    uint32_t size{0};
};

struct MultiresolutionGrid
{
    OptimizedHalfBuffer hashTable;
    uint32_t *levelOffsets{nullptr};
    uint32_t hashTableSize{0};
    uint32_t hashTableValueCount{0};

    uint32_t numLevels{OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_LEVELS};
    uint32_t baseResolution{8};
    uint32_t maxResolution{2048};
    float levelScale{1.0f};
};

struct NeuralRadianceCache
{
    enum class TrainingState
    {
        Idle,
        Running,
        StopRequested,
    };

    using Sample = NeuralRadianceCacheSample;

    // Pinned upload buffer for samples
    Sample *cpuRadianceCacheSamples;
    uint32_t cpuRadianceCacheSamplesSize;
    uint32_t cpuRadianceCacheSamplesCapacity;

    MultiresolutionGrid grid;
    OptimizedHalfBuffer weightsLayer0;
    OptimizedHalfBuffer weightsLayer1;
    OptimizedHalfBuffer weightsLayer2;
    OptimizedHalfBuffer weightsOutputLayer;

    // Ring buffer
    struct RingBuffer 
    {
        Sample *samples{nullptr};
        uint32_t ringStart{0};
        uint32_t size{0};
        uint32_t capacity{0};
    };

    RingBuffer gpuRingBuffer;

    // Scene bounds
    Vector3 sceneLower;
    Vector3 sceneUpper;
    Vector3 invSceneExtent;
    bool sceneBoundsSet;

    // Synchronization
    tbb::task_group trainingTasks;
    std::atomic<TrainingState> trainingState{TrainingState::Idle};
    cudaStream_t trainingStream{nullptr};
    uint32_t adamStep{0};
    float beta1Power{1.0f};
    float beta2Power{1.0f};

    NeuralRadianceCache(uint32_t ringCapacity = 10000000);
    ~NeuralRadianceCache();

    void setSceneBounds(const Vector3 &low, const Vector3 &high);
    Vector3 incomingRadiance(const Vector3 &direction) const;
    bool startTrainingAsync(uint32_t maxEpochs);
    void requestStopTraining();
    void waitForTraining();
    void addSamples(const ContainerInternal<SampleData> &sampleContainer);

private:
    static constexpr uint32_t trainingBatchSize = 8192;
    static constexpr float learningRate = 1.0e-3f;
    static constexpr float weightDecay = 1.0e-5f;
    static constexpr float beta1 = 0.9f;
    static constexpr float beta2 = 0.999f;
    static constexpr float adamEpsilon = 1.0e-8f;

    static constexpr uint32_t alignedFeatureVectorSize = ((OPENPGL_NEURAL_RADIANCE_CACHE_FEATURE_VECTOR_SIZE + 15) / 16) * 16;
    static constexpr uint32_t hiddenLayerSize = OPENPGL_NEURAL_RADIANCE_CACHE_HIDDEN_LAYER_SIZE;
    static constexpr uint32_t alignedOutputSize = ((OPENPGL_NEURAL_RADIANCE_CACHE_OUTPUT_SIZE + 15) / 16) * 16;
    static constexpr uint32_t layer0WeightCount = alignedFeatureVectorSize * hiddenLayerSize;
    static constexpr uint32_t hiddenWeightCount = hiddenLayerSize * hiddenLayerSize;
    static constexpr uint32_t outputWeightCount = hiddenLayerSize * alignedOutputSize;
    static constexpr uint32_t maxHashEntriesPerLevel = 1u << 16;

    Vector3 inference(const Vector3 &direction) const;
    void allocateHalfBuffer(OptimizedHalfBuffer &buffer, uint32_t size, float initScale, uint32_t seed);
    void freeHalfBuffer(OptimizedHalfBuffer &buffer);
    void clearGradientsAsync();
    void applyAdamWNetworkBuffer(OptimizedHalfBuffer &buffer, uint32_t batchCount);
    void applyAdamWHashBuffer(uint32_t batchCount);
    void initializeTrainingBuffers();
    void releaseTrainingBuffers();
    void trainingLoop(uint32_t maxEpochs);
    bool stopTrainingRequested() const;
};

NeuralRadianceCache::NeuralRadianceCache(uint32_t ringCapacity) : sceneBoundsSet(false), cpuRadianceCacheSamples(0), cpuRadianceCacheSamplesCapacity(0), cpuRadianceCacheSamplesSize(0)
{
    CUDA_ASSERT(cudaFree(nullptr));
    CUDA_ASSERT(cudaStreamCreateWithFlags(&trainingStream, cudaStreamNonBlocking));

    OPENPGL_ASSERT(ringCapacity > 0);
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&gpuRingBuffer.samples), sizeof(Sample) * ringCapacity));
    gpuRingBuffer.ringStart = 0;
    gpuRingBuffer.size = 0;
    gpuRingBuffer.capacity = ringCapacity;

    initializeTrainingBuffers();
}

NeuralRadianceCache::~NeuralRadianceCache()
{
    requestStopTraining();
    waitForTraining();

    if (trainingStream)
    {
        CUDA_ASSERT(cudaStreamSynchronize(trainingStream));
    }

    releaseTrainingBuffers();
    CUDA_ASSERT(cudaFree(gpuRingBuffer.samples));
    if (cpuRadianceCacheSamples)
    {
        CUDA_ASSERT(cudaFreeHost(cpuRadianceCacheSamples));
    }

    if (trainingStream)
    {
        CUDA_ASSERT(cudaStreamDestroy(trainingStream));
        trainingStream = nullptr;
    }
}

void NeuralRadianceCache::setSceneBounds(const Vector3 &low, const Vector3 &high)
{
    sceneLower = low;
    sceneUpper = high;
    invSceneExtent = sceneUpper - sceneLower;
    invSceneExtent.x = invSceneExtent.x == 0 ? 0.f : 1.f / invSceneExtent.x;
    invSceneExtent.y = invSceneExtent.y == 0 ? 0.f : 1.f / invSceneExtent.y;
    invSceneExtent.z = invSceneExtent.z == 0 ? 0.f : 1.f / invSceneExtent.z;

    sceneBoundsSet = true;
}

Vector3 NeuralRadianceCache::incomingRadiance(const Vector3 &direction) const
{
    return inference(direction);
}

Vector3 NeuralRadianceCache::inference(const Vector3 &direction) const
{
    (void)direction;
    // TODO: add a GPU inference path for the half-precision trainable parameters.
    return Vector3(0.f, 0.f, 0.f);
}

bool NeuralRadianceCache::startTrainingAsync(uint32_t maxEpochs)
{
    TrainingState expected = TrainingState::Idle;
    if (!trainingState.compare_exchange_strong(expected, TrainingState::Running, std::memory_order_acq_rel))
    {
        return false;
    }

    trainingTasks.wait();

    trainingTasks.run([this, maxEpochs]() {
        trainingLoop(maxEpochs);
        trainingState.store(TrainingState::Idle, std::memory_order_release);
    });

    return true;
}

void NeuralRadianceCache::requestStopTraining()
{
    TrainingState expected = TrainingState::Running;
    trainingState.compare_exchange_strong(expected, TrainingState::StopRequested, std::memory_order_acq_rel);
}

void NeuralRadianceCache::waitForTraining()
{
    trainingTasks.wait();
}

void NeuralRadianceCache::addSamples(const ContainerInternal<SampleData> &sampleContainer)
{
    OPENPGL_ASSERT(sceneBoundsSet);
    if (sampleContainer.size() == 0)
    {
        return;
    }

    CUDA_ASSERT(cudaStreamSynchronize(trainingStream));

    if (cpuRadianceCacheSamplesCapacity < sampleContainer.size())
    {
        if (cpuRadianceCacheSamples)
        {
            CUDA_ASSERT(cudaFreeHost(cpuRadianceCacheSamples));
        }
        CUDA_ASSERT(cudaMallocHost(&cpuRadianceCacheSamples, sizeof(Sample) * 2 * sampleContainer.size()));
        cpuRadianceCacheSamplesCapacity = 2 * sampleContainer.size();
    }
    cpuRadianceCacheSamplesSize = sampleContainer.size();

    auto clamp01 = [](float x)
    {
        return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
    };

#ifdef USE_EMBREE_PARALLEL
    embree::parallel_for(size_t(0), sampleContainer.size(), size_t(4 * 4096), [&](const embree::range<size_t> &r) {
#else
    tbb::parallel_for(tbb::blocked_range<int>(0, sampleContainer.size()), [&](tbb::blocked_range<int> r) {
#endif
        for (size_t i = r.begin(); i < r.end(); i++) {
            const SampleData &data = sampleContainer[i];

            pgl_vec3f scaledPosition;
            scaledPosition.x = clamp01((data.position.x - sceneLower.x) * invSceneExtent.x);
            scaledPosition.y = clamp01((data.position.y - sceneLower.y) * invSceneExtent.y);
            scaledPosition.z = clamp01((data.position.z - sceneLower.z) * invSceneExtent.z);

            pgl_vec3f dir = pgl_vec3f(data.direction);
            pgl_vec3f radiance = pgl_vec3f(data.radianceIn);

            pgl_vec3f radianceInNoMIS = isDirectLight(data) ? radiance / data.radianceInMISWeight : radiance;

            Sample &sample = cpuRadianceCacheSamples[i];
            sample.pos.x = scaledPosition.x;
            sample.pos.y = scaledPosition.y;
            sample.pos.z = scaledPosition.z;

            sample.dir.x = dir.x;
            sample.dir.y = dir.y;
            sample.dir.z = dir.z;

            sample.radianceEstimate.x = radianceInNoMIS.x;
            sample.radianceEstimate.y = radianceInNoMIS.y;
            sample.radianceEstimate.z = radianceInNoMIS.z;
        }
    });

    if (cpuRadianceCacheSamplesSize >= gpuRingBuffer.capacity)
    {
        const Sample *copySource = cpuRadianceCacheSamples + (cpuRadianceCacheSamplesSize - gpuRingBuffer.capacity);
        CUDA_ASSERT(cudaMemcpyAsync(gpuRingBuffer.samples, copySource, sizeof(Sample) * gpuRingBuffer.capacity, cudaMemcpyKind::cudaMemcpyHostToDevice, trainingStream));
        gpuRingBuffer.ringStart = 0;
        gpuRingBuffer.size = gpuRingBuffer.capacity;
        return;
    }

    uint32_t wrappedRingStart = gpuRingBuffer.ringStart % gpuRingBuffer.capacity;

    uint32_t range0Size = std::min(gpuRingBuffer.capacity - wrappedRingStart, cpuRadianceCacheSamplesSize);
    CUDA_ASSERT(cudaMemcpyAsync(gpuRingBuffer.samples + wrappedRingStart, cpuRadianceCacheSamples, sizeof(Sample) * range0Size, cudaMemcpyKind::cudaMemcpyHostToDevice, trainingStream));
    uint32_t range1Size = range0Size == cpuRadianceCacheSamplesSize ? 0 : std::min(gpuRingBuffer.capacity, cpuRadianceCacheSamplesSize - range0Size);
    if (range1Size)
    {
        CUDA_ASSERT(cudaMemcpyAsync(gpuRingBuffer.samples, cpuRadianceCacheSamples + range0Size, sizeof(Sample) * range1Size, cudaMemcpyKind::cudaMemcpyHostToDevice,
                                    trainingStream));
    }

    gpuRingBuffer.ringStart += cpuRadianceCacheSamplesSize;
    gpuRingBuffer.size = std::min(gpuRingBuffer.size + cpuRadianceCacheSamplesSize, gpuRingBuffer.capacity);
}

bool NeuralRadianceCache::stopTrainingRequested() const
{
    return trainingState.load(std::memory_order_acquire) == TrainingState::StopRequested;
}

void NeuralRadianceCache::allocateHalfBuffer(OptimizedHalfBuffer &buffer, uint32_t size, float initScale, uint32_t seed)
{
    buffer.size = size;

    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.values), sizeof(__half) * buffer.size));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.gradients), sizeof(float) * buffer.size));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.firstMoments), sizeof(float) * buffer.size));
    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&buffer.secondMoments), sizeof(float) * buffer.size));

    CUDA_ASSERT(cudaMemsetAsync(buffer.gradients, 0, sizeof(float) * buffer.size, trainingStream));
    CUDA_ASSERT(cudaMemsetAsync(buffer.firstMoments, 0, sizeof(float) * buffer.size, trainingStream));
    CUDA_ASSERT(cudaMemsetAsync(buffer.secondMoments, 0, sizeof(float) * buffer.size, trainingStream));
    CUDA_ASSERT(LaunchInitializeHalfBufferKernel(buffer.values, buffer.size, initScale, seed, trainingStream));
}

void NeuralRadianceCache::freeHalfBuffer(OptimizedHalfBuffer &buffer)
{
    CUDA_ASSERT(cudaFree(buffer.values));
    CUDA_ASSERT(cudaFree(buffer.gradients));
    CUDA_ASSERT(cudaFree(buffer.firstMoments));
    CUDA_ASSERT(cudaFree(buffer.secondMoments));
    buffer = OptimizedHalfBuffer{};
}

void NeuralRadianceCache::clearGradientsAsync()
{
    CUDA_ASSERT(cudaMemsetAsync(grid.hashTable.gradients, 0, sizeof(float) * grid.hashTable.size, trainingStream));
    CUDA_ASSERT(cudaMemsetAsync(weightsLayer0.gradients, 0, sizeof(float) * weightsLayer0.size, trainingStream));
    CUDA_ASSERT(cudaMemsetAsync(weightsLayer1.gradients, 0, sizeof(float) * weightsLayer1.size, trainingStream));
    CUDA_ASSERT(cudaMemsetAsync(weightsLayer2.gradients, 0, sizeof(float) * weightsLayer2.size, trainingStream));
    CUDA_ASSERT(cudaMemsetAsync(weightsOutputLayer.gradients, 0, sizeof(float) * weightsOutputLayer.size, trainingStream));
}

void NeuralRadianceCache::applyAdamWNetworkBuffer(OptimizedHalfBuffer &buffer, uint32_t batchCount)
{
    const float invBatchSize = batchCount > 0 ? 1.0f / float(batchCount) : 0.0f;
    CUDA_ASSERT(LaunchAdamWNetworkWeightsKernel(buffer.values, buffer.gradients, buffer.firstMoments, buffer.secondMoments, buffer.size, learningRate, beta1, beta2, beta1Power,
                                                beta2Power, adamEpsilon, weightDecay, invBatchSize, trainingStream));
}

void NeuralRadianceCache::applyAdamWHashBuffer(uint32_t batchCount)
{
    const float invBatchSize = batchCount > 0 ? 1.0f / float(batchCount) : 0.0f;
    CUDA_ASSERT(LaunchAdamWHashFeaturesKernel(grid.hashTable.values, grid.hashTable.gradients, grid.hashTable.firstMoments, grid.hashTable.secondMoments, grid.hashTable.size,
                                              learningRate, beta1, beta2, beta1Power, beta2Power, adamEpsilon, invBatchSize, trainingStream));
}

void NeuralRadianceCache::initializeTrainingBuffers()
{
    grid.levelScale = std::exp(std::log(float(grid.maxResolution) / float(grid.baseResolution)) / float(grid.numLevels - 1));

    std::array<uint32_t, OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_LEVELS + 1> levelOffsets{};
    levelOffsets[0] = 0;
    for (uint32_t level = 0; level < grid.numLevels; ++level)
    {
        const float scale = float(grid.baseResolution) * std::pow(grid.levelScale, float(level)) - 1.0f;
        const uint32_t resolution = uint32_t(std::ceil(scale)) + 1;
        const uint64_t denseTableSize = uint64_t(resolution) * uint64_t(resolution) * uint64_t(resolution);
        const uint32_t levelTableSize = uint32_t(std::min<uint64_t>(denseTableSize, maxHashEntriesPerLevel));
        levelOffsets[level + 1] = levelOffsets[level] + levelTableSize;
    }

    grid.hashTableSize = levelOffsets[grid.numLevels];
    grid.hashTableValueCount = grid.hashTableSize * OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_FEATURES;

    CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&grid.levelOffsets), sizeof(uint32_t) * levelOffsets.size()));
    CUDA_ASSERT(cudaMemcpyAsync(grid.levelOffsets, levelOffsets.data(), sizeof(uint32_t) * levelOffsets.size(), cudaMemcpyHostToDevice, trainingStream));

    allocateHalfBuffer(grid.hashTable, grid.hashTableValueCount, 1.0e-4f, 0x4f70674cU);
    allocateHalfBuffer(weightsLayer0, layer0WeightCount, std::sqrt(2.0f / float(alignedFeatureVectorSize)), 0x1234U);
    allocateHalfBuffer(weightsLayer1, hiddenWeightCount, std::sqrt(2.0f / float(hiddenLayerSize)), 0x2345U);
    allocateHalfBuffer(weightsLayer2, hiddenWeightCount, std::sqrt(2.0f / float(hiddenLayerSize)), 0x3456U);
    allocateHalfBuffer(weightsOutputLayer, outputWeightCount, std::sqrt(1.0f / float(hiddenLayerSize)), 0x4567U);

    CUDA_ASSERT(cudaStreamSynchronize(trainingStream));
}

void NeuralRadianceCache::releaseTrainingBuffers()
{
    freeHalfBuffer(grid.hashTable);
    CUDA_ASSERT(cudaFree(grid.levelOffsets));
    grid.levelOffsets = nullptr;
    grid.hashTableSize = 0;
    grid.hashTableValueCount = 0;

    freeHalfBuffer(weightsLayer0);
    freeHalfBuffer(weightsLayer1);
    freeHalfBuffer(weightsLayer2);
    freeHalfBuffer(weightsOutputLayer);
}

void NeuralRadianceCache::trainingLoop(uint32_t maxEpochs)
{
    if (gpuRingBuffer.size == 0)
    {
        return;
    }

    const uint32_t trainingSampleCount = gpuRingBuffer.size;
    const bool ringIsFull = gpuRingBuffer.size == gpuRingBuffer.capacity;
    const uint32_t oldestOffset = ringIsFull ? (gpuRingBuffer.ringStart % gpuRingBuffer.capacity) : 0;
    const uint32_t ringSize = ringIsFull ? gpuRingBuffer.capacity : gpuRingBuffer.size;

    for (uint32_t epoch = 0; epoch < maxEpochs; ++epoch)
    {
        if (stopTrainingRequested())
            break;

        for (uint32_t batchStart = 0; batchStart < trainingSampleCount; batchStart += trainingBatchSize)
        {
            if (stopTrainingRequested())
                break;

            const uint32_t batchCount = std::min(trainingBatchSize, trainingSampleCount - batchStart);
            const uint32_t numBlocks = std::min<uint32_t>((batchCount + 127) / 128, 1024);

            clearGradientsAsync();

            KernelParams params{};
            params.samples = gpuRingBuffer.samples;
            params.numSamples = batchCount;
            params.ringOffset = (oldestOffset + batchStart) % gpuRingBuffer.capacity;
            params.ringSize = ringSize;
            params.hashTable = grid.hashTable.values;
            params.hashTableGradients = grid.hashTable.gradients;
            params.levelOffsets = grid.levelOffsets;
            params.hashTableSize = grid.hashTableSize;
            params.baseResolution = grid.baseResolution;
            params.levelScale = grid.levelScale;
            params.weightsLayer0 = weightsLayer0.values;
            params.weightsLayer1 = weightsLayer1.values;
            params.weightsLayer2 = weightsLayer2.values;
            params.weightsOutputLayer = weightsOutputLayer.values;
            params.weightsGradientsLayer0 = weightsLayer0.gradients;
            params.weightsGradientsLayer1 = weightsLayer1.gradients;
            params.weightsGradientsLayer2 = weightsLayer2.gradients;
            params.weightsGradientsOutputLayer = weightsOutputLayer.gradients;

            CUDA_ASSERT(LaunchTrainingKernel(params, numBlocks, trainingStream));

            ++adamStep;
            beta1Power *= beta1;
            beta2Power *= beta2;

            applyAdamWNetworkBuffer(weightsLayer0, batchCount);
            applyAdamWNetworkBuffer(weightsLayer1, batchCount);
            applyAdamWNetworkBuffer(weightsLayer2, batchCount);
            applyAdamWNetworkBuffer(weightsOutputLayer, batchCount);
            applyAdamWHashBuffer(batchCount);
        }
    }

    CUDA_ASSERT(cudaStreamSynchronize(trainingStream));
}

}
