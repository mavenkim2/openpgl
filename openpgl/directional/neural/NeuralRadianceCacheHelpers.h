#pragma once

#ifndef OPENPGL_DEVICE
#ifdef __CUDACC__
#define OPENPGL_DEVICE __device__
#else
#define OPENPGL_DEVICE
#endif
#endif

#include <cuda_fp16.h>

#include <cmath>
#include <type_traits>
// #include "../../openpgl_common.h"
#include "cuda_math.h"
#include "NeuralRadianceCacheConstants.h"

namespace openpgl
{

OPENPGL_DEVICE inline uint32_t SpatialHash(uint32_t vertexX, uint32_t vertexY, uint32_t vertexZ)
{
    const uint32_t pi1 = 1;
    const uint32_t pi2 = 2654435761;
    const uint32_t pi3 = 805459861;

    uint32_t result = 0;
    result ^= vertexX * pi1;
    result ^= vertexY * pi2;
    result ^= vertexZ * pi3;

    return result;
}

OPENPGL_DEVICE inline float GridScale(uint32_t level, uint32_t baseResolution, float levelScale)
{
    return float(baseResolution) * powf(levelScale, float(level)) - 1.0f;
}

OPENPGL_DEVICE inline uint32_t GridResolution(float scale)
{
    return uint32_t(ceilf(scale)) + 1;
}

OPENPGL_DEVICE inline uint32_t HashGridIndex(uint3 gridVertex, uint32_t levelTableSize, uint32_t resolution)
{
    const uint64_t denseTableSize = uint64_t(resolution) * uint64_t(resolution) * uint64_t(resolution);
    if (denseTableSize <= levelTableSize)
    {
        uint32_t denseIndex = resolution * (resolution * gridVertex.z + gridVertex.y) + gridVertex.x;
        return denseIndex % levelTableSize;
    }

    return SpatialHash(gridVertex.x, gridVertex.y, gridVertex.z) % levelTableSize;
}

OPENPGL_DEVICE inline float Quartic(float x, float invRadius)
{
    float u = x * invRadius;
    float tmp = fmaxf(1.0f - u * u, 0.0f);
    return (15.0f / 16.0f) * tmp * tmp;
}

OPENPGL_DEVICE inline float QuarticCDFDeriv(float x, float invRadius)
{
    return Quartic(x, invRadius) * invRadius;
}

OPENPGL_DEVICE inline float QuarticCDF(float x, float invRadius)
{
    float u = x * invRadius;
    float u2 = u * u;
    float u4 = u2 * u2;
    float v = (15.0f / 16.0f) * u * (1.0f - (2.0f / 3.0f) * u2 + (1.0f / 5.0f) * u4) + 0.5f;
    return fminf(1.0f, fmaxf(v, 0.f));
}

template <typename T, uint32_t numBins>
OPENPGL_DEVICE inline void OneBlobEncoding(float s, T *encoding)
{
    float invRadius = float(numBins);
    float leftCDF = QuarticCDF(-s, invRadius) /*+ QuarticCDF(-s - 1, invRadius) */ + QuarticCDF(-s + 1, invRadius);

    for (uint32_t bin = 0; bin < numBins; bin++)
    {
        float boundary = (bin + 1) / float(numBins);
        float rightCDF = QuarticCDF(boundary - s, invRadius) + QuarticCDF(boundary - s - 1, invRadius) + QuarticCDF(boundary - s + 1, invRadius);

        if constexpr (std::is_same_v<T, float>)
        {
            encoding[bin] = rightCDF - leftCDF;
        }
        else if constexpr (std::is_same_v<T, __half>)
        {
            encoding[bin] = __float2half(rightCDF - leftCDF);
        }
        leftCDF = rightCDF;
    }
}

template <typename T, uint32_t numLevels = OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_LEVELS, uint32_t numBins = OPENPGL_NEURAL_RADIANCE_CACHE_ONE_BLOB_BINS, uint32_t featuresPerHashEntry = OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_FEATURES>
OPENPGL_DEVICE inline void HashGridForward(T *featureVector,
                                           float3 position,
                                           float3 direction,
                                           const T *hashTable,
                                           const uint32_t *levelOffsets,
                                           uint32_t baseResolution,
                                           float levelScale)
{
    for (uint32_t level = 0; level < numLevels; level++)
    {
        const uint32_t levelOffset = levelOffsets[level];
        const uint32_t levelTableSize = levelOffsets[level + 1] - levelOffset;

        const float scale = GridScale(level, baseResolution, levelScale);
        const uint32_t resolution = GridResolution(scale);
        float3 gridPosition = position * scale + 0.5f;
        uint3 gridVertexLow = make_uint3(Floor(gridPosition));
        uint3 gridVertexHigh = make_uint3(gridVertexLow.x + 1, gridVertexLow.y + 1, gridVertexLow.z + 1);
        float3 weight = gridPosition - Floor(gridPosition);

        float feature[featuresPerHashEntry] = {0.f};
        for (int corner = 0; corner < 8; corner++)
        {
            uint3 gridVertex;
            gridVertex.x = (corner & 1) ? gridVertexHigh.x : gridVertexLow.x;
            gridVertex.y = ((corner >> 1) & 1) ? gridVertexHigh.y : gridVertexLow.y;
            gridVertex.z = (corner >> 2) ? gridVertexHigh.z : gridVertexLow.z;

            float cornerWeight = (corner & 1) ? weight.x : 1 - weight.x;
            cornerWeight *= ((corner >> 1) & 1) ? weight.y : 1 - weight.y;
            cornerWeight *= (corner >> 2) ? weight.z : 1 - weight.z;

            const uint32_t hashTableIndex = HashGridIndex(gridVertex, levelTableSize, resolution);

            for (uint32_t i = 0; i < featuresPerHashEntry; i++)
            {
                T entry = hashTable[featuresPerHashEntry * (levelOffset + hashTableIndex) + i];
                if constexpr (std::is_same_v<T, float>)
                {
                    feature[i] += entry * cornerWeight;
                }
                else if constexpr (std::is_same_v<T, __half>)
                {
                    feature[i] += __half2float(entry) * cornerWeight;
                }
            }
        }
        for (uint32_t i = 0; i < featuresPerHashEntry; i++)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                featureVector[featuresPerHashEntry * level + i] = feature[i];
            }
            else if constexpr (std::is_same_v<T, __half>)
            {
                featureVector[featuresPerHashEntry * level + i] = __float2half(feature[i]);
            }
        }
    }
    OneBlobEncoding<T, numBins>(direction.x, featureVector + featuresPerHashEntry * numLevels);
    OneBlobEncoding<T, numBins>(direction.y, featureVector + featuresPerHashEntry * numLevels + numBins);
    OneBlobEncoding<T, numBins>(direction.z, featureVector + featuresPerHashEntry * numLevels + 2 * numBins);
}

template <typename T, 
          uint32_t numLevels = OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_LEVELS,
          uint32_t featuresPerHashEntry = OPENPGL_NEURAL_RADIANCE_CACHE_GRID_NUM_FEATURES>
OPENPGL_DEVICE inline void HashGridBackward(const T &featureGradientVector,
                                            float3 position,
                                            float *hashTableGradients,
                                            const uint32_t *levelOffsets,
                                            uint32_t baseResolution,
                                            float levelScale)
{
    for (uint32_t level = 0; level < numLevels; level++)
    {
        const uint32_t levelOffset = levelOffsets[level];
        const uint32_t levelTableSize = levelOffsets[level + 1] - levelOffset;

        const float scale = GridScale(level, baseResolution, levelScale);
        const uint32_t resolution = GridResolution(scale);
        float3 gridPosition = position * scale + 0.5f;
        uint3 gridVertexLow = make_uint3(Floor(gridPosition));
        uint3 gridVertexHigh = make_uint3(gridVertexLow.x + 1, gridVertexLow.y + 1, gridVertexLow.z + 1);
        float3 weight = gridPosition - Floor(gridPosition);

        for (int corner = 0; corner < 8; corner++)
        {
            uint3 gridVertex;
            gridVertex.x = (corner & 1) ? gridVertexHigh.x : gridVertexLow.x;
            gridVertex.y = ((corner >> 1) & 1) ? gridVertexHigh.y : gridVertexLow.y;
            gridVertex.z = (corner >> 2) ? gridVertexHigh.z : gridVertexLow.z;

            float cornerWeight = (corner & 1) ? weight.x : 1 - weight.x;
            cornerWeight *= ((corner >> 1) & 1) ? weight.y : 1 - weight.y;
            cornerWeight *= (corner >> 2) ? weight.z : 1 - weight.z;

            const uint32_t hashTableIndex = HashGridIndex(gridVertex, levelTableSize, resolution);

            for (uint32_t i = 0; i < featuresPerHashEntry; i++)
            {
                const uint32_t featureIndex = featuresPerHashEntry * level + i;
                const float gradient = __half2float(featureGradientVector[featureIndex]);
                atomicAdd(&hashTableGradients[featuresPerHashEntry * (levelOffset + hashTableIndex) + i], gradient * cornerWeight);
            }
        }
    }
}

// in is (1 x weightRows)
inline void MatrixMult(float *out, float *in, float *weights, uint32_t weightRows, uint32_t weightCols)
{
    for (uint32_t c = 0; c < weightCols; c++)
    {
        float total = 0.f;
        for (uint32_t r = 0; r < weightRows; r++)
        {
            total += in[r] * weights[c * weightRows + r];
        }
        out[c] = fmaxf(0.f, total);
    }
}

}  // namespace openpgl
