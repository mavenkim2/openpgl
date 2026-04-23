#pragma once

#include "../../data/SampleData.h"

namespace openpgl 
{

struct VarianceAccumulator 
{
    void accumulateSamples(const SampleData *samples, const size_t numSamples);
    void decay(const float alpha);

    float varianceDeltaEstimate;
    uint32_t totalNumSamples;
};

void VarianceAccumulator::accumulateSamples(const SampleData *samples, const size_t numSamples)
{
    for (size_t n = 0; n < numSamples; n++)
    {
        const SampleData &sampleData = samples[n];
        float varianceDeltaEstimator = float(sampleData.risSampleCount) / (float(sampleData.risSampleCount) - 1.f);
        varianceDeltaEstimator *= sampleData.risWeight * sampleData.weight * sampleData.weight;
        varianceDeltaEstimator *= 1.f / sampleData.sourceFunction - sampleData.risWeight;

        totalNumSamples++;
        varianceDeltaEstimate += (varianceDeltaEstimator - varianceDeltaEstimate) / float(totalNumSamples);
        // 1. "Lr denotes the estimator of the radiance at the end of the path prefix reflected towards the preceeding vertex." Does 
        // sample data actually store this?
        // 2. Do I have to decay this variance estimate?
    }
}

void VarianceAccumulator::decay(const float alpha)
{
    varianceDeltaEstimate *= alpha;
}

}