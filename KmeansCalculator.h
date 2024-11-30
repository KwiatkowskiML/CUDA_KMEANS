#pragma once
#include "Vectors.h"
#include "cuda_runtime.h"

#define gpuErrchk(ans) { KmeansCalculator::gpuAssert((ans), __FILE__, __LINE__); }

class KmeansCalculator {
protected:
    Vectors* vectors;
public:
    KmeansCalculator(Vectors& inputVectors): vectors(&inputVectors){}

    virtual ~KmeansCalculator() = default;

    virtual void CalculateKmeans() = 0;

    void calculateElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds, const char* message);

    inline static void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);
};