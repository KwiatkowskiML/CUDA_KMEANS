#pragma once
#include "Vectors.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "HostConstants.h"

//#define gpuErrchk(ans) { KmeansCalculator::gpuAssert((ans), __FILE__, __LINE__); }
#define gpuError(cudaStatus) { printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(cudaStatus), __FILE__, __LINE__); }

class KmeansCalculator {
protected:
    Vectors* vectorsStorage;
    cudaError_t calculateElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds, const char* message);

public:
    KmeansCalculator(Vectors& inputVectors): vectorsStorage(&inputVectors){}

    virtual ~KmeansCalculator() = default;

    virtual void CalculateKmeans() = 0;

    static void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);
};