#include "KmeansCalculator.h"
#include "CpuKmeans.h"

cudaError_t KmeansCalculator::calculateElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds, const char* message)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaEventRecord(stop, 0);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaEventElapsedTime(milliseconds, start, stop);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    fprintf(stdout, "%s: %f ms\n", message, *milliseconds);
    return cudaStatus;
}

void KmeansCalculator::gpuAssert(cudaError_t code, const char* file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}