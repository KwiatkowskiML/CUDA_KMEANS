#include "KmeansCalculator.h"
#include "CpuKmeans.h"

void KmeansCalculator::calculateElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds, const char* message)
{
    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(milliseconds, start, stop));
    fprintf(stdout, "%s: %f ms\n", message, *milliseconds);
}

void KmeansCalculator::gpuAssert(cudaError_t code, const char* file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}