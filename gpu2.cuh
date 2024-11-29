#pragma once
#include "cuda_runtime.h"

void CalculateKmean2(float* clusters, const float* vectors, int* belonging, int N, int K, int D);

void calculateElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds, const char* miliseconds);

__global__ void CalculateBelongings2(const float* clusters, const float* vectors, int* belonging, const int& N, const int& D, const int& K);