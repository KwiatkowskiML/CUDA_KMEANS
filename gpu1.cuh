#pragma once
#include "cuda_runtime.h"

void CalculateKmean(float* clusters, const float* vectors, int* belonging, int N, int K, int D);

__global__ void CalculateBelongings(const float* clusters, const float* vectors, int* belonging, const int& N, const int& D, const int& K);