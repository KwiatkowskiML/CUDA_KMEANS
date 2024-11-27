#pragma once
#include "cuda_runtime.h"

cudaError_t CalculateKmean(float* clusters, const float* vectors, int* belonging, const int& N, const int& K, const int& D);