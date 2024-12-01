#pragma once
#include "KmeansCalculator.h"

__global__ void CalculateBelongings(const float* clusters, const float* vectors, int* belonging, int* cluster_count, const int& N, const int& D, const int& K);
__global__ void AddKernel(float* clusters, const float* vectors, const int* belonging, const int& N, const int& D, const int& K);

class GpuKmeans1 : public KmeansCalculator
{
public:
	using KmeansCalculator::KmeansCalculator;
	void CalculateKmeans() override;
};

