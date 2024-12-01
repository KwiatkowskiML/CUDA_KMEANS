#pragma once
#include "KmeansCalculator.h"

__global__ void CalculateBelongings2(const float* clusters, const float* vectors, int* belonging, const int& N, const int& D, const int& K);

class GpuKmeans2 : public KmeansCalculator
{
public:
	using KmeansCalculator::KmeansCalculator;
	void CalculateKmeans() override;
};

