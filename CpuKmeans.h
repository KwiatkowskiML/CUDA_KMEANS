#pragma once
#include "KmeansCalculator.h"
class CpuKmeans : public KmeansCalculator
{
private:
	int CalculateBelongingCpu(const float* vectors, const float* clusters, int* belonging, int* cluster_count, const int& N, const int& D, const int& K);
	void CalculateClusters(const float* vectors, float* clusters, const int* belonging, const int* cluster_count, const int& N, const int& D, const int& K);
public:
	using KmeansCalculator::KmeansCalculator;
	void CalculateKmeans() override;
};

