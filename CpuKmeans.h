#pragma once
#include "KmeansCalculator.h"
class CpuKmeans : public KmeansCalculator
{
public:
	using KmeansCalculator::KmeansCalculator;
	void CalculateKmeans() override;
};

