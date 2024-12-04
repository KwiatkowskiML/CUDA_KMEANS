#pragma once
#include "KmeansCalculator.h"

class GpuKmeans2 : public KmeansCalculator
{
public:
	using KmeansCalculator::KmeansCalculator;
	void CalculateKmeans() override;
};

