#pragma once
#include "KmeansCalculator.h"

class GpuKmeans1 : public KmeansCalculator
{
public:
	using KmeansCalculator::KmeansCalculator;
	void CalculateKmeans() override;
};

