#pragma once
#include "cuda_runtime.h"

// Function declaration for adding vectors using CUDA
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);