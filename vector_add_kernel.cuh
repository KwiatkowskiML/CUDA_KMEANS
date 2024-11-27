#ifndef VECTOR_ADD_KERNEL_H
#define VECTOR_ADD_KERNEL_H

#include "cuda_runtime.h"

// Function declaration for adding vectors using CUDA
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

#endif // VECTOR_ADD_KERNEL_H