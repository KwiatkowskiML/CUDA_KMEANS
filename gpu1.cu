#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gpu1.cuh"
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define DEBUG

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void CalculateBelongings(const float* clusters, const float* vectors, int* belonging, const int& N, const int& D, const int& K)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int min_cluster = 0;
    float min_distance = FLT_MAX;

    for (int i = 0; i < K; i++) {
        float distance = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = vectors[idx * D + j] - clusters[i * D + j];
            distance += diff * diff;
        }
        if (distance < min_distance) {
            min_distance = distance;
            min_cluster = i;
        }
    }

    for (int i = 0; i < D; i++)
    {
        belonging[idx + i * N] = i * K + min_cluster;
    }
}

void CalculateKmean(float* clusters, const float* vectors, int* belonging, int N, int K, int D)
{
    float* dev_clusters = 0;
    float* dev_vectors = 0;
    int* dev_belonging = 0;
    int* dev_n = 0;
    int* dev_k = 0;
    int* dev_d = 0;

    gpuErrchk(cudaSetDevice(0));

    // Memory allocation on the side of the device
    gpuErrchk(cudaMalloc((void**)&dev_clusters, K * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_vectors, N * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_belonging, N * D * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_n, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_k, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_d, sizeof(int)));

    // Copying memory from host to device
    gpuErrchk(cudaMemcpy(dev_vectors, vectors, N * D * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_clusters, clusters, K * D * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_belonging, 0, N * D * sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_n, &N, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_k, &K, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_d, &D, sizeof(int), cudaMemcpyHostToDevice));

    //-------------------------------
    //            LOGIC
    //-------------------------------

    CalculateBelongings << <1, N >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);

    gpuErrchk(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.cudaStatus = cudaDeviceSynchronize();
    gpuErrchk(cudaDeviceSynchronize())    

    // sorting 
    thrust::device_ptr<int> keys(dev_belonging);
    thrust::device_ptr<float> vals(dev_vectors);
    thrust::sort_by_key(keys, keys + N * D, vals);

    //-------------------------------
    //         END OF LOGIC
    //-------------------------------


    // Copy memory back to the host
    gpuErrchk(cudaMemcpy(clusters, dev_clusters, K * D * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(belonging, dev_belonging, N * D * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleaning
    gpuErrchk(cudaFree(dev_clusters));
    gpuErrchk(cudaFree(dev_vectors));
    gpuErrchk(cudaFree(dev_belonging));
    gpuErrchk(cudaFree(dev_n));
    gpuErrchk(cudaFree(dev_k));
    gpuErrchk(cudaFree(dev_d));
}