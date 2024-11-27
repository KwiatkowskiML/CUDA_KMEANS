#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gpu1.cuh"

cudaError_t CalculateKmean(float* clusters, const float* vectors, int* belonging, const int& N, const int& K, const int& D)
{
    float* dev_clusters = 0;
    float* dev_vectors = 0;
    int* dev_belonging = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_clusters, K * D * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_clusters!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vectors, N * D * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_vectors!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_belonging, N * D * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_belonging!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_vectors, vectors, N * D * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_vectors!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_clusters, dev_vectors, K * D * sizeof(float), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for copying dev_vectors to dev_clusters!");
        goto Error;
    } 

    cudaStatus = cudaMemset(dev_belonging, 1, N * D * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed for dev_belonging!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(clusters, dev_clusters, K * D * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for clusters!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(belonging, dev_belonging, N * D * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for belonging!");
        goto Error;
    }

Error:
    cudaFree(dev_clusters);
    cudaFree(dev_vectors);
    cudaFree(dev_belonging);

    return cudaStatus;
}