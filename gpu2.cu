#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include "gpu2.cuh"
#include "HostConstants.h"
#include <thrust/device_vector.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void calculateElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds, const char* message)
{
    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(milliseconds, start, stop));
    fprintf(stdout, "%s: %f\n", message, *milliseconds);
}

__global__ void CalculateBelongings2(const float* clusters, const float* vectors, int* belonging, const int& N, const int& D, const int& K)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;

    int min_cluster = 0;
    float min_distance = FLT_MAX;

    for (int i = 0; i < K; i++) {
        float distance = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = vectors[idx + j * N] - clusters[i + j * K];
            distance += diff * diff;
        }
        if (distance < min_distance) {
            min_distance = distance;
            min_cluster = i;
        }
    }
    belonging[idx] = min_cluster;
}

__global__ void AddKernel(float* clusters, const float* vectors, const int* belonging, const int& N, const int& D, const int& K)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < N * D)
    {
        int cord_idx = idx / N;
        int vec_idx = idx % N;

        int cluster_id = belonging[vec_idx];
        int cluster_offset = cord_idx * K + cluster_id;
        atomicAdd(clusters + cluster_offset, vectors[idx]);
    }
}

void CalculateKmean2(float* clusters, const float* vectors, int* belonging, int N, int K, int D)
{
    float* dev_clusters = 0;
    float* dev_vectors = 0;
    int* dev_belonging = 0;
    int* dev_n = 0;
    int* dev_k = 0;
    int* dev_d = 0;
    int* dev_cluster_count = 0;
    thrust::device_vector<int> vector_order(N);

    //-------------------------------
    //      TIME MEASUREMENT
    //-------------------------------

    float milliseconds = 0;
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    //-------------------------------
    //      DATA PREPARATION
    //-------------------------------

    gpuErrchk(cudaSetDevice(0));

    // Memory allocation on the side of the device
    gpuErrchk(cudaMalloc((void**)&dev_clusters, K * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_vectors, N * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_belonging, N * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_n, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_k, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_d, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_cluster_count, K * sizeof(int)));

    // Copying memory from host to device
    gpuErrchk(cudaMemcpy(dev_vectors, vectors, N * D * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_clusters, clusters, K * D * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_belonging, 0, N * sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_n, &N, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_k, &K, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_d, &D, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_cluster_count, 0, K * sizeof(int)));

    // pointers initialization

    //-------------------------------
    //            LOGIC
    //-------------------------------

    gpuErrchk(cudaEventRecord(start, 0));
    int block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    CalculateBelongings2 << <block_count, THREADS_PER_BLOCK >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);
    calculateElapsedTime(start, stop, &milliseconds, "Belonging");

    gpuErrchk(cudaMemset(dev_clusters, 0, K * D * sizeof(float)));

    gpuErrchk(cudaEventRecord(start, 0));
    block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    AddKernel << <1, N * D>> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);
    calculateElapsedTime(start, stop, &milliseconds, "Adding v1");


    //-------------------------------
    //         END OF LOGIC
    //-------------------------------

    // error checking and synchronization
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());


    // Copy memory back to the host
    gpuErrchk(cudaMemcpy(clusters, dev_clusters, K * D * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(belonging, dev_belonging, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleaning
    gpuErrchk(cudaFree(dev_clusters));
    gpuErrchk(cudaFree(dev_vectors));
    gpuErrchk(cudaFree(dev_belonging));
    gpuErrchk(cudaFree(dev_n));
    gpuErrchk(cudaFree(dev_k));
    gpuErrchk(cudaFree(dev_d));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
}