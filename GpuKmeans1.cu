#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GpuKmeans1.cuh"

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

__global__ void CalculateBelongings(const float* clusters, const float* vectors, int* belonging, int* cluster_count, const int& N, const int& D, const int& K, int* vectors_moved)
{
    int idx =  blockDim.x* blockIdx.x + threadIdx.x;

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

    if (belonging[idx] != min_cluster)
    {
        belonging[idx] = min_cluster;
        vectors_moved[idx] = 1;
    }
    
    atomicAdd(cluster_count + min_cluster, 1);
}

__global__ void CalculateClusters(float* clusters, const int* cluster_count, const int& D, const int& K)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < D; i++)
    {
        clusters[idx + K * i] /= cluster_count[idx];
    }
}

__global__ void AddKernel2(float* clusters, const float* vectors, const int* belonging, const int& N, const int& D, const int& K)
{
    extern __shared__ float shared_clusters[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadIdx.x < K * D)
    {
        shared_clusters[threadIdx.x] = 0;
    }

    __syncthreads();

    if (idx < N * D)
    {
        int cord_idx = idx / N;
        int vec_idx = idx % N;

        int cluster_id = belonging[vec_idx];
        int cluster_offset = cord_idx * K + cluster_id;
        atomicAdd_block(shared_clusters + cluster_offset, vectors[idx]);
    }

    __syncthreads();

    if (threadIdx.x < K * D)
    {
        atomicAdd(clusters + threadIdx.x, shared_clusters[threadIdx.x]);
    }
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

void GpuKmeans1::CalculateKmeans()
{
    int N = vectorsStorage->getNumPoints();
    int D = vectorsStorage->getNumDimensions();
    int K = vectorsStorage->getNumClusters();
    float* vectors_array = vectorsStorage->vectors;
    float* clusters = vectorsStorage->clusters;
    int* belonging = vectorsStorage->belonging;

    float* dev_clusters = 0;
    float* dev_vectors = 0;
    int* dev_belonging = 0;
    int* dev_n = 0;
    int* dev_k = 0;
    int* dev_d = 0;
    int* dev_cluster_count = 0;
    int* dev_vectors_moved = 0;

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

    // Device setup
    gpuErrchk(cudaSetDevice(0));

    // Memory allocation on the side of the device
    gpuErrchk(cudaMalloc((void**)&dev_clusters, K * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_vectors, N * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_belonging, N * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_n, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_k, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_d, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_cluster_count, K * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_vectors_moved, N * sizeof(int)));

    // Copying memory from host to device
    gpuErrchk(cudaMemcpy(dev_vectors, vectors_array, N * D * sizeof(float), cudaMemcpyHostToDevice));
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

    printf("\nStarting calculation\n");
    int iter = 0;
    int vectors_moved_count;

    do {
        gpuErrchk(cudaMemset(dev_vectors_moved, 0, N * sizeof(int)));
        gpuErrchk(cudaMemset(dev_cluster_count, 0, K * sizeof(int)));

#ifdef DEEP_TIME_ANALYTICS
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // find closest cluster for each vector
        int block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        CalculateBelongings << <block_count, THREADS_PER_BLOCK >> > (dev_clusters, dev_vectors, dev_belonging, dev_cluster_count, *dev_n, *dev_d, *dev_k, dev_vectors_moved);

#ifdef DEEP_TIME_ANALYTICS
        calculateElapsedTime(start, stop, &milliseconds, "Belonging calculation");
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // count moved vectors
        vectors_moved_count = thrust::reduce(thrust::device, dev_vectors_moved, dev_vectors_moved + N);

#ifdef DEEP_TIME_ANALYTICS
        calculateElapsedTime(start, stop, &milliseconds, "Moved vectors calculation (reduction)");
#endif

        printf("Iteration %d, moved %d vectors\n", iter, vectors_moved_count);
        if (vectors_moved_count == 0)
            break;

        gpuErrchk(cudaMemset(dev_clusters, 0, K * D * sizeof(float)));

#ifdef DEEP_TIME_ANALYTICS
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // Sum vectors in each cluster
        block_count = (N * D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        AddKernel2 << <block_count, THREADS_PER_BLOCK, K * D * sizeof(float) >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);
        // AddKernel << <block_count, THREADS_PER_BLOCK >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);


#ifdef DEEP_TIME_ANALYTICS
        calculateElapsedTime(start, stop, &milliseconds, "Summing vectors in each cluster");
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // Calculate each cluster's mean
        CalculateClusters << <1, K >> > (dev_clusters, dev_cluster_count, *dev_d, *dev_k);

#ifdef DEEP_TIME_ANALYTICS
        calculateElapsedTime(start, stop, &milliseconds, "Calculating cluster means");
#endif

        iter++;
    } while (iter < MAX_ITERATIONS && vectors_moved_count > 0);


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
    gpuErrchk(cudaFree(dev_vectors_moved));
}
