#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GpuKmeans1.cuh"

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

//#define DEEP_TIME_ANALYSIS

__global__ void CalculateBelongings(const float* clusters, const float* vectors, int* belonging, int* cluster_count, const int& N, const int& D, const int& K, int* vectors_moved)
{
    int idx =  blockDim.x * blockIdx.x + threadIdx.x;

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

__global__ void CalculateBelongingsShared(const float* clusters, const float* vectors, int* belonging, int* cluster_count, const int& N, const int& D, const int& K, int* vectors_moved)
{
    extern __shared__ float shared_clusters[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;

    if (threadIdx.x < K * D)
    {
        shared_clusters[threadIdx.x] = clusters[threadIdx.x];

        if (blockDim.x < K * D)
        {
            int next_cluster = threadIdx.x + blockDim.x;
            while (next_cluster < K * D)
            {
                shared_clusters[next_cluster] = clusters[next_cluster];
                next_cluster += blockDim.x;
            }
        }
    }

    __syncthreads();

    int min_cluster = 0;
    float min_distance = FLT_MAX;

    for (int i = 0; i < K; i++) {
        float distance = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = vectors[idx + j * N] - shared_clusters[i + j * K];
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

__global__ void CalculateClusterMean(float* clusters, const int* cluster_count, const int& D, const int& K)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cluster_id = idx % K;
    clusters[idx] /= cluster_count[cluster_id];
}

__global__ void AddKernel2(float* clusters, const float* vectors, const int* belonging, const int& N, const int& D, const int& K)
{
    extern __shared__ float shared_clusters[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadIdx.x < K * D)
    {
        shared_clusters[threadIdx.x] = 0;

        if (blockDim.x < K * D)
        {
            int i = threadIdx.x + blockDim.x;
            while (i < K * D)
            {
                shared_clusters[i] = 0;
                i += blockDim.x;
            }
        }
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
        if (shared_clusters[threadIdx.x] != 0)
            atomicAdd(clusters + threadIdx.x, shared_clusters[threadIdx.x]);

        if (blockDim.x < K * D)
        {
            int i = threadIdx.x + blockDim.x;
            while (i < K * D)
            {
                if (shared_clusters[i] != 0)
                    atomicAdd(clusters + i, shared_clusters[i]);
                i += blockDim.x;
            }
        }
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

    cudaError_t cudaStatus;
    int iter = 0;
    int vectors_moved_count = 0;

    //-------------------------------
    //      TIME MEASUREMENT
    //-------------------------------

    float milliseconds = 0;
    cudaEvent_t start, stop;

    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    //-------------------------------
    //      DATA PREPARATION
    //-------------------------------

    // Device setup
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    // time measurment
    printf("Copying data...\n");
    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    //-------------------------------
    //      MEMORY ALLOCATION
    //-------------------------------
    cudaStatus = cudaMalloc((void**)&dev_clusters, K * D * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vectors, N * D * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_belonging, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_n, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_k, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_d, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_cluster_count, K * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_vectors_moved, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    //-------------------------------
    //      COPYING MEMORY
    //-------------------------------

    cudaStatus = cudaMemcpy(dev_vectors, vectors_array, N * D * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(dev_clusters, clusters, K * D * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMemset(dev_belonging, 0, N * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(dev_n, &N, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_k, &K, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(dev_d, &D, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
    
    cudaStatus = cudaMemset(dev_cluster_count, 0, K * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    // time measurment
    cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Copying data from host to device took");
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    //-------------------------------
    //            LOGIC
    //-------------------------------

#ifndef DEEP_TIME_ANALYSIS
    printf("Calculating kmeans...\n");
    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
#endif

    do {
        cudaStatus = cudaMemset(dev_vectors_moved, 0, N * sizeof(int));
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemset(dev_cluster_count, 0, K * sizeof(int));
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = cudaEventRecord(start, 0);
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif

        // find closest cluster for each vector
        int block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        CalculateBelongingsShared << <block_count, THREADS_PER_BLOCK, K * D * sizeof(float) >> > (dev_clusters, dev_vectors, dev_belonging, dev_cluster_count, *dev_n, *dev_d, *dev_k, dev_vectors_moved);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Belonging calculation");
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

        cudaStatus = cudaEventRecord(start, 0);
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif

        // count moved vectors
        vectors_moved_count = thrust::reduce(thrust::device, dev_vectors_moved, dev_vectors_moved + N);

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Moved vectors calculation (reduction)");
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif

        printf("Iteration %d, moved %d vectors\n", iter, vectors_moved_count);
        if (vectors_moved_count == 0)
            break;

        cudaStatus = cudaMemset(dev_clusters, 0, K * D * sizeof(float));
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = cudaEventRecord(start, 0);
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif

        // Sum vectors in each cluster
        block_count = (N * D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        AddKernel2 << <block_count, THREADS_PER_BLOCK, K * D * sizeof(float) >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);
        //AddKernel << <block_count, THREADS_PER_BLOCK >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Summing vectors in each cluster");
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

        cudaStatus = cudaEventRecord(start, 0);
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif

        // Calculate each cluster's mean
        CalculateClusterMean << <1, K * D>> > (dev_clusters, dev_cluster_count, *dev_d, *dev_k);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Calculating cluster means");
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif

        iter++;
    } while (iter < MAX_ITERATIONS && vectors_moved_count > 0);

#ifndef DEEP_TIME_ANALYSIS
    cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Calculating kmeans took");
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }
#endif

    //-------------------------------
    //         END OF LOGIC
    //-------------------------------

    // error checking and synchronization
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    printf("Copying data...\n");
    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    // Copy memory back to the host
    cudaStatus = cudaMemcpy(clusters, dev_clusters, K * D * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(belonging, dev_belonging, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

    // time measurment
    cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Copying data from device to host took");
    if (cudaStatus != cudaSuccess)
    {
        gpuError(cudaStatus);
        goto Error;
    }

Error:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_clusters);
    cudaFree(dev_vectors);
    cudaFree(dev_belonging);
    cudaFree(dev_n);
    cudaFree(dev_k);
    cudaFree(dev_d);
    cudaFree(dev_vectors_moved);
}
