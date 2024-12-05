#include "GpuKmeans2.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#define DEEP_TIME_ANALYSIS

__global__ void CalculateBelongings2(const float* clusters, const float* vectors, int* belonging, const int& N, const int& D, const int& K, int* vectors_moved)
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

    if (belonging[idx] != min_cluster)
    {
        vectors_moved[idx] = 1;   
    }

    for (int i = 0; i < D; i++)
    {
        belonging[idx + i * N] = i * K + min_cluster;
    }
}

__global__ void CalculateClusterMean2(float* clusters, const int* cluster_count, const int& D, const int& K)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cluster_id = idx % K;
    clusters[idx] /= cluster_count[cluster_id];
}

void GpuKmeans2::CalculateKmeans()
{
    int N = vectorsStorage->getNumPoints();
    int D = vectorsStorage->getNumDimensions();
    int K = vectorsStorage->getNumClusters();
    float* vectors = vectorsStorage->vectors;
    float* clusters = vectorsStorage->clusters;
    int* belonging = vectorsStorage->belonging;
    int vectors_moved_count = 0;
    float milliseconds = 0;
    int iter = 0;

    float* dev_clusters = 0;
    float* dev_vectors = 0;
    int* dev_belonging = 0;
    int* dev_n = 0;
    int* dev_k = 0;
    int* dev_d = 0;
    int* dev_cluster_count = 0;
    int* dev_vectors_moved = 0;
    thrust::device_vector<int> vector_order(N);
    cudaError_t cudaStatus;

    // pointers initialization
    thrust::device_ptr<int> keys(dev_belonging);
    thrust::device_ptr<float> vals(dev_vectors);
    thrust::device_ptr<float> clusters_ptr(dev_clusters);
    thrust::device_ptr<int> cluster_count_ptr(dev_cluster_count);
    thrust::constant_iterator<int> const_iter(1);
    thrust::counting_iterator<int> count_iter(0);
    thrust::equal_to<int> binary_pred;
    thrust::copy(count_iter, count_iter + N, vector_order.begin());

    //-------------------------------
    //      TIME MEASUREMENT
    //-------------------------------
    
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

    // Memory allocation on the side of the device
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

    cudaStatus = cudaMalloc((void**)&dev_belonging, N * D * sizeof(int));
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

    // Copying memory from host to device
    cudaStatus = cudaMemcpy(dev_vectors, vectors, N * D * sizeof(float), cudaMemcpyHostToDevice);
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

    cudaStatus = cudaMemset(dev_belonging, 0, N * D * sizeof(int));
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

    cudaStatus = cudaMemset(dev_vectors_moved, 0, N * sizeof(int));
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
        vectors_moved_count = 0;
        cudaStatus = cudaMemset(dev_vectors_moved, 0, N * sizeof(int));
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
        // calculating each vectors belonging
        int block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        CalculateBelongings2 << <block_count, THREADS_PER_BLOCK >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k, dev_vectors_moved);
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
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Calculate belongings");
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
        // calculating the number of moved vectors
        vectors_moved_count = thrust::reduce(thrust::device, dev_vectors_moved, dev_vectors_moved + N);

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "reduce 1");
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif

        printf("Iteration %d, number of vectors moved: %d\n", iter, vectors_moved_count);
        if (vectors_moved_count == 0)
            break;
        
#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = cudaEventRecord(start, 0);
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
#endif
        // sorting the order of vectors to keep track of the initial vector order
        printf("test\n");
        thrust::sort_by_key(keys, keys + N, thrust::make_zip_iterator(thrust::make_tuple(vector_order.begin(), vals)));
        printf("test\n");

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Sorting vector order");
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

        // sorting the rest of the belongings
        thrust::sort_by_key(keys + N, keys + N * D, vals + N);
        //thrust::sort_by_key(keys, keys + N * D, vals);

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Sorting belongings");
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

        // summing vectors in each cluster
        thrust::reduce_by_key(keys, keys + N * D, vals, thrust::make_discard_iterator(), clusters_ptr, binary_pred);

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

        // counting number of vectors in each cluster
        thrust::reduce_by_key(keys, keys + N, const_iter, thrust::make_discard_iterator(), cluster_count_ptr, binary_pred);

#ifdef DEEP_TIME_ANALYSIS
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Counting number of vectors in each cluster");
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

        // Updating cluster means
        CalculateClusterMean2 << <1, K * D >> > (dev_clusters, dev_cluster_count, *dev_d, *dev_k);
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
        cudaStatus = calculateElapsedTime(start, stop, &milliseconds, "Calculating means");
        if (cudaStatus != cudaSuccess)
        {
            gpuError(cudaStatus);
            goto Error;
        }
        printf("\n");
#endif

        // increment iteration count
        iter++;
    } while (vectors_moved_count > 0 && iter < MAX_ITERATIONS);

    // back to original vectors order
    thrust::sort_by_key(vector_order.begin(), vector_order.end(), keys);

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

    // copying time
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