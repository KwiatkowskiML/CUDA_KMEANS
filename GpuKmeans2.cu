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

    float* dev_clusters = 0;
    float* dev_vectors = 0;
    int* dev_belonging = 0;
    int* dev_n = 0;
    int* dev_k = 0;
    int* dev_d = 0;
    int* dev_cluster_count = 0;
    thrust::device_vector<int> vector_order(N);

    // temp
    int* dev_vectors_moved = 0;
    gpuErrchk(cudaMalloc((void**)&dev_vectors_moved, N * sizeof(int)));
    gpuErrchk(cudaMemset(dev_vectors_moved, 0, N * sizeof(int)));

    //-------------------------------
    //      TIME MEASUREMENT
    //-------------------------------

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //-------------------------------
    //      DATA PREPARATION
    //-------------------------------

    gpuErrchk(cudaSetDevice(0));

    // time measurment
    printf("Copying data...\n");
    gpuErrchk(cudaEventRecord(start, 0));

    // Memory allocation on the side of the device
    gpuErrchk(cudaMalloc((void**)&dev_clusters, K * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_vectors, N * D * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_belonging, N * D * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_n, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_k, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_d, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_cluster_count, K * sizeof(int)));

    // Copying memory from host to device
    gpuErrchk(cudaMemcpy(dev_vectors, vectors, N * D * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_clusters, clusters, K * D * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_belonging, 0, N * D * sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_n, &N, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_k, &K, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_d, &D, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_cluster_count, 0, K * sizeof(int)));

    // time measurment
    calculateElapsedTime(start, stop, &milliseconds, "Copying data from host to device took");

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
    //            LOGIC
    //-------------------------------
#ifndef DEEP_TIME_ANALYSIS
    printf("Calculating kmeans...\n");
    gpuErrchk(cudaEventRecord(start, 0));
#endif

    int iter = 0;
    do {
        vectors_moved_count = 0;
        gpuErrchk(cudaMemset(dev_vectors_moved, 0, N * sizeof(int)));
        
#ifdef DEEP_TIME_ANALYSIS
        gpuErrchk(cudaEventRecord(start, 0));
#endif
        // calculating belongings for each vector
        int block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        CalculateBelongings2 << <block_count, THREADS_PER_BLOCK >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k, dev_vectors_moved);

#ifdef DEEP_TIME_ANALYSIS
        calculateElapsedTime(start, stop, &milliseconds, "Calculate belongings");
        gpuErrchk(cudaEventRecord(start, 0));
#endif
        // calculating the number of moved vectors
        vectors_moved_count = thrust::reduce(thrust::device, dev_vectors_moved, dev_vectors_moved + N);

#ifdef DEEP_TIME_ANALYSIS
        calculateElapsedTime(start, stop, &milliseconds, "reduce 1");
#endif

        printf("Iteration %d, number of vectors moved: %d\n", iter, vectors_moved_count);
        if (vectors_moved_count == 0)
            break;

        
#ifdef DEEP_TIME_ANALYSIS
        gpuErrchk(cudaEventRecord(start, 0));
#endif
        // sorting the order of vectors, accordingly to the current order of belonging
        thrust::sort_by_key(keys, keys + N, thrust::make_zip_iterator(thrust::make_tuple(vector_order.begin(), vals)));

#ifdef DEEP_TIME_ANALYSIS
        calculateElapsedTime(start, stop, &milliseconds, "Sorting vector order");
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // sorting the rest of the belongings
        thrust::sort_by_key(keys + N, keys + N * D, vals + N);
        //thrust::sort_by_key(keys, keys + N * D, vals);

#ifdef DEEP_TIME_ANALYSIS
        calculateElapsedTime(start, stop, &milliseconds, "Sorting belongings");
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // summing vectors in each cluster
        thrust::reduce_by_key(keys, keys + N * D, vals, thrust::make_discard_iterator(), clusters_ptr, binary_pred);

#ifdef DEEP_TIME_ANALYSIS
        calculateElapsedTime(start, stop, &milliseconds, "Summing vectors in each cluster");
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // counting number of vectors in each cluster
        thrust::reduce_by_key(keys, keys + N, const_iter, thrust::make_discard_iterator(), cluster_count_ptr, binary_pred);

#ifdef DEEP_TIME_ANALYSIS
        calculateElapsedTime(start, stop, &milliseconds, "Counting number of vectors in each cluster");
        gpuErrchk(cudaEventRecord(start, 0));
#endif

        // Updating cluster means
        CalculateClusterMean2 << <1, K * D >> > (dev_clusters, dev_cluster_count, *dev_d, *dev_k);

#ifdef DEEP_TIME_ANALYSIS
        calculateElapsedTime(start, stop, &milliseconds, "Calculating means");
        printf("\n");
#endif

        // increment iteration count
        iter++;
    } while (vectors_moved_count > 0 && iter < MAX_ITERATIONS);

    // back to original vectors order
    thrust::sort_by_key(vector_order.begin(), vector_order.end(), keys);

#ifndef DEEP_TIME_ANALYSIS
    calculateElapsedTime(start, stop, &milliseconds, "Calculating kmeans took");
    gpuErrchk(cudaEventRecord(start, 0));
#endif

    //-------------------------------
    //         END OF LOGIC
    //-------------------------------

    // error checking and synchronization
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // copying time
    printf("Copying data...\n");
    gpuErrchk(cudaEventRecord(start, 0));

    // Copy memory back to the host
    gpuErrchk(cudaMemcpy(clusters, dev_clusters, K * D * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(belonging, dev_belonging, N * sizeof(int), cudaMemcpyDeviceToHost));

    // time measurment
    calculateElapsedTime(start, stop, &milliseconds, "Copying data from device to host took");

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