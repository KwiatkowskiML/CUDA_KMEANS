//#include "cuda_runtime.h"
//
//#include "device_launch_parameters.h"
//#include "gpu1.cuh"
//#include "HostConstants.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <thrust/sort.h>
//#include <thrust/device_ptr.h>
//#include <thrust/iterator/discard_iterator.h>
//#include <thrust/iterator/constant_iterator.h>
//#include <thrust/iterator/zip_iterator.h>
//#include <thrust/tuple.h>
//#include <thrust/device_vector.h>
//
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//
//inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
//{
//    if (code != cudaSuccess)
//    {
//        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//        if (abort) exit(code);
//    }
//}
//
//void calculateElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds, const char* message)
//{
//    gpuErrchk(cudaEventRecord(stop, 0));
//    gpuErrchk(cudaEventSynchronize(stop));
//    gpuErrchk(cudaEventElapsedTime(milliseconds, start, stop));
//    fprintf(stdout, "%s: %f\n", message, *milliseconds);
//}
//
//__global__ void CalculateBelongings(const float* clusters, const float* vectors, int* belonging, const int& N, const int& D, const int& K)
//{
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    
//    if (idx >= N)
//        return;
//
//    int min_cluster = 0;
//    float min_distance = FLT_MAX;
//
//    for (int i = 0; i < K; i++) {
//        float distance = 0.0f;
//        for (int j = 0; j < D; j++) {
//            float diff = vectors[idx + j * N] - clusters[i + j * K];
//            distance += diff * diff;
//        }
//        if (distance < min_distance) {
//            min_distance = distance;
//            min_cluster = i;
//        }
//    }
//
//    for (int i = 0; i < D; i++)
//    {
//        belonging[idx + i * N] = i * K + min_cluster;
//    }
//}
//
//__global__ void CalculateClusters(float* clusters, const int* cluster_count, const int& D, const int& K)
//{
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//    for (int i = 0; i < D; i++)
//    {
//        clusters[idx + K * i] /= cluster_count[idx];
//    }
//}
//
//void CalculateKmean(float* clusters, const float* vectors, int* belonging, int N, int K, int D)
//{
//    float* dev_clusters = 0;
//    float* dev_vectors = 0;
//    int* dev_belonging = 0;
//    int* dev_n = 0;
//    int* dev_k = 0;
//    int* dev_d = 0;
//    int* dev_cluster_count = 0;
//    thrust::device_vector<int> vector_order(N);
//
//    //-------------------------------
//    //      TIME MEASUREMENT
//    //-------------------------------
//
//    float milliseconds = 0;
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//
//    //-------------------------------
//    //      DATA PREPARATION
//    //-------------------------------
//    
//    gpuErrchk(cudaSetDevice(0));
//
//    // Memory allocation on the side of the device
//    gpuErrchk(cudaMalloc((void**)&dev_clusters, K * D * sizeof(float)));
//    gpuErrchk(cudaMalloc((void**)&dev_vectors, N * D * sizeof(float)));
//    gpuErrchk(cudaMalloc((void**)&dev_belonging, N * D * sizeof(int)));
//    gpuErrchk(cudaMalloc((void**)&dev_n, sizeof(int)));
//    gpuErrchk(cudaMalloc((void**)&dev_k, sizeof(int)));
//    gpuErrchk(cudaMalloc((void**)&dev_d, sizeof(int)));
//    gpuErrchk(cudaMalloc((void**)&dev_cluster_count, K * sizeof(int)));
//
//    // Copying memory from host to device
//    gpuErrchk(cudaMemcpy(dev_vectors, vectors, N * D * sizeof(float), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(dev_clusters, clusters, K * D * sizeof(float), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemset(dev_belonging, 0, N * D * sizeof(int)));
//    gpuErrchk(cudaMemcpy(dev_n, &N, sizeof(int), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(dev_k, &K, sizeof(int), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(dev_d, &D, sizeof(int), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemset(dev_cluster_count, 0, K * sizeof(int)));
//
//    // pointers initialization
//    thrust::device_ptr<int> keys(dev_belonging);
//    thrust::device_ptr<float> vals(dev_vectors);
//    thrust::device_ptr<float> clusters_ptr(dev_clusters);
//    thrust::device_ptr<int> cluster_count_ptr(dev_cluster_count);
//    thrust::constant_iterator<int> const_iter(1);
//    thrust::counting_iterator<int> count_iter(0);
//    thrust::equal_to<int> binary_pred;
//    thrust::copy(count_iter, count_iter + N, vector_order.begin());
//
//    //-------------------------------
//    //            LOGIC
//    //-------------------------------
//   
//    cudaEventRecord(start, 0);
//
//    int block_count = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//    CalculateBelongings << <block_count, THREADS_PER_BLOCK >> > (dev_clusters, dev_vectors, dev_belonging, *dev_n, *dev_d, *dev_k);
//
//    calculateElapsedTime(start, stop, &milliseconds, "Calculate belongings");
//    cudaEventRecord(start, 0);
//
//    // sorting 
//    thrust::sort_by_key(keys, keys + N, thrust::make_zip_iterator(thrust::make_tuple(vector_order.begin(), vals)));
//
//    calculateElapsedTime(start, stop, &milliseconds, "Sorting vector order");
//    cudaEventRecord(start, 0);
//
//    thrust::sort_by_key(keys + N, keys + N * D, vals + N);
//
//    calculateElapsedTime(start, stop, &milliseconds, "Sorting vectors and keys");
//    cudaEventRecord(start, 0);
//
//    // reduction
//    thrust::reduce_by_key(keys, keys + N * D, vals, thrust::make_discard_iterator(), clusters_ptr, binary_pred);
//
//    calculateElapsedTime(start, stop, &milliseconds, "Reduction");
//    cudaEventRecord(start, 0);
//
//    // updating clusters
//    thrust::reduce_by_key(keys, keys + N, const_iter, thrust::make_discard_iterator(), cluster_count_ptr, binary_pred);
//
//    calculateElapsedTime(start, stop, &milliseconds, "Calculating number of vectors in each cluster");
//    cudaEventRecord(start, 0);
//
//    CalculateClusters << <1, K >> > (dev_clusters, dev_cluster_count, *dev_d, *dev_k);
//
//    calculateElapsedTime(start, stop, &milliseconds, "Updating clusters mean");
//    cudaEventRecord(start, 0);
//
//    // back to original vectors order
//    thrust::sort_by_key(vector_order.begin(), vector_order.end(), keys);   
//
//    calculateElapsedTime(start, stop, &milliseconds, "Resorting the belongings to original order");
//
//    //-------------------------------
//    //         END OF LOGIC
//    //-------------------------------
//
//    // error checking and synchronization
//    gpuErrchk(cudaGetLastError());
//    gpuErrchk(cudaDeviceSynchronize());
//
//
//    // Copy memory back to the host
//    gpuErrchk(cudaMemcpy(clusters, dev_clusters, K * D * sizeof(float), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(belonging, dev_belonging, N * sizeof(int), cudaMemcpyDeviceToHost));
//    
//    // Cleaning
//    gpuErrchk(cudaFree(dev_clusters));
//    gpuErrchk(cudaFree(dev_vectors));
//    gpuErrchk(cudaFree(dev_belonging));
//    gpuErrchk(cudaFree(dev_n));
//    gpuErrchk(cudaFree(dev_k));
//    gpuErrchk(cudaFree(dev_d));
//    gpuErrchk(cudaEventDestroy(start));
//    gpuErrchk(cudaEventDestroy(stop));
//
//}