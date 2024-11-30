#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GpuKmeans1.cuh"

__global__ void CalculateBelongings(const float* clusters, const float* vectors, int* belonging, int* cluster_count, const int& N, const int& D, const int& K)
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
    belonging[idx] = min_cluster;
    atomicAdd(cluster_count + min_cluster, 1);
}

void GpuKmeans1::CalculateKmeans()
{
	printf("test");
}
