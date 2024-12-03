#include "CpuKmeans.h"

int CpuKmeans::CalculateBelongingCpu(const float* vectors, const float* clusters, int* belonging, int* cluster_count, const int& N, const int& D, const int& K)
{
    int min_cluster = 0;
    float min_distance = FLT_MAX;
    int cluster_changes = 0;

    for (int i = 0; i < N; i++)
    {
        min_cluster = 0;
        min_distance = FLT_MAX;
        for (int j = 0; j < K; j++)
        {
            float distance = 0.0f;
            for (int dim = 0; dim < D; dim++)
            {
                float diff = vectors[i + dim * N] - clusters[j + dim * K];
                distance += diff * diff;
            }
            if (distance < min_distance) {
                min_distance = distance;
                min_cluster = j;
            }
        }
        if (belonging[i] != min_cluster)
        {
            cluster_changes++;
            belonging[i] = min_cluster;
        }
        cluster_count[min_cluster]++;
    }

    return cluster_changes;
}

void CpuKmeans::CalculateClusters(const float* vectors, float* clusters, const int* belonging, const int* cluster_count, const int& N, const int& D, const int& K)
{
    for (int i = 0; i < N; i++)
    {
        int clusterd_id = belonging[i];
        for (int dim = 0; dim < D; dim++)
        {
            clusters[clusterd_id + dim * K] += vectors[i + dim * N];
        }
    }

    for (int j = 0; j < K; j++)
    {
        int curr_cluster_count = cluster_count[j];
        for (int dim = 0; dim < D; dim++)
        {
            clusters[j + dim * K] /= curr_cluster_count;
        }
    }
}

void CpuKmeans::CalculateKmeans()
{
    int N = vectorsStorage->getNumPoints();
    int D = vectorsStorage->getNumDimensions();
    int K = vectorsStorage->getNumClusters();
    float* vectors = vectorsStorage->vectors;
    float* clusters = vectorsStorage->clusters;
    int* belonging = vectorsStorage->belonging;
    int* cluster_count = new int[K];
    int cluster_changes = 0;

    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        memset(cluster_count, 0, K * sizeof(int));
        cluster_changes = CalculateBelongingCpu(vectors, clusters, belonging, cluster_count, N, D, K);

        printf("Iteration %d, number of vectors that changed cluster: %d\n", i, cluster_changes);
        if (cluster_changes == 0)
            break;

        memset(clusters, 0, K * D * sizeof(float));
        CalculateClusters(vectors, clusters, belonging, cluster_count, N, D, K);
    }

    delete[] cluster_count;
}
