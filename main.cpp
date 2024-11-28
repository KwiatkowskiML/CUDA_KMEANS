#include <stdio.h>
#include <iostream>
#include <iomanip>
#include "HostConstants.h"
#include "vector_add_kernel.cuh"
#include "Vectors.h"
#include "gpu1.cuh"

int main() {
    Vectors vectorsStorage;
    if (!vectorsStorage.readFromTextFile(FILE_PATH)) {
        std::cerr << "Failed to read input file." << std::endl;
        return 1;
    }

    std::cout << "Number of points: " << vectorsStorage.getNumPoints() << std::endl;
    std::cout << "Number of dimensions: " << vectorsStorage.getNumDimensions() << std::endl;
    std::cout << "Number of clusters: " << vectorsStorage.getNumClusters() << std::endl;

    vectorsStorage.PrintVectors();

    float* clusters = vectorsStorage.clusters;
    int* belonging = vectorsStorage.belonging;
    float* vectors = vectorsStorage.vectors;

    int N = vectorsStorage.getNumPoints();
    int K = vectorsStorage.getNumClusters();
    int D = vectorsStorage.getNumDimensions();

    vectorsStorage.PrintClusters();
    vectorsStorage.PrintBelonging();

    cudaError_t cudaStatus = CalculateKmean(clusters, vectors, belonging, N, K, D);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    vectorsStorage.PrintClusters();
    vectorsStorage.PrintBelonging();

    return 0;
}