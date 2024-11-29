#include <stdio.h>
#include <iomanip>
#include "vector_add_kernel.cuh"
#include "Vectors.h"
#include "gpu1.cuh"
#include "HostConstants.h"

int main() {
    Vectors vectorsStorage;
    if (!vectorsStorage.readFromTextFile(FILE_PATH)) {
        fprintf(stderr, "Failed to read input file.\n");
        return 1;
    }

    fprintf(stdout, "Number of points: %d\n", vectorsStorage.getNumPoints());
    fprintf(stdout, "Number of dimensions: %d\n", vectorsStorage.getNumDimensions());
    fprintf(stdout, "Number of clusters: %d\n", vectorsStorage.getNumClusters());

    float* clusters = vectorsStorage.clusters;
    int* belonging = vectorsStorage.belonging;
    float* vectors = vectorsStorage.vectors;

    int N = vectorsStorage.getNumPoints();
    int K = vectorsStorage.getNumClusters();
    int D = vectorsStorage.getNumDimensions();

    vectorsStorage.PrintVectors();
    vectorsStorage.PrintClusters();

    CalculateKmean(clusters, vectors, belonging, N, K, D);

    vectorsStorage.PrintClusters();
    vectorsStorage.PrintBelonging();

    return 0;
}
