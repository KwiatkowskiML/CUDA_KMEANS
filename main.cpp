#include "Vectors.h"
#include "HostConstants.h"

#include "GpuKmeans1.cuh"
#include "GpuKmeans2.cuh"

int main() {
    Vectors vectorsStorage;
    if (!vectorsStorage.readFromFile(INPUT_FILE_PATH)) {
        fprintf(stderr, "Failed to read input file.\n");
        return 1;
    }

    GpuKmeans2 kmeans(vectorsStorage);
    kmeans.CalculateKmeans();
    vectorsStorage.WriteResults(OUTPUT_FILE_PATH);

    return 0;
}
