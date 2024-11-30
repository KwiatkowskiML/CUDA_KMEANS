#include "Vectors.h"
#include "HostConstants.h"

#include "GpuKmeans1.h"

int main() {
    Vectors vectorsStorage;
    if (!vectorsStorage.readFromTextFile(FILE_PATH)) {
        fprintf(stderr, "Failed to read input file.\n");
        return 1;
    }

    GpuKmeans1 kmeans(vectorsStorage);
    kmeans.CalculateKmeans();

    return 0;
}
