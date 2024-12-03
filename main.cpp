#include "Vectors.h"
#include "HostConstants.h"

#include "GpuKmeans1.cuh"
#include "GpuKmeans2.cuh"
#include "CpuKmeans.h"
#include <chrono>

int main() {
    Vectors vectorsStorage;
    
    printf("Reading data...\n");
    std::chrono::steady_clock::time_point start_time, end_time;
    long long time;

    start_time = std::chrono::high_resolution_clock::now();
    if (!vectorsStorage.readFromFile(INPUT_FILE_PATH)) {
        fprintf(stderr, "Failed to read input file.\n");
        return 1;
    }
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    printf("Reading data took %lld ms\n", time);

    CpuKmeans kmeans(vectorsStorage);
    printf("Computing kmeans...\n");
    start_time = std::chrono::high_resolution_clock::now();
    kmeans.CalculateKmeans();
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    printf("Computing kmeans took %lld ms\n", time);

    printf("Writing data...\n");
    start_time = std::chrono::high_resolution_clock::now();
    vectorsStorage.WriteResults(OUTPUT_FILE_PATH);
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    printf("Writing data took %lld ms\n", time);

    return 0;
}
