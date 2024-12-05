#include "Vectors.h"
#include "HostConstants.h"

#include "GpuKmeans1.cuh"
#include "GpuKmeans2.cuh"
#include "CpuKmeans.h"
#include <chrono>

void usage()
{
    fprintf(stderr, "Expected arguments format: <data_format> <computation_method> <input_file> <output_file>\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {

    //----------------------------------
    //       Reading argumetns
    //----------------------------------

    if (argc < ARGUMENT_COUNT)
    {
        fprintf(stderr, "Too few arguments!\n");
        usage();
    }

    const char* input_format = argv[1];
    const char* computation_method = argv[2];
    const char* input_file = argv[3];
    const char* output_file = argv[4];

    bool is_txt = strcmp(input_format, "txt") == 0;
	bool is_bin = strcmp(input_format, "bin") == 0;
	bool is_cpu = strcmp(computation_method, "cpu") == 0;
	bool is_gpu1 = strcmp(computation_method, "gpu1") == 0;
	bool is_gpu2 = strcmp(computation_method, "gpu2") == 0;

    if ((!is_txt && !is_bin) || (!is_cpu && !is_gpu1 && !is_gpu2))
    {
        fprintf(stderr, "Wrong arguments format!\n");
        usage();
    }

    //----------------------------------
    //            Read data
    //----------------------------------
    
    Vectors vectorsStorage;
    printf("Reading data...\n");
    std::chrono::steady_clock::time_point start_time, end_time;
    long long time;

    start_time = std::chrono::high_resolution_clock::now();
    if (!vectorsStorage.readFromFile(is_txt, input_file)) {
        fprintf(stderr, "Failed to read input file.\n");
        return 1;
    }
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    printf("Reading data took %lld ms\n", time);

    //----------------------------------
    //   Choose correct algorithm
    //----------------------------------

    KmeansCalculator* kmeansCalculator = nullptr;
    if (is_cpu)
        kmeansCalculator = new CpuKmeans(vectorsStorage);
    else if (is_gpu1)
        kmeansCalculator = new GpuKmeans1(vectorsStorage);
    else if (is_gpu2)
        kmeansCalculator = new GpuKmeans2(vectorsStorage);

    if (kmeansCalculator == nullptr)
        return 1;

    //----------------------------------
    //        Compute kmeans
    //----------------------------------

    kmeansCalculator->CalculateKmeans();

    //--------------------------------
    //       Writing data
    //--------------------------------

    printf("Writing data...\n");
    start_time = std::chrono::high_resolution_clock::now();
    vectorsStorage.WriteResults(output_file);
    end_time = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    printf("Writing data took %lld ms\n", time);

    delete kmeansCalculator;
    return 0;
}
