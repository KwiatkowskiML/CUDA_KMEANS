#include "Vectors.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#pragma warning(disable:4996) // disabling issues with fopen

bool Vectors::readFromTextFile(const std::string& filename) {
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename.c_str());
        return false;
    }

    // Read the header
    if (fscanf(file, "%d %d %d", &N, &D, &K) != 3) {
        fprintf(stderr, "Failed to read header from file: %s\n", filename.c_str());
        fclose(file);
        return false;
    }
    
    // Allocate the array
    try {
        vectors = new float[N * D];
        clusters = new float[K * D];
        belonging = new int[N];
    }
    catch (const std::bad_alloc& e) {
        fprintf(stderr, "Memory allocation failed: %s\n", e.what());
        goto Error;
    }

    // Read the data points
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            if (fscanf(file, "%f", &vectors[i + j * N]) != 1) {
                fprintf(stderr, "Failed to read vector %d from file: %s\n", i, filename.c_str());
                goto Error;
            }
        }
    }
    // Filling clusters array
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < D; j++)
        {
            clusters[i + j * K] = vectors[i + j * N];
        }
    }
    memset(belonging, 0, N * sizeof(int));

    fclose(file);
    return true;
Error:
    fclose(file);
    delete[] vectors;
    delete[] clusters;
    delete[] belonging;
    vectors = nullptr;
    clusters = nullptr;
    belonging = nullptr;
    return false;
}

bool Vectors::readFromBytesfile(const std::string& filename)
{
    FILE* file = fopen(filename.c_str(), "rb");  // Open in binary read mode
    if (!file) {
        fprintf(stderr, "Failed to open binary file: %s\n", filename.c_str());
        return false;
    }

    if (fread(&N, sizeof(int), 1, file) != 1 ||
        fread(&D, sizeof(int), 1, file) != 1 ||
        fread(&K, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Failed to read header from binary file: %s\n", filename.c_str());
        fclose(file);
        return false;
    }

    try {
        vectors = new float[N * D];
        clusters = new float[K * D];
        belonging = new int[N];
    }
    catch (const std::bad_alloc& e) {
        fprintf(stderr, "Memory allocation failed: %s\n", e.what());
        fclose(file);
        return false;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            if (fread(&vectors[i + j * N], sizeof(float), 1, file) != 1) {
                fprintf(stderr, "Failed to read all vectors from binary file: %s, %d, %d\n", filename.c_str(), i, j);
                goto Error;
            }
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < D; j++) {
            clusters[i + j * K] = vectors[i + j * N];
        }
    }

    memset(belonging, 0, N * sizeof(int));

    fclose(file);
    return true;

Error:
    fclose(file);
    delete[] vectors;
    delete[] clusters;
    delete[] belonging;
    vectors = nullptr;
    clusters = nullptr;
    belonging = nullptr;
    return false;
}

bool Vectors::readFromFile(const std::string& filename)
{
    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) {
        fprintf(stderr, "File has no extension: %s\n", filename.c_str());
        return false;
    }

    std::string extension = filename.substr(dotPos + 1);
    if (extension == "txt") {
        return readFromTextFile(filename);
    }
    else if (extension == "dat") {
        return readFromBytesfile(filename);
    }
    else {
        fprintf(stderr, "Unsupported file type: %s\n", filename.c_str());
        return false;
    }
}

int Vectors::getNumPoints() const {
    return N;
}

int Vectors::getNumDimensions() const {
    return D;
}

int Vectors::getNumClusters() const {
    return K;
}

void Vectors::PrintClusters()
{
    if (clusters == nullptr)
        return;
    fprintf(stdout, "\n\n");
    for (int i = 0; i < K; i++)
    {
        fprintf(stdout, "Cluster %d: ", i);
        for (int j = 0; j < D; j++)
        {
            fprintf(stdout, "%f ", clusters[i + j * K]);
        }
        fprintf(stdout, "\n");
    }
}

void Vectors::PrintVectors()
{
    if (vectors == nullptr)
        return;
    fprintf(stdout, "\n\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            fprintf(stdout, "%f ", vectors[i + j * N]);
        }
        fprintf(stdout, "\n");
    }
}

void Vectors::PrintBelonging()
{
    if (belonging == nullptr)
        return;
    fprintf(stdout, "\n\n");
    for (int i = 0; i < N; i++)
    {
        fprintf(stdout, "%d ", belonging[i]);
    }
}

void Vectors::WriteResults(const std::string& filename)
{
    if (clusters == nullptr || vectors == nullptr || belonging == nullptr)
    {
        fprintf(stderr, "Arrays not correctly initialized");
        return;
    }

    FILE* output_file = fopen(filename.c_str(), "w");
    if (output_file == nullptr) {
        fprintf(stderr, "Failed to open output file: %s\n", filename.c_str());
        return;
    }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < D; j++)
        {
            fprintf(output_file, "%f ", clusters[i + j * K]);
        }
        fprintf(output_file, "\n");
    }

    for (int i = 0; i < N; i++)
    {
        fprintf(output_file, "%d\n", belonging[i]);
    }

    fclose(output_file);
}

Vectors::~Vectors() {
    if (vectors != nullptr) {
        delete[] vectors;
        vectors = nullptr; 
    }

    if (clusters != nullptr) {
        delete[] clusters;
        clusters = nullptr; 
    }

    if (belonging != nullptr) {
        delete[] belonging;
        belonging = nullptr; 
    }
}