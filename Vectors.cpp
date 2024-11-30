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
    fclose(file);
    return true;
Error:
    fclose(file);
    delete[] vectors;
    delete[] clusters;
    delete[] belonging;
    return false;
}

bool Vectors::readFromBytesfile(const std::string& filename)
{
    return false;
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