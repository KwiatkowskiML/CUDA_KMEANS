#include "Vectors.h"
#include <iostream>
#include <fstream>
#pragma warning(disable:4996) // disabling issues with fopen

bool Vectors::readFromTextFile(const std::string& filename) {
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Read the header
    if (fscanf(file, "%d %d %d", &N, &D, &K) != 3) {
        std::cerr << "Failed to read header from file: " << filename << std::endl;
        fclose(file);
        return false;
    }

    // Allocate the array
    try {
        vectors = new float[N * D];
        clusters = new float[K * D];
        belonging = new int[N * D];
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        goto Error;
    }

    // Read the data points
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            if (fscanf(file, "%f", &vectors[i + j * N]) != 1) {
                std::cerr << "Failed to read vector " << i << " from file: " << filename << std::endl;
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

    std::cout << std::endl << std::endl;
    for (int i = 0; i < K; i++)
    {
        std::cout << "Cluster " << i << ": ";
        for (int j = 0; j < D; j++)
        {
            std::cout << clusters[i + j * K] << " ";
        }
        std::cout << std::endl;
    }
}

void Vectors::PrintVectors()
{
    if (vectors == nullptr)
        return;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            std::cout << vectors[i + j * N] << " ";
        }
        std::cout << std::endl;
    }
}

void Vectors::PrintBelonging()
{
    if (belonging == nullptr)
        return;

    std::cout << std::endl << std::endl;
    for (int i = 0; i < N * D; i++)
    {
        std::cout << belonging[i] << " ";
    }
}

Vectors::~Vectors() {
    delete[] vectors;
    delete[] clusters;
    delete[] belonging;
}