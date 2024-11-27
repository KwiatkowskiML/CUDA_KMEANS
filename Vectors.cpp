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
    if (fscanf(file, "%d %d %d", &numPoints, &numDimensions, &numClusters) != 3) {
        std::cerr << "Failed to read header from file: " << filename << std::endl;
        fclose(file);
        return false;
    }

    // Allocate the array
    try {
        points = new float[numPoints * numDimensions];
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        fclose(file);
        return false;
    }

    // Read the data points
    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < numDimensions; j++) {
            if (fscanf(file, "%f", &points[i * numDimensions + j]) != 1) {
                std::cerr << "Failed to read point " << i << " from file: " << filename << std::endl;
                delete[] points;
                fclose(file);
                return false;
            }
        }
    }

    if (0 != fclose(file))
    {
        std::cerr << "Error closing a file.";
        delete[] points;
        return false;
    }

    return true;
}

int Vectors::getNumPoints() const {
    return numPoints;
}

int Vectors::getNumDimensions() const {
    return numDimensions;
}

int Vectors::getNumClusters() const {
    return numClusters;
}

float* Vectors::getPoints() {
    return points;
}

Vectors::~Vectors() {
    delete[] points;
}
