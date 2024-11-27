#include <stdio.h>
#include "vector_add_kernel.cuh"
#include "Vectors.h"
#include <iostream>
#include <iomanip>

int main() {
    Vectors vectors;
    if (vectors.readFromTextFile("C://Users//Kmicha//studia//sem5//gpu//points//points_test.txt")) {
        std::cout << "Number of points: " << vectors.getNumPoints() << std::endl;
        std::cout << "Number of dimensions: " << vectors.getNumDimensions() << std::endl;
        std::cout << "Number of clusters: " << vectors.getNumClusters() << std::endl;

        float* points = vectors.getPoints();
        for (int i = 0; i < vectors.getNumPoints(); i++) {
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Point: ";
            for (int j = 0; j < vectors.getNumDimensions(); j++) {
                std::cout << points[i * vectors.getNumDimensions() + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        std::cerr << "Failed to read input file." << std::endl;
        return 1;
    }

    return 0;
}