#pragma once
#include <string>

class Vectors {
public:
    float* vectors;
    float* clusters;
    int* belonging;
    
    bool readFromTextFile(const std::string& filename);
    int getNumPoints() const;
    int getNumDimensions() const;
    int getNumClusters() const;

    void PrintClusters();
    void PrintVectors();
    void PrintBelonging();

    ~Vectors();

private:
    int N, D, K;
};