#pragma once
#include <string>

class Vectors {
public:
    bool readFromTextFile(const std::string& filename);
    int getNumPoints() const;
    int getNumDimensions() const;
    int getNumClusters() const;
    float* getPoints();
    ~Vectors();

private:
    int numPoints, numDimensions, numClusters;
    float* points;
};