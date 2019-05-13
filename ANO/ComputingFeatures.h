#pragma once
#include "pch.h"
#include "ComputedObject.h"
#include "MainCentroid.h"

using namespace cv;

Mat thresholdImage(Mat image, int threshold);
void floodFill(Mat image, ComputedObject &co);
void computeMoment(Mat input, int p, int q);
void computeCenteredMoment(Mat input, FeatureObject &feature);
void computeMoments(ComputedObject &co);
void setPerimeter(ComputedObject &co);
void setFeatures(ComputedObject &co);
list<Centroid> computeEthalons(ComputedObject &co, int numOfObjects);
double getEuclideanDistance(FeatureObject f, Ethalon b);
list<MainCentroid> computeKMeans(ComputedObject co, int numOfCentroids);
void writeCentroidsToObjects(ComputedObject &co, list<MainCentroid> centroids);
void doHog(Mat image, int blockSize, int cellSize);
void clasifyObjects(ComputedObject& co, list<Centroid> ethalons);
int getMinimumType(list<Centroid> ethalons, Centroid c);