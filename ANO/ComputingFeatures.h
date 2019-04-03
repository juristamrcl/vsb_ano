#pragma once
#include "pch.h"
#include "ComputedObject.h"
#include "MainCentroid.h"

using namespace cv;

Mat thresholdImage(Mat image, int threshold);
void floodFill(Mat image, ComputedObject &co);
void computeMoment(Mat input, int p, int q);
void computeCenteredMoment(Mat input, Feature &feature);
void computeMoments(ComputedObject &co);
void setPerimeter(ComputedObject &co);
void setFeatures(ComputedObject &co);
Ethalon* computeEthalons(ComputedObject &co);
double getEuclideanDistance(Feature f, Ethalon b);
list<MainCentroid> computeKMeans(ComputedObject co);
void writeCentroidsToObjects(ComputedObject &co, list<MainCentroid> centroids);