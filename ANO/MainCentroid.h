#pragma once
#include "pch.h"
#include "ComputedObject.h"

using namespace cv;

class MainCentroid {
public:
	float x;
	float y;
	list<FeatureObject> nearestCentroids;
	int objectClass;

	MainCentroid() {}
	MainCentroid(float x, float y) {
		this->x = x;
		this->y = y;
	}

	MainCentroid(float x, float y, int objectClass) {
		this->x = x;
		this->y = y;
		this->objectClass = objectClass;
	}
};