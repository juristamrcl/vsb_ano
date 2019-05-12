#pragma once
#include "pch.h"

using namespace cv;

class Ethalon {
public:
	float x;
	float y;
	int objectClass;
	//FeatureObject closestObject = FeatureObject();

	Ethalon() {
		this->x = 0;
		this->y = 0;
		this->objectClass = -1;
	}

	Ethalon(float x, float y) {
		this->x = x;
		this->y = y;
		this->objectClass = 0;
	}

	Ethalon(float x, float y, int oClass) {
		this->x = x;
		this->y = y;
		this->objectClass = oClass;
	}

	void setObjectClass(int oc) {
		this->objectClass = oc;
	}
};