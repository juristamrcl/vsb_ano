#pragma once
#include "pch.h"

using namespace cv;

class Ethalon {
public:
	float x;
	float y;
	int objectClass;
	Feature closestObject = Feature();

	Ethalon() {
		this->x = 0;
		this->y = 0;
	}

	Ethalon(float x, float y) {
		this->x = x;
		this->y = y;
		this->objectClass = 0;
	}

	void setObjectClass(int oc) {
		this->objectClass = oc;
	}
};