#pragma once
#include "pch.h"

class Ethalon {
public:
	float x;
	float y;
	int objectClass;
	FeatureObject closestObject = FeatureObject();

	Ethalon() {
		this->x = 0.0f;
		this->y = 0.0f;
		this->objectClass = -1.0f;
	}

	Ethalon(float x, float y) {
		this->x = x;
		this->y = y;
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