#pragma once
#include "pch.h"

class Feature {
public:
	int id;
	int objectIndex;
	int Perimeter;
	int type;

	float m00;
	float m01;
	float m10;
	float cM00;
	float cM11;
	float cM02;
	float cM20;
	float xt;
	float yt;
	float f1;
	float f2;

	Feature(){}

	Feature(int id, float m00, float m01, float m10, int objectIndex) {
		this->id = id;
		this->objectIndex = objectIndex;
		this->m00 = m00;
		this->m01 = m01;
		this->m10 = m10;

		this->xt = m10 / m00;
		this->yt = m01 / m00;
	}

	int getIndex() {
		return this->objectIndex;
	}

	float getXt() {
		return this->xt;
	}

	float getYt() {
		return this->yt;
	}

	void setCm00(float cM00) {
		this->cM00 = cM00;
	}

	void setCm02(float cM02) {
		this->cM02 = cM02;
	}

	void setCm20(float cM20) {
		this->cM20 = cM20;
	}

	void setCm11(float cM11) {
		this->cM11 = cM11;
	}

	void setFeatures(float f1, float f2) {
		this->f1 = f1;
		this->f2 = f2;
	}

	void setPerimeter(int Perimeter) {
		this->Perimeter = Perimeter;
	}

	void setType(int type) {
		this->type = type;
	}
};