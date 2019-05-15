#pragma once
#include "pch.h"
#include "computedFeature.h"

using namespace cv;
using namespace std;

class ComputedObject {
private:
	int indexCount;
	float perimeter;
	Mat thresholded;
	Mat indexed;
	Mat colored;
	list<FeatureObject> featureObjects;

public:
	ComputedObject(Mat thresholded) {
		this->thresholded = thresholded;
	};

	void setIndexCount(int indexCount) {
		this->indexCount = indexCount;
	}

	void setPerimeter(int perimeter) {
		this->perimeter = perimeter;
	}

	void setIndexed(Mat indexed) {
		this->indexed = indexed;
	}

	void setColored(Mat colored) {
		this->colored = colored;
	}

	void setFeatures(list<FeatureObject> features) {
		this->featureObjects = features;
	}


	Mat getColored() {
		return this->colored;
	}

	Mat getIndexed() {
		return this->indexed;
	}

	list<FeatureObject> getObjects() {
		return this->featureObjects;
	}
	
	int getIndexCount() {
		return this->indexCount;
	}

	FeatureObject getFeature(int index) {
		for (FeatureObject &feature : this->featureObjects)
			if (feature.getIndex() == index)
				return feature;
	}
	
	FeatureObject &getFeaturePointer(int index) {
		for (FeatureObject &feature : this->featureObjects)
			if (feature.getIndex() == index)
				return feature;
	}

	void showStoredImages() {
		for (FeatureObject &feature : this->featureObjects) {
			std::ostringstream ss;
			ss << feature.objectIndex;
			string s1(ss.str());

			std::ostringstream ss2;
			ss2 << feature.f2;
			string s2(ss2.str());

			putText(this->colored,
				s1,
				Point(feature.getXt() - 5, feature.getYt() - 6), // Coordinates
				cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
				0.5, // Scale. 2.0 = 2x bigger
				cv::Scalar(255, 255, 255));
		}
		
		//imshow("Thresholded", this->thresholded);
		imshow("Colored", this->colored);

		cv::waitKey(0);
	}
};