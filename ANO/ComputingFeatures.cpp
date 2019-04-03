#include "pch.h"
#include "ComputedObject.h"
#include "ObjectEthalon.h"
#include "MainCentroid.h"

using namespace std;

int featureIndex = 0;

Mat thresholdImage(Mat image, int threshold) {
	Mat output = Mat::zeros(image.size(), CV_32FC1);
	Mat fc1_image = Mat::zeros(image.size(), CV_32FC1);

	image.convertTo(fc1_image, CV_32FC1);

	for (int y = 0; y < fc1_image.rows; y++) {
		for (int x = 0; x < fc1_image.cols; x++) {
			if (fc1_image.at<float>(y, x) > threshold)
			{
				output.at<float>(y, x) = fc1_image.at<float>(y, x);
			}
		}
	}

	return output;
}

void floodRecursive(Mat &image, Mat &indexedImage, Mat &coloredImage, int y, int x, int index, Vec3b color)
{
	if (x > image.cols || x < 0)
		return;
	if (y > image.rows || y < 0)
		return;


	if (image.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
	{
		indexedImage.at<float>(y, x) = index;
		coloredImage.at<Vec3b>(y, x) = color;

		floodRecursive(image, indexedImage, coloredImage, y, x + 1, index, color);
		floodRecursive(image, indexedImage, coloredImage, y + 1, x, index, color);
		floodRecursive(image, indexedImage, coloredImage, y, x - 1, index, color);
		floodRecursive(image, indexedImage, coloredImage, y - 1, x, index, color);
	}
}

void floodFill(Mat image, ComputedObject &co)
{
	image.convertTo(image, CV_32FC1);

	Mat indexedImage = Mat::zeros(image.size(), CV_32FC1);
	Mat coloredImage = Mat::zeros(image.size(), CV_8UC3);

	int index = 0;
	Vec3b color = Vec3b(rand() % 255, rand() % 255, rand() % 255);

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (image.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
			{
				floodRecursive(image, indexedImage, coloredImage, y, x, ++index, color);
				color = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
			}
		}
	}

	co.setIndexed(indexedImage);
	co.setColored(coloredImage);
	co.setIndexCount(index);

	return;
}

int computeMoment(Mat input, int p, int q, int index) {
	double moment = 0.0;

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			if (input.at<float>(y, x) == index) {
				moment += pow(x, p) * pow(y, q);
			}
		}
	}
	return moment;
}

void computeCenteredMoment(Mat input, Feature &feature) {
	double moment00 = 0.0;
	double moment02 = 0.0;
	double moment20 = 0.0;
	double moment11 = 0.0;

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			if (input.at<float>(y, x) == feature.getIndex()) {
				moment00 += pow((x - feature.getXt()), 0) * pow((y - feature.getYt()), 0);
				moment02 += pow((x - feature.getXt()), 0) * pow((y - feature.getYt()), 2);
				moment20 += pow((x - feature.getXt()), 2) * pow((y - feature.getYt()), 0);
				moment11 += pow((x - feature.getXt()), 1) * pow((y - feature.getYt()), 1);
			}
		}
	}

	feature.setCm00(moment00);
	feature.setCm11(moment11);
	feature.setCm02(moment02);
	feature.setCm20(moment20);

	return;
}

void computeMoments(ComputedObject &co) {
	list<Feature> features;

	for (int i = 1; i < co.getIndexCount() + 1; i++) {
		float moment00 = computeMoment(co.getIndexed(), 0, 0, i);
		float moment01 = computeMoment(co.getIndexed(), 0, 1, i);
		float moment10 = computeMoment(co.getIndexed(), 1, 0, i);

		Feature m = Feature(featureIndex++, moment00, moment01, moment10, i);

		computeCenteredMoment(co.getIndexed(), m);

		features.push_back(m);
	}

	co.setFeatures(features);
}

void setPerimeter(ComputedObject &co) {
	Mat image = co.getIndexed();
	list<Feature> tempFeatures = co.getFeatures();

	for (Feature &feature : tempFeatures) {
		int perimeter = 0;
		for (int y = 0; y < image.rows; y++) {
			for (int x = 0; x < image.cols; x++) {
				if (y > 0 && y < image.rows - 1 && x > 0 && x < image.cols - 1 && feature.getIndex() == image.at<float>(y, x)) {
					if (image.at<float>(y - 1, x) == image.at<float>(y, x) &&
						image.at<float>(y, x - 1) == image.at<float>(y, x) &&
						image.at<float>(y + 1, x) == image.at<float>(y, x) &&
						image.at<float>(y, x + 1) == image.at<float>(y, x)) {
					}
					else {
						perimeter++;
					}
				}
			}
		}
		feature.setPerimeter(perimeter);
	}

	co.setFeatures(tempFeatures);
}

void setFeatures(ComputedObject &co) {
	list<Feature> tempFeatures = co.getFeatures();

	for (Feature &feature : tempFeatures) {
		float f1, f2;
		f1 = pow(feature.Perimeter, 2) / (100 * feature.cM00);

		double microMax = ((1.0 / 2.0) * (feature.cM20 + feature.cM02)) + ((1.0 / 2.0) * sqrt((4 * pow(feature.cM11, 2)) + pow(feature.cM20 - feature.cM02, 2)));
		double microMin = ((1.0 / 2.0) * (feature.cM20 + feature.cM02)) - ((1.0 / 2.0) * sqrt((4 * pow(feature.cM11, 2)) + pow(feature.cM20 - feature.cM02, 2)));

		f2 = microMin / microMax;
		feature.setFeatures(f1, f2);
	}

	co.setFeatures(tempFeatures);
}

double getEuclideanDistance(Feature f, Ethalon b)
{
	return sqrt(pow(b.x - f.f1, 2) + pow(b.y - f.f2, 2));
}

double getEuclideanDistance(Feature f, Feature f2)
{
	return sqrt(pow(f2.f1 - f.f1, 2) + pow(f2.f2 - f.f2, 2));
}

double getEuclideanDistance(Centroid f, MainCentroid f2)
{
	return sqrt(pow(f2.x - f.x, 2) + pow(f2.y - f.y, 2));
}

double getEuclideanDistance(Centroid f, Centroid f2)
{
	return sqrt(pow(f2.x - f.x, 2) + pow(f2.y - f.y, 2));
}
//void kMeans(ComputedObject co, int clustersNumber) {
//	float limit = 0.1;
//	float change = INFINITY;;
//	list<Feature> tempFeatures = co.getFeatures();
//
//	list<Feature> randomFeatures;
//	for (int i = 0; i < clustersNumber; i++) {
//		int rand = 1 + (std::rand() % (co.getIndexCount() - 1 + 1));
//		randomFeatures.push_back(co.getFeature(rand));
//	}
//
//
//	while (change > limit) {
//		for (Feature &feature : tempFeatures) {
//
//			for (int i = 0; i < k; i++) {
//
//			}
//		}
//	}
//
//	for (Feature &feature : tempFeatures) {
//		for (Feature &feature2 : tempFeatures) {
//			if (feature.id != feature2.id && feature.id < feature2.id) {
//				if (getEuclideanDistance(feature, feature2) < 0.05) {
//					sumF1 += feature.f1;
//					sumF2 += feature.f2;
//					total++;
//				}
//			}
//		}
//	}
//}

list<MainCentroid> computeKMeans(ComputedObject co)
{
	int numOfCentroids = 3;
	srand(time(NULL));
	std::list<MainCentroid> centroids;

	for (int i = 0; i < numOfCentroids; i++)
	{
		int index = rand() % (co.getIndexCount() - 1) + 1;
		Feature f = co.getFeature(index);
		MainCentroid c = MainCentroid(f.f1, f.f2);
		centroids.push_back(c);
	}
	double change = 0.01;
	bool iterate = true;

	while (iterate)
	{
		for (auto &cen : centroids) {
			cen.nearestCentroids.clear();
		}

		for (Feature &feature : co.getFeatures()) {
			double distance = INFINITY;
			MainCentroid *closestCentroid = &MainCentroid();
			MainCentroid objectPoint = MainCentroid(feature.f1, feature.f2);
			for (auto &cen : centroids) {
				double temp = getEuclideanDistance(Centroid(cen.x, cen.y), objectPoint);
				if (temp < distance)
				{
					distance = temp;
					closestCentroid = &cen;
				}
			}

			closestCentroid->nearestCentroids.push_back(feature);
		}

		int tmp = 0;
		//compute centroids
		for (auto &cen : centroids) {
			if (cen.nearestCentroids.size() > 0)
			{
				Centroid oldCentroid = Centroid(cen.x, cen.y);
				double sumX = 0.0;
				double sumY = 0.0;
				int count = 0;
				for (auto &obj : cen.nearestCentroids) {
					sumX += (obj).f1;
					sumY += (obj).f2;
					count++;
				}

				Centroid newCentroid = Centroid(sumX / count, sumY / count);
				double dist = getEuclideanDistance(oldCentroid, newCentroid);
				if (dist <= change)
				{
					if (tmp == 0)
						iterate = false;
				}
				else
				{
					iterate = true;
				}

				cen.x = newCentroid.x;
				cen.y = newCentroid.y;

				tmp++;
			}
		}
	}

	return centroids;
	//std::cout << "Done ..." << std::endl;

}

void writeCentroidsToObjects(ComputedObject &co, list<MainCentroid> centroids) {
	list<Feature> tempFeatures = co.getFeatures();

	int index = 0;
	for (auto &cen : centroids) {
		for (auto &shortest : cen.nearestCentroids) {
			for (Feature &computed : tempFeatures) {
				if (computed.getIndex() == shortest.getIndex())
					computed.setType(index);
			}
		}
		++index;
	}

	co.setFeatures(tempFeatures);

}

// todo finish
Ethalon* computeEthalons(ComputedObject &co) {
	list<Feature> tempFeatures = co.getFeatures();
	list<Feature> tempFeatures2 = co.getFeatures();
	list<Ethalon> ethalons;

	Ethalon *objects[3];

	for (int k = 0; k < 3; k++) {
		float sumF1 = 0.0f;
		float sumF2 = 0.0f;
		int total = 0;

		for (int x = 1; x < co.getIndexCount() + 1; x++) {
			Feature f1 = co.getFeature(x);
			for (int y = x; y < co.getIndexCount() + 1; y++) {
				Feature f2 = co.getFeature(y);
				if (getEuclideanDistance(f1, f2) < 0.05) {
					sumF1 += f1.f1;
					sumF2 += f2.f2;
					total++;
				}
			}
		}

		//for (Feature &feature : tempFeatures) {
		//	for (Feature &feature2 : tempFeatures2) {
		//		if (feature.id != feature2.id && feature.id < feature2.id) {
		//			if (getEuclideanDistance(feature, feature2) < 0.05) {
		//				sumF1 += feature.f1;
		//				sumF2 += feature.f2;
		//				total++;
		//			}
		//		}
		//	}
		//}

		Ethalon eth = Ethalon((sumF1 / total), (sumF2 / total));

		cout << eth.x << endl << eth.y << endl;
		objects[k] = &eth;
		//ethalons.push_back(eth);

		sumF1 = 0.0f;
		sumF2 = 0.0f;
		total = 0;
	}

	return *objects;
}