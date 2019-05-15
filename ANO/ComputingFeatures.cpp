#include "pch.h"
#include "ComputedObject.h"
#include "ObjectEthalon.h"
#include "MainCentroid.h"
#define _USE_MATH_DEFINES
#include <math.h>

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

void computeCenteredMoment(Mat input, FeatureObject &feature) {
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
	list<FeatureObject> features;

	for (int i = 1; i < co.getIndexCount() + 1; i++) {
		float moment00 = computeMoment(co.getIndexed(), 0, 0, i);
		float moment01 = computeMoment(co.getIndexed(), 0, 1, i);
		float moment10 = computeMoment(co.getIndexed(), 1, 0, i);

		FeatureObject m = FeatureObject(featureIndex++, moment00, moment01, moment10, i);

		computeCenteredMoment(co.getIndexed(), m);

		features.push_back(m);
	}

	co.setFeatures(features);
}

void setPerimeter(ComputedObject &co) {
	Mat image = co.getIndexed();
	list<FeatureObject> tempFeatures = co.getObjects();

	for (FeatureObject &feature : tempFeatures) {
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
	list<FeatureObject> tempFeatures = co.getObjects();

	for (FeatureObject &feature : tempFeatures) {
		float f1, f2;
		f1 = pow(feature.Perimeter, 2) / (100 * feature.cM00);

		double microMax = ((1.0 / 2.0) * (feature.cM20 + feature.cM02)) + ((1.0 / 2.0) * sqrt((4 * pow(feature.cM11, 2)) + pow(feature.cM20 - feature.cM02, 2)));
		double microMin = ((1.0 / 2.0) * (feature.cM20 + feature.cM02)) - ((1.0 / 2.0) * sqrt((4 * pow(feature.cM11, 2)) + pow(feature.cM20 - feature.cM02, 2)));

		f2 = microMin / microMax;
		feature.setFeatures(f1, f2);
	}

	co.setFeatures(tempFeatures);
}

double getEuclideanDistance(FeatureObject f, Ethalon b)
{
	return sqrt(pow(b.x - f.f1, 2) + pow(b.y - f.f2, 2));
}

double getEuclideanDistance(FeatureObject f, FeatureObject f2)
{
	return sqrt(pow(f2.f1 - f.f1, 2) + pow(f2.f2 - f.f2, 2));
}

double getEuclideanDistance(MainCentroid f, MainCentroid f2)
{
	return sqrt(pow(f2.x - f.x, 2) + pow(f2.y - f.y, 2));
}

//float getEuclideanDistance(MainCentroid f, MainCentroid f2)
//{
//	return sqrt(pow(f2.x - f.x, 2) + pow(f2.y - f.y, 2));
//}

list<MainCentroid> computeKMeans(ComputedObject co, int numOfCentroids)
{
	srand(time(NULL));
	list<MainCentroid> centroids;

	for (int i = 0; i < numOfCentroids; i++)
	{
		int index = rand() % (co.getIndexCount() - 1) + 1;
		FeatureObject f = co.getFeature(index);
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

		for (FeatureObject &feature : co.getObjects()) {
			double distance = INFINITY;
			MainCentroid *closestCentroid = &MainCentroid();
			MainCentroid objectPoint = MainCentroid(feature.f1, feature.f2);
			for (auto &cen : centroids) {
				double temp = getEuclideanDistance(MainCentroid(cen.x, cen.y), objectPoint);
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
				MainCentroid oldCentroid = MainCentroid(cen.x, cen.y);
				double sumX = 0.0;
				double sumY = 0.0;
				int count = 0;
				for (auto &obj : cen.nearestCentroids) {
					sumX += (obj).f1;
					sumY += (obj).f2;
					count++;
				}

				MainCentroid newCentroid = MainCentroid(sumX / count, sumY / count);
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

	int objectIndex = -1;
	for (auto &centroid : centroids) {
		centroid.objectClass = ++objectIndex;
	}

	return centroids;
}

void writeCentroidsToObjects(ComputedObject &co, list<MainCentroid> centroids) {
	list<FeatureObject> tempFeatures = co.getObjects();

	int index = 0;
	for (auto &cen : centroids) {
		for (auto &shortest : cen.nearestCentroids) {
			for (FeatureObject &computed : tempFeatures) {
				if (computed.getIndex() == shortest.getIndex())
					computed.setType(index);
			}
		}
		++index;
	}

	co.setFeatures(tempFeatures);

}

// todo finish
list<MainCentroid> computeEthalons(ComputedObject &co, int numOfObjects) {
	list<MainCentroid> centroids;

	int typeIndex = -1;

	cout << "outputting ethalons" << endl;

	for (int x = 1; x < co.getIndexCount(); x += 4) {
		FeatureObject &fo1 = co.getFeaturePointer(x);
		++typeIndex;
		for (int y = x; y < x + 4; y++) {
			FeatureObject &fo2 = co.getFeaturePointer(y);
			fo2.setType(typeIndex);
		}
	}


	for (int x = 0; x < numOfObjects; x++) {
		float sumF1 = 0.0f;
		float sumF2 = 0.0f;
		int total = 0;

		for (int y = 1; y < co.getIndexCount() + 1; y++) {
			FeatureObject fo = co.getFeature(y);
			if (fo.getType() == x) {

				sumF1 += fo.f1;
				sumF2 += fo.f2;
				total++;
			}
		}

		MainCentroid ethalon = MainCentroid((sumF1 / total), (sumF2 / total), x);

		centroids.push_back(ethalon);
	}

	return centroids;
}

void normalize(Mat &input) {
	double min, max;
	minMaxLoc(input, &min, &max);

	for (int x = 0; x < input.rows; x++) {
		if ((max - min) == 0){
			input.at<float>(x, 0) = 0;
		}
		else input.at<float>(x, 0) = (input.at<float>(x, 0) - min) / (max - min);
	}
	return;
}

int getMinimumType(list<MainCentroid> ethalons, MainCentroid c) {
	int minClass = 0;
	float minDist = 10000.0;
	for (auto eth : ethalons) {
		if (getEuclideanDistance(c, eth) < minDist) {
			minDist = getEuclideanDistance(c, eth);
			minClass = eth.objectClass;
		}
	}

	return minClass;
}

void clasifyObjects(ComputedObject& co, list<MainCentroid> ethalons, string windowName) {

	Mat out;
	co.getColored().copyTo(out);

	for (auto &obj : co.getObjects()) {
		MainCentroid c = MainCentroid(obj.f1, obj.f2);
		int temp = getMinimumType(ethalons, c);
		FeatureObject &fo2 = co.getFeaturePointer(obj.getIndex());
		obj.setType(temp);
		fo2.setType(temp);

		cout << "Object: " << obj.getIndex() << ", type: " << temp << endl;

		std::ostringstream ss;
		ss << obj.getType();
		string s1(ss.str());


		std::ostringstream ss2;
		ss2 << obj.getIndex();
		string s2(ss2.str());

		putText(out,
			s1,
			Point(obj.getXt() + 5, obj.getYt() - 18), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			0.5, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 255, 255));

		putText(out,
			s2,
			Point(obj.getXt() - 3, obj.getYt() + 5), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			0.8, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 255, 255));
	}

	imshow("Clasified - " + windowName, out);
	cv::waitKey(0);
}

void doHog(Mat image, int blockSize, int cellSize)
{
	int dividedSize = 9;

	Mat orientation_of_gradiens_image = Mat::zeros(image.size(), CV_32FC1);
	Mat size_of_gradiens_image = Mat::zeros(image.size(), CV_32FC1);

	Mat fc1_image = Mat::zeros(image.size(), CV_32FC1);
	image.convertTo(fc1_image, CV_32FC1);

	for (int y = 1; y < fc1_image.rows - 1; y++) {
		for (int x = 1; x < fc1_image.cols - 1; x++) {
			float fx = fc1_image.at<float>(y, x + 1) - fc1_image.at<float>(y, x);
			float fy = fc1_image.at<float>(y + 1, x) - fc1_image.at<float>(y, x);

			float pos = (fx == 0) ? 0.0f : (atan(fy / fx) * (180 / M_PI) + 90);
			orientation_of_gradiens_image.at<float>(y, x) = pos;
			size_of_gradiens_image.at<float>(y, x) = sqrt(pow(fx, 2) + pow(fy, 2));
		}
	}

	int actualBlockX = 0;
	int actualBlockY = 0;

	int finalBlockCols = image.cols / (blockSize * cellSize);
	int finalBlockRows = image.rows / (blockSize * cellSize);

	Mat block_image = Mat::zeros(finalBlockRows, finalBlockCols, CV_32FC1);

	vector<vector<Mat>> histograms(finalBlockRows);
	
	Mat tmpHistogram = Mat::zeros(dividedSize, 1, CV_32FC1);

	for (int y = 0; y < finalBlockRows; y++) {
		histograms[y] = vector<Mat>(finalBlockCols);
		for (int x = 0; x < finalBlockCols; x++) {
			for (int k = 0; k < cellSize; k++) {
				for (int l = 0; l < cellSize; l++) {
					int realY = y * cellSize + k;
					int realX = x * cellSize + l;
					float pos = orientation_of_gradiens_image.at<float>(realY, realX);
					int posX = int(pos) / 20;

					float sizeAtPos = size_of_gradiens_image.at<float>(realY, realX);

					tmpHistogram.at<float>(posX, 0) += sizeAtPos;
				}
			}
			normalize(tmpHistogram);
			histograms[y][x] = tmpHistogram.clone();
			cout << tmpHistogram << endl;

			tmpHistogram = Mat::zeros(dividedSize, 1, CV_32FC1);
		}
	}

	//vector<Mat> finalHistograms(finalBlockRows);

	//for (int y = 1; y < finalBlockRows - 1; y++) {
	//	Mat hist = Mat::zeros(dividedSize, 1, CV_32FC1);
	//	for (int x = 1; x < finalBlockCols - 1; x++) {
	//		for (int k = 0; k < histograms[y][x].rows; k++) {
	//			hist.at<float>(k, 0) += histograms[y][x].at<float>(k, 0);
	//			hist.at<float>(k, 0) += histograms[y + 1][x].at<float>(k, 0);
	//			hist.at<float>(k, 0) += histograms[y][x + 1].at<float>(k, 0);
	//			hist.at<float>(k, 0) += histograms[y + 1][x + 1].at<float>(k, 0);
	//			float tmp = hist.at<float>(k, 0);
	//		}
	//	}
	//	normalize(hist);
	//	finalHistograms[y] = hist.clone();
	//	cout << hist << endl;
	//}

	imshow("Gradient", orientation_of_gradiens_image);

	cv::waitKey(0);
}