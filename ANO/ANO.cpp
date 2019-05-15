// ANO.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "ObjectFeature.h"
#include "ComputingFeatures.h"
#include "ComputedObject.h"

#define M_PI 3.14159265358979323846

using namespace cv;
using namespace std;

void train(NN* nn)
{
	int n = 1000;
	double ** trainingSet = new double *[n];
	for (int i = 0; i < n; i++) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

		bool classA = i % 2;

		for (int j = 0; j < nn->n[0]; j++) {
			if (classA) {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}

		trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
		trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;
	}

	double error = 1.0;
	int i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i%n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i%n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void trainDataset(NN* nn, ComputedObject co)
{
	list<FeatureObject> objects = co.getObjects();
	int n = objects.size();
	double ** trainingSet = new double *[n];

	int i = 0;
	for (auto &object : objects) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

		bool classA = i % 2;

		for (int j = 0; j < nn->n[0]; j++) {
			if (j % 2) {
				trainingSet[i][j] = (double)object.f1;
			}
			else {
				trainingSet[i][j] = (double)object.f2;
			}
		}

		if (object.getType() == 0) {
			trainingSet[i][nn->n[0]] = 1.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
		}
		else if (object.getType() == 1) {
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 1.0;
			trainingSet[i][nn->n[0] + 2] = 0.0;
		}
		else if (object.getType() == 2) {
			trainingSet[i][nn->n[0]] = 0.0;
			trainingSet[i][nn->n[0] + 1] = 0.0;
			trainingSet[i][nn->n[0] + 2] = 1.0;
		}

		++i;
	}

	double error = 1.0;
	i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i%n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i%n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, int num_samples = 10)
{
	double* in = new double[nn->n[0]];

	int num_err = 0;
	for (int n = 0; n < num_samples; n++)
	{
		bool classA = rand() % 2;

		for (int j = 0; j < nn->n[0]; j++)
		{
			if (classA)
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}
		printf("predicted: %d\n", !classA);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classA) num_err++;
		printf("\n");
	}

	double err = (double)num_err / num_samples;
	printf("test error: %.2f\n", err);
}

void testDataset(NN* nn, ComputedObject co)
{
	list<FeatureObject> objects = co.getObjects();
	double* in = new double[nn->n[0]];

	int num_err = 0;
	int i = 0;
	for (auto &object : objects) {
		bool classA = i % 2;

		for (int j = 0; j < nn->n[0]; j++)
		{
			if (j % 2) {
				in[j] = object.f1;
			}
			else {
				in[j] = object.f2;
			}
		}
		int classReal = object.getType();

		printf("predicted: %d\n", classReal);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classReal) num_err++;
		printf("\n");
		i++;
	}

	double err = (double)num_err / objects.size();
	printf("test error: %.2f\n", err);
}

int main(int argc, char** argv)
{
	Mat thresholded, testThresholded;

	Mat trainGrayscale = imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat testGrayscale = imread("images/test01.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat hogImage = imread("images/hog_test.png", CV_LOAD_IMAGE_GRAYSCALE);

	thresholded = thresholdImage(trainGrayscale, 40.0f);
	testThresholded = thresholdImage(testGrayscale, 40.0f);

	ComputedObject co = ComputedObject(thresholded);
	ComputedObject coTest = ComputedObject(testThresholded);

	//doHog(hogImage, 2, 8);

	floodFill(thresholded, co);
	computeMoments(co);
	setPerimeter(co);
	setFeatures(co);

	floodFill(testThresholded, coTest);
	computeMoments(coTest);
	setPerimeter(coTest);
	setFeatures(coTest);

	list<MainCentroid> output = computeEthalons(co, 3);

	cout << endl << "Ethalons clasified: " << endl;
	clasifyObjects(coTest, output, "ethalons");

	list<MainCentroid> centroids = computeKMeans(co, 3);

	cout << endl << "K-means clasified: " << endl;
	clasifyObjects(coTest, centroids, "kmeans");

	//writeCentroidsToObjects(co, centroids);

	//NN * nn = createNN(2, 4, 2);
	//train(nn);

	//getchar();

	//test(nn,100);

	//getchar();

	//releaseNN(nn);

	//NN * nn = createNN(2, 4, 3);
	//trainDataset(nn, co);

	//getchar();

	//testDataset(nn,coTest);

	//getchar();

	//releaseNN(nn);

	co.showStoredImages();
	return 0;

}

