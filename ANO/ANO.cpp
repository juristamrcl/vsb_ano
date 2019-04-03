// ANO.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "ObjectFeature.h"
#include "ComputingFeatures.h"
#include "ComputedObject.h"

#define M_PI 3.14159265358979323846

using namespace cv;
using namespace std;


double ComputeEuclideanDistance(MyPoint a, Ethalon b)
{
	double distance = sqrt( pow(b.x-a.x,2) + pow(b.y - a.y,2) );
	return distance;
}

double ComputeEuclideanDistance(MyPoint a, MyPoint b)
{
	double distance = sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
	return distance;
}

void ComputeEthalons(ObjectFeature &feature)
{
	//std::cout << "Computing  ethalons" << std::endl;

	double x = 0.0;
	double y = 0.0;
	std::list<Ethalon> ethalons;
	Ethalon currentEthalon = Ethalon(0.0,0.0);
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		//feature1 => x
		//feature2 => y

		if (currentEthalon.x == 0.0)
		{
			currentEthalon = Ethalon((*obj).Feature1, (*obj).Feature2);
			ethalons.push_back(currentEthalon);
		}
		else
		{
			MyPoint currentPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
			std::list<Ethalon>::iterator eth = ethalons.begin();
			bool found = false;
			while (eth != ethalons.end())
			{
				if (ComputeEuclideanDistance(currentPoint, (*eth)) < 0.2)
				{
					(*eth) = Ethalon((currentPoint.x + (*eth).x) / 2, (currentPoint.y + (*eth).y) / 2);
					found = true;
				}

				eth++;
			}
			if (!found)
			{
				ethalons.push_back(Ethalon(currentPoint.x,currentPoint.y));
			}
		}
		obj++;
	}
	std::list<Ethalon>::iterator eth = ethalons.begin();
	while (eth != ethalons.end())
	{
		(*eth).AddClass();
		eth++;
	}

	feature.Ethalons = ethalons;
	//std::cout << "Done ..." << std::endl;

}

//void ComputeKMeans(ObjectFeature &feature)
//{
//	//std::cout << "Computing k-means" << std::endl;
//
//	int numOfCentroids = feature.Ethalons.size();
//	srand(time(NULL));
//	std::list<CentroidObject> centroids;
//
//	for (int i = 0; i < numOfCentroids; i++)
//	{
//		int index = rand() % (feature.Objects.size() - 1) + 1;
//		std::list<FeatureList>::iterator it = feature.Objects.begin();
//		std::advance(it, index);
//		CentroidObject c;
//		c.Centroid = MyPoint((*it).Feature1, (*it).Feature2);
//		centroids.push_back(c);
//	}
//	double delta = 0.05;
//	bool iterate = true;
//
//	while (iterate)
//	{
//		//clear closestobject list from previous iteration
//		std::list<CentroidObject>::iterator cen = centroids.begin();
//		while (cen != centroids.end())
//		{
//			(*cen).ClosestObjects.clear();
//			cen++;
//		}
//
//		//asign objects to centroids
//		std::list<FeatureList>::iterator obj = feature.Objects.begin();
//		while (obj != feature.Objects.end())
//		{
//			double distance = INFINITY;
//			CentroidObject *closestCentroid = nullptr;
//			MyPoint objectPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
//			std::list<CentroidObject>::iterator cen = centroids.begin();
//			while (cen != centroids.end())
//			{
//				double temp = ComputeEuclideanDistance((*cen).Centroid, objectPoint);
//				if (temp < distance)
//				{
//					distance = temp;
//					closestCentroid = &(*cen);
//				}
//
//				cen++;
//			}
//			closestCentroid->ClosestObjects.push_back((*obj));
//			obj++;
//		}
//
//		int tmp = 0;
//		//compute centroids
//		cen = centroids.begin();
//		while (cen != centroids.end())
//		{
//			if ((*cen).ClosestObjects.size() > 0)
//			{
//				MyPoint oldCentroid = (*cen).Centroid;
//				double sumX = 0.0;
//				double sumY = 0.0;
//				int count = 0;
//				std::list<FeatureList>::iterator obj = (*cen).ClosestObjects.begin();
//				while (obj != (*cen).ClosestObjects.end())
//				{
//					sumX += (*obj).Feature1;
//					sumY += (*obj).Feature2;
//					count++;
//					obj++;
//				}
//
//				MyPoint newCentroid = MyPoint(sumX / count, sumY / count);
//				double dist = ComputeEuclideanDistance(oldCentroid, newCentroid);
//				if (dist <= delta)
//				{
//					if (tmp == 0)
//						iterate = false;
//				}
//				else
//				{
//					iterate = true;
//				}
//
//				(*cen).Centroid = newCentroid;
//
//				tmp++;
//			}
//			
//			cen++;
//		}
//	}
//	
//	feature.Centroids = centroids;
//	//std::cout << "Done ..." << std::endl;
//
//}

void train(NN* nn)
{
	int n =1000;
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

int main(int argc, char** argv)
{
	Mat thresholded;

	Mat trainGrayscale = imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	thresholded = thresholdImage(trainGrayscale, 40.0f);

	ComputedObject co = ComputedObject(thresholded);

	floodFill(thresholded, co);
	computeMoments(co);
	setPerimeter(co);
	setFeatures(co);
	Ethalon* output = computeEthalons(co);

	list<MainCentroid> centroids = computeKMeans(co);

	writeCentroidsToObjects(co, centroids);

	/*NN * nn = createNN(2, 4, 2);
	train(nn);

	getchar();

	test(nn,100);

	getchar();

	releaseNN(nn);*/

	co.showStoredImages();
	return 0;

}

