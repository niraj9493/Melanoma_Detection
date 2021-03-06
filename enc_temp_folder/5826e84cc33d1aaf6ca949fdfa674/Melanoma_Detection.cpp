// Melanoma_Detection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;
class Melanoma_Detection {

public:

	static void Denoise(Mat* img)
	{
		Mat temp;
		fastNlMeansDenoisingColored(*img, temp, 3, 3, 7, 11);
		*img = temp;
	}

	static void EqualizeHistogram(Mat* img, vector<Mat>* color_channels)
	{
		color_channels->clear();
		cvtColor(*img, *img, COLOR_BGR2YCrCb);	//Can't equilize a BRG, have to convert it first
		split(*img, *color_channels);						//Split the color channels
		equalizeHist((*color_channels)[0], (*color_channels)[0]);	//Equilize the Y component, this will not effect color intensity
		merge(*color_channels, *img);
		cvtColor(*img, *img, COLOR_YCrCb2BGR);

	}

	/**Threshold each color channel to create a binary mask**/
	static void OtsuSplit(Mat* img, vector<Mat>* color_channels, Mat* dst)
	{
		color_channels->clear();
		split(*img, *color_channels);
		for (int i = 0; i < color_channels->size(); i++)
		{
			threshold((*color_channels)[i], (*color_channels)[i], 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
		}

		BinaryMaskMajority(color_channels, dst);
	}

	static void BinaryMaskMajority(vector<Mat>* mask_channels, Mat* dst)
	{
		add((*mask_channels)[0], (*mask_channels)[1], *dst, Mat(), CV_32S);
		add(*dst, (*mask_channels)[2], *dst, Mat(), CV_32S);
		*dst = *dst / 3;
		dst->convertTo(*dst, CV_8U);
		threshold(*dst, *dst, 85 , 255, CV_THRESH_BINARY);

	}

		/** Denoise the image and equilize the histogram to
		account for varying lighting conditions**/
	static void Preprocessing(Mat* img, vector<Mat>* color_channels)
	{
		//Denoising the Image
		//Denoise(img);
		//Equilizing the histogram to deal with different lighting
		//EqualizeHistogram(img, color_channels);
	}

	static void ApplyMask(Mat* img, Mat* mask)
	{
		img->copyTo(*mask, *mask);
	}

	/** Split the image into various forms to prepare for feature
	extraction**/
	static void Segmentation(Mat* img, vector<Mat>* color_channels, Mat* dst)
	{
		OtsuSplit(img, color_channels, dst);
	}

	static void FeatureExtraction(Mat* img, Mat* mask, vector<float>* dst, HOGDescriptor* hog) 
	{
		vector<vector<Point>> contours;
		morphologyEx(*mask, *mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(4, 4)));
		medianBlur(*mask, *mask, 7);
		medianBlur(*mask, *mask, 5);
		medianBlur(*mask, *mask, 3);
		findContours(*mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		ApplyMask(img, mask);
		drawContours(*mask, contours, -1, Scalar(0, 256, 0), 2);
		hog->compute(*mask, *dst);
	}

static void ProcessImages(vector<Mat>* imgs, vector<Mat>* training_data, HOGDescriptor* descriptor)
{
	Mat prep_img;
	Mat binary_mask;
	vector<float> hog;
	vector<Mat> color_channels;

	for (int i = 0; i < imgs->size(); i++)
	{

		(*imgs)[i].copyTo(prep_img);

		Melanoma_Detection::Preprocessing(&prep_img, &color_channels);

		Melanoma_Detection::Segmentation(&prep_img, &color_channels, &binary_mask);

		Melanoma_Detection::FeatureExtraction(&(*imgs)[i], &binary_mask, &hog, descriptor);

		training_data->push_back(Mat(hog).clone());
	}
}


static void BatchTrain(string pos_path, string neg_path, Ptr<ml::SVM> svm, HOGDescriptor* hog)
{
	vector<cv::String> img_paths;
	vector<Mat> img_list;
	vector<Mat> training_samples;
	Mat training_data;
	vector<int> labels;
	Mat img;

	size_t positive_count;
	size_t negative_count;

	glob(pos_path, img_paths, false);
	cout << img_paths.size()<<endl;
	for (int i = 0; i < img_paths.size(); i++)
	{
		img = imread(img_paths[i]);
		resize(img, img, Size(600, 480));
		img_list.push_back(img);
		cout << i <<endl;
	}

	ProcessImages(&img_list, &training_samples, hog);

	positive_count = training_samples.size();
	labels.assign(positive_count, +1);

	glob(neg_path, img_paths, false);

	for (int i = 0; i < img_paths.size(); i++)
	{
		img = imread(img_paths[i]);
		if (img.empty()) // Check for failure
		{
			cout << "Could not open or find the image: " << img_paths[i] << endl;
			system("pause"); //wait for any key press
			exit(-1);
		}
		img_list.push_back(img);
	}

	ProcessImages(&img_list, &training_samples, hog);

	negative_count = training_samples.size() - positive_count;
	labels.insert(labels.end(), negative_count, -1);
	CV_Assert(positive_count < labels.size());

	ConvertForSVM(&training_samples, &training_data);

	svm = ml::SVM::create();

	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(ml::SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(training_data, ml::ROW_SAMPLE, labels);

	hog->setSVMDetector(GetSVMDetector(svm));

}

static vector< float > GetSVMDetector(const Ptr< ml::SVM >& svm)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);

	vector< float > hog_detector(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
	return hog_detector;
}

static void ConvertForSVM(vector<Mat>* training_samples, Mat* training_data)
{
	int rows = (int)training_samples->size();
	int cols = (int)std::max((*training_samples)[0].cols, (*training_samples)[0].rows);
	Mat tmp(1, cols, CV_32FC1);

	for (size_t i = 0; i < (*training_samples).size(); i++)
	{
		CV_Assert((*training_samples)[i].cols == 1 || (*training_samples)[i].rows == 1);
		
		if((*training_samples)[i].cols == 1)
		{
			transpose((*training_samples)[i], tmp);
			tmp.copyTo(training_data->row((int)i));
		}
		else if ((*training_samples)[i].rows == 1)
		{
			(*training_samples)[i].copyTo(training_data->row((int)i));
		}
	}
}
};

int main(int argc, char** argv)
{
	string input;
	bool running = true;

	Ptr <ml::SVM> svm;
	HOGDescriptor hog;

	while (running)
	{
		cout << "Please select Train, Test, Classify or Exit" << endl;
		cin >> input;
		transform(input.begin(), input.end(), input.begin(), tolower);

		if (input == "train")
		{
			cout << "Training..." << endl;
			string positive_path = "C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/data/data_positive/*.jpg";
			string negative_path = "C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/data/data_negative/*.jpg";

			HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 4, 0.2, true, 64, 1);

			Melanoma_Detection::BatchTrain(positive_path, negative_path, svm, &hog);

			svm->save("C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/Models/Melanoma_SVM.svm");
			hog.save("C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/Models/Melanoma_HOG.hog");
			cout << "Training Complete" << endl;
		}
		else if (input == "test")
		{
			cout << "Testing..." << endl;

			svm->load("C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/Models/Melanoma_SVM.svm");
			hog.load("C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/Models/Melanoma_HOG.hog");
			hog.setSVMDetector(Melanoma_Detection::GetSVMDetector(svm));

			string positive_path = "C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/data/test/data_positive/*.jpg";
			string negative_path = "C:/Users/Travis/Desktop/School/SJSU/Machine Learning/Melanoma SVM/data/test/data_negative/*.jpg";

			cout << "Testing Complete" << endl;
		}
		else if (input == "classify")
		{
			string path;
			bool valid_path = false;
			vector<Rect> boundary;
			vector<double> weights;
			Mat img;
			Mat prep_img;
			Mat binary_mask;
			Mat img_with_mask;
			vector<Mat> color_channels;

			while (valid_path == false)
			{
				cout << "Please input the path" << endl;
				cin >> path;

				img = imread(path);

				if (img.empty()) // Check for failure
				{
					cout << "Could not open or find the image: " << path << endl;
					system("pause"); //wait for any key press
				}
				else 
				{
					valid_path = true;
				}
			}

			cout << "Classifying..." << endl;

				Melanoma_Detection::Preprocessing(&prep_img, &color_channels);

				Melanoma_Detection::Segmentation(&prep_img, &color_channels, &binary_mask);

				Melanoma_Detection::ApplyMask(&img, &binary_mask);



				hog.detectMultiScale(binary_mask, boundary, weights);

				for (size_t j = 0; j < boundary.size(); j++)
				{
					Scalar color = Scalar(0, weights[j] * weights[j] * 200, 0);
					rectangle(img, boundary[j], color, img.cols / 400 + 1);
				}

				imshow(path, img);
				cin.ignore();
			cout << "Classification Complete" << endl;

		}
		else if (input == "exit")
		{
			cout << "exiting...";
			running = false;
		}


	}

	//String windowName = "Melanoma Detection"; //Name of the window

	//namedWindow(windowName); // Create a window

	//destroyWindow(windowName); //destroy the created window

	return 0;
}

