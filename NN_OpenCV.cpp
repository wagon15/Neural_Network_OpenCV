// NN_OpenCV.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include <time.h>
#include <filesystem>


using namespace std;
using namespace cv;
using namespace cv::ml;
namespace fs = std::experimental::filesystem;

#define NEURAL_NETWORK_FILE_NAME	"trained_model_5_class_2_hands_uniform_distribution_4_nodes.xml"
#define DATABASE_PATH				"D:/data_testing"/*"D:/data_whole"*/
#define DATA_SPLIT_TRAINING_RATIO	0
#define NUMBER_OF_SAMPLES			2500
#define NUMBER_OF_TRAINING_SAMPLES	(int)(DATA_SPLIT_TRAINING_RATIO*NUMBER_OF_SAMPLES)
#define NUMBER_OF_TESTING_SAMPLES	(int)(NUMBER_OF_SAMPLES-NUMBER_OF_TRAINING_SAMPLES)
#define RANDOM_DISTRIBUTION			1
#define UNIFORM_DISTRIBUTION		2

#define NUMBER_OF_CLASSES			5
#define ATTRIBUTES_PER_SAMPLE		1024
#define NUMBER_OF_HIDDEN_LAYER_NODES 4

struct inputOutput
{
	Mat image;
	Mat imgResponse;
};

/*
	Imports data from files, rehapes its structure, normalizes values and assigns a class
	@param files List of paths of files to import
	@return A list of structures containing data and class of the sample
*/
vector<inputOutput> ImportData(vector<string> files)
{
	vector<inputOutput> data;
	inputOutput temp;

	for (size_t i = 0; i < files.size(); i++)
	{
		//loading file
		
		temp.image = imread(files[i], IMREAD_GRAYSCALE);

		if (temp.image.data == NULL)
		{
			std::cout << "Reading error in:" << endl << files[i] << endl;
			break;
		}
		
		//reshape to single row
		temp.image = temp.image.reshape(1, 1);

		//normalize to 0-1 values at 125 threshold
		threshold( temp.image, temp.image, 125, 1, THRESH_BINARY);

		//necessary conversion to float
		temp.image.convertTo(temp.image, CV_32FC1);

		//categorise class

		if (files[i].find("fist") != string::npos)
		{
			temp.imgResponse.push_back(Mat(vector<float>{ 1, 0, 0, 0, 0 },true));
		}
		else if (files[i].find("hi") != string::npos)
		{
			temp.imgResponse.push_back(Mat(vector<float>{ 0, 1, 0, 0, 0 }, true));
		}
		else if (files[i].find("ok") != string::npos)
		{
			temp.imgResponse.push_back(Mat(vector<float>{ 0, 0, 1, 0, 0 }, true));
		}
		else if (files[i].find("rock") != string::npos)
		{
			temp.imgResponse.push_back(Mat(vector<float>{ 0, 0, 0, 1, 0 }, true));
		}
		else if (files[i].find("victory") != string::npos)
		{
			temp.imgResponse.push_back(Mat(vector<float>{ 0, 0, 0, 0, 1 }, true));
		}
		else
		{
			cout << "non response found" << endl;
			getchar();
		}
		//reshape to single row
		temp.imgResponse = temp.imgResponse.reshape(1, 1);
		//necessary conversion to float
		temp.imgResponse.convertTo(temp.imgResponse, CV_32FC1);
		//adds data to list
		data.push_back(temp);

		//// checking content
		//cout << "Dane ze zdjecia:" << endl;
		//for (size_t i = 0; i < temp.image.rows; i++)
		//{
		//	float* Mr = temp.image.ptr<float>(i);
		//	for (size_t j = 0; j < temp.image.cols; j++)
		//	{
		//		cout << Mr[j];
		//	}
		//	cout << endl;
		//}
		//cout << "Dane z odpowiedzi" << endl;
		//for (size_t i = 0; i < temp.imgResponse.rows; i++)
		//{
		//	float* Mr = temp.imgResponse.ptr<float>(i);
		//	for (size_t j = 0; j < temp.imgResponse.cols; j++)
		//	{
		//		cout << Mr[j];
		//	}
		//	cout << endl;
		//}
		////
		temp.imgResponse.release();
	}
	//shuffle the data layout
	std::random_shuffle(data.begin(), data.end());

	return data;
}

/*
	Get paths to all files in directory
	@param directory Path where files are contained
	@return A list of paths to every file in directory
*/
vector<string> getFilesInDirectory(string path)
{
	vector<string> files;
	for (auto & p : fs::directory_iterator(path))
	{
		files.push_back(p.path().string());
	}
	return files;
}

/*
	Splits dataset into training and testing sets
	@param files List of files in dataset
	@param trainingFiles Pointer to container of list of training files
	@param testingFiles Pointer to container of list of testing files
	@param distributionType Two types of training data choosing: random classes density or uniform classes density
*/
void splitFiles(vector<string> files, vector<string>* trainingFiles, vector<string>* testingFiles, int distributionType=UNIFORM_DISTRIBUTION)
{
	switch (distributionType)
	{
	case RANDOM_DISTRIBUTION:
	{
		int* trainingNumbers = new int[NUMBER_OF_TRAINING_SAMPLES];
		int randPosition;
		bool repetition = true;
		
		//random positon of files drawing - training set
		srand(time(NULL));
		for (size_t i = 0; i < NUMBER_OF_TRAINING_SAMPLES; i++)
		{
			while (repetition)
			{
				//drawing a number between 0-4999
				randPosition = rand() % NUMBER_OF_SAMPLES;

				//checking the lack of repetition 
				for (size_t j = 0; j < i; j++)
				{
					if (trainingNumbers[j]==randPosition)
					{
						repetition = true;
						break;
					}
					repetition = false;
				}

				if (i==0)
				{
					repetition = false;
				}
			}
			trainingNumbers[i] = randPosition;
			repetition = true;
		}
		//filling training files list
		for (size_t i = 0; i < NUMBER_OF_TRAINING_SAMPLES; i++)
		{
			trainingFiles->push_back(files[trainingNumbers[i]]);
			files[trainingNumbers[i]] = "";
		}
		//filling testing files list
		for (size_t i = 0; i < NUMBER_OF_SAMPLES; i++)
		{
			if (files[i] != "")
			{
				testingFiles->push_back(files[i]);
			}
			
		}

		delete[] trainingNumbers;
		//Distribution of samples
		int amounOfSamples[5] = { 0,0,0,0,0 };
		for (size_t i = 0; i < trainingFiles->size(); i++)
		{
			if ((*trainingFiles)[i].find("fist") != string::npos)
			{
				amounOfSamples[0]++;
			}
			else if ((*trainingFiles)[i].find("hi") != string::npos)
			{
				amounOfSamples[1]++;
			}
			else if ((*trainingFiles)[i].find("ok") != string::npos)
			{
				amounOfSamples[2]++;
			}
			else if ((*trainingFiles)[i].find("rock") != string::npos)
			{
				amounOfSamples[3]++;
			}
			else if ((*trainingFiles)[i].find("victory") != string::npos)
			{
				amounOfSamples[4]++;
			}
			else
			{
				cout << "non response found" << endl;
				getchar();
			}
		}

		cout << "Distribution of training samples:" << endl
			<< "1. fist =\t" << amounOfSamples[0] << "/" << trainingFiles->size() << " " << amounOfSamples[0] * 100.0 / trainingFiles->size() << "%" << endl
			<< "2. hi =\t\t" << amounOfSamples[1] << "/" << trainingFiles->size() << " " << amounOfSamples[1] * 100.0 / trainingFiles->size() << "%" << endl
			<< "3. ok =\t\t" << amounOfSamples[2] << "/" << trainingFiles->size() << " " << amounOfSamples[2] * 100.0 / trainingFiles->size() << "%" << endl
			<< "4. rock =\t" << amounOfSamples[3] << "/" << trainingFiles->size() << " " << amounOfSamples[3] * 100.0 / trainingFiles->size() << "%" << endl
			<< "5. victory =\t" << amounOfSamples[4] << "/" << trainingFiles->size() << " " << amounOfSamples[4] * 100.0 / trainingFiles->size() << "%" << endl;
		break;
	}
	case UNIFORM_DISTRIBUTION:
	{
		vector<string> class1, class2, class3, class4, class5;
		int* trainingNumbers = new int[NUMBER_OF_TRAINING_SAMPLES / NUMBER_OF_CLASSES];
		int randPosition;
		bool repetition = true;

		//split data set into classes
		for (size_t i = 0; i < NUMBER_OF_SAMPLES; i++)
		{
			if (files[i].find("fist") != string::npos)
				class1.push_back(files[i]);
			else if (files[i].find("hi") != string::npos)
				class2.push_back(files[i]);
			else if (files[i].find("ok") != string::npos)
				class3.push_back(files[i]);
			else if (files[i].find("rock") != string::npos)
				class4.push_back(files[i]);
			else if (files[i].find("victory") != string::npos)
				class5.push_back(files[i]);
			else
			{
				cout << "non response found" << endl
					<< files[i] << endl;
				getchar();
			}
		}
		//drawing random positon of files  - training set
		srand(time(NULL));
		for (size_t i = 0; i < NUMBER_OF_TRAINING_SAMPLES/NUMBER_OF_CLASSES; i++)
		{
			while (repetition)
			{
				//drawing a number between 0-999
				randPosition = rand() % (NUMBER_OF_SAMPLES/NUMBER_OF_CLASSES);

				//checking the lack of repetition 
				for (size_t j = 0; j < i; j++)
				{
					if (trainingNumbers[j] == randPosition)
					{
						repetition = true;
						break;
					}
					repetition = false;
				}

				if (i == 0)
				{
					repetition = false;
				}
			}
			trainingNumbers[i] = randPosition;
			repetition = true;
		}
		//filling training files list
		for (size_t i = 0; i < NUMBER_OF_TRAINING_SAMPLES / NUMBER_OF_CLASSES; i++)
		{
			
			trainingFiles->push_back(class1[trainingNumbers[i]]);
			class1[trainingNumbers[i]] = "";

			trainingFiles->push_back(class2[trainingNumbers[i]]);
			class2[trainingNumbers[i]] = "";

			trainingFiles->push_back(class3[trainingNumbers[i]]);
			class3[trainingNumbers[i]] = "";

			trainingFiles->push_back(class4[trainingNumbers[i]]);
			class4[trainingNumbers[i]] = "";

			trainingFiles->push_back(class5[trainingNumbers[i]]);
			class5[trainingNumbers[i]] = "";
		}
		//filling testing files list
		for (size_t i = 0; i < NUMBER_OF_SAMPLES / NUMBER_OF_CLASSES; i++)
		{
			if (class1[i] != "")
			{
				testingFiles->push_back(class1[i]);
			}
			if (class2[i] != "")
			{
				testingFiles->push_back(class2[i]);
			}
			if (class3[i] != "")
			{
				testingFiles->push_back(class3[i]);
			}
			if (class4[i] != "")
			{
				testingFiles->push_back(class4[i]);
			}
			if (class5[i] != "")
			{
				testingFiles->push_back(class5[i]);
			}
		}

		delete[] trainingNumbers;
		break;
	}
	}
}

/*
	Transform dataset into Mat types with samples and answers in rows
	@param dataset A list of structures containing data and class of the sample
	@param samples Mat type for containing sample in each row
	@param answers Mat type for containing class in each row
*/
void datasetToMatType(vector<inputOutput> dataset, Mat* samples, Mat* answers)
{
	for (size_t i = 0; i < dataset.size(); i++)
	{
		samples->push_back(dataset[i].image);
		answers->push_back(dataset[i].imgResponse);
	}
}

int main()
{
	std::cout << "Reading data set.." << endl;

	vector<string> trainingFiles, testingFiles, files = getFilesInDirectory(DATABASE_PATH);

	splitFiles(files, &trainingFiles, &testingFiles, UNIFORM_DISTRIBUTION);

	std::cout << "Nr of found files in directory		" << files.size() << endl
		<< "Nr of files in training set		" << trainingFiles.size() << endl
		<< "Nr of files in testing set		" << testingFiles.size() << endl
		<< "press button to start training phase" << endl;
	getchar();

	//training data
	vector<inputOutput> trainingData = ImportData(trainingFiles);
	
	Mat trainingSamples, trainingAnswers;

	datasetToMatType(trainingData, &trainingSamples, &trainingAnswers);
	std::cout << "Training data loaded..." << endl;
	//test data
	vector<inputOutput> testingData = ImportData(testingFiles);

	Mat testingSamples, testingAnswers;

	datasetToMatType(testingData, &testingSamples, &testingAnswers);
	std::cout << "Testing data loaded..." << endl;
	if (true)
	{

		std::cout << "Size of a training dataset: " << trainingData.size() << std::endl;
		std::cout << "Size of a test dataset: " << testingData.size() << std::endl;
		std::cout << "Do you want to train NNetwork? y/n\t";
		char train = getchar();
		Ptr<ANN_MLP> nnetwork;

		if (train == 'y')
		{
			// Configurate topology of the network to be 3 layer 1024->10->3

			int layers_dim[] = { ATTRIBUTES_PER_SAMPLE, NUMBER_OF_HIDDEN_LAYER_NODES ,  NUMBER_OF_CLASSES }; //vector specifying the number of neurons in each layer
			Mat layerSizes = Mat(1, 3, CV_32SC1, layers_dim);
			std::cout << "MLP topology: " << layerSizes << std::endl;

			// Create the neural network and define its parameters
			nnetwork = ANN_MLP::create();
			nnetwork->setLayerSizes(layerSizes);
			nnetwork->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.6, 1); // sigmoid function with parameters alpha=0.6 and beta=1 
			nnetwork->setTrainMethod(ANN_MLP::BACKPROP);
			nnetwork->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.000001));
			nnetwork->setBackpropMomentumScale(0.1);  // co-efficents for backpropogation training
			nnetwork->setBackpropWeightScale(0.1);    // co-efficents for backpropogation training

													  // training of the neural network 

			printf("\nUsing training dataset: %s\n", DATABASE_PATH);

			cout << "Training in progress..." << endl;

			clock_t timeHolder = clock();

			bool iterations = nnetwork->train(trainingSamples, ROW_SAMPLE, trainingAnswers);

			timeHolder = clock() - timeHolder;
			cout << "Training time: " << timeHolder / CLOCKS_PER_SEC << "s" << endl;


			cout << "Training done correct? :" << (iterations ? "true" : "false") << endl
				<< "press buton to save neural network" << endl;

			getchar();

			// saving the trained model
			nnetwork->cv::ml::ANN_MLP::save(NEURAL_NETWORK_FILE_NAME);

		}
		else if (train == 'n')
		{
			const cv::String filepath = NEURAL_NETWORK_FILE_NAME;
			nnetwork = Algorithm::load<ANN_MLP>(filepath);
		}
		

		// classifier testing and results 

		Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);  // classification output vector
		Point max_loc = Point(0, 0);

		Mat test_sample;
		int correct_class = 0;
		int wrong_class = 0;
		int false_positives[NUMBER_OF_CLASSES] = { 0,0,0 };
		float answer = 0;

		printf("\nUsing test dataset: %s\n\n", DATABASE_PATH);

		std::string class_label;

		for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
		{

			test_sample = testingSamples.row(tsample); // takes one row from test matrix

			float answer = nnetwork->predict(test_sample); // neural network prediction

			if (!(testingAnswers.at<float>(tsample, answer))) // if the answer class is not the same as in the testing classifications...
			{
				wrong_class++;

				false_positives[(int)answer]++;
			}
			else
			{
				correct_class++;
			}
		}

		std::cout << "\nTest results:\n"
			<< "- Correct classifications: \t" << correct_class << " (" << (float)correct_class * 100 / NUMBER_OF_TESTING_SAMPLES << "%)\n"
			<< "- Wrong classifications: \t" << wrong_class << " (" << (float)wrong_class * 100 / NUMBER_OF_TESTING_SAMPLES << "%)\n";

		for (int i = 0; i < NUMBER_OF_CLASSES; i++)
		{
			if (i == 0)
				class_label = "hi";
			if (i == 1)
				class_label = "ok";
			if (i == 2)
				class_label = "victory";
			if (i == 3)
				class_label = "rock";
			if (i == 4)
				class_label = "fist";

			std::cout << "  - Class '" << class_label << "' false postives: \t" << false_positives[i]
				<< " (" << (float)false_positives[i] * 100 / NUMBER_OF_TESTING_SAMPLES << "%)\n";

		}
		std::cout << std::endl;

		std::system("Pause");
		return 0;
	}

	std::system("Pause");

	getchar();
	return 0;
}