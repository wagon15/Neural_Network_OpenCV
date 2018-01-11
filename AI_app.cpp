// AI_app.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include <Windows.h>
#include <list>
#include <time.h>

using namespace cv;
using namespace std;

#define TRACKIN_MODE				1
#define CALCULATE_HSV_OF_ROI		-1
#define NO_TRACKING					0
#define DIR_FOR_SAVED_DATA			"D:\\ML_test" //->double backslashes "\\"
#define PICTURE_AMOUNT_IN_SAVED_SET 10
#define NAME_OF_SAVED_CLASS			"fist"
#define NAME_OF_USED_NEURAL_NETWORK	"trained_model_5_class_2_hands_uniform_distribution_5_nodes.xml"

//Global variables
Mat image;                      //Container for catured frame from camera
bool selectObject = false;      //Trigger for selecting region
int trackObject = NO_TRACKING;            //Variable for tracking mode    0-no tracking   -1-calculating HSV of ROI   1-tracking
Point origin;
Rect selection;                 //Rectangle for user selection

/*
	Calculates coordinates of select region 
	@param event Mouse button event - up or down
	@param x position on horizontal axi
	@param y position on vertical axi
*/
static void onMouse(int event, int x, int y, int, void*)        //User selected ROI for HSV values
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);

		selection &= Rect(0, 0, image.cols, image.rows);

	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			trackObject = -1;
		break;
	}
}

/*
	Function prints program information - help
*/
static void help()          //Print info
{
	cout << "\nContours based object tracking\n"
		"You select a colored object and the algorithm tracks it.\n"
		"This takes the input from the webcam\n";

	cout << "\n\nKeyboard input options: \n"
		"\tESC - quit the program\n"
		"\ts - stop the tracking\n"
		"\tp - pause video\n"
		"\tw - take set of pictures"
		"\nTo start tracking an object, select the rectangular region in it with the mouse\n\n"
		"First select will calculate tracking color"
		"HINT - you cam pause image for selection the region";
}

/*
	Procedure used to store catured frame in directory with particullar name
	@param source Catured frame
	@param destinationDIR Direction for saving files
	@param pictureNr Pointer to picture number in saving set
*/
static void takePicture(Mat source, string destinationDIR, int* pictureNr)       //Take one frame and store it as 64x64 pixels JPG file
{
	Mat tmp;
	resize(source, tmp, Size(32, 32));
	bool done = imwrite(destinationDIR + "\\" + NAME_OF_SAVED_CLASS +"_img_" + to_string(*pictureNr) + ".jpg", tmp);

	if (!done)
	{
		cout << "***Could not save file...***\n";
		cout << "Current parameter's value: " << "img_" + to_string(*pictureNr) + ".jpg" << endl;
	}
}

/*
	Calculates max and min HSV values of selected region of interest
	@par hsv Frame in HSV color space
	@param hMin Pointer to minimal value of hue
	@param hMax Pointer to maximal value of hue
	@param sMin Pointer to minimal value of saturation
	@param sMax Pointer to maximal value of saturation
	@param vMin Pointer to minimal value 
	@param vMax Pointer to maximal value
*/
void calculateHSVOfROI(Mat hsv, int* hMin, int* hMax, int* sMin, int* sMax, int* vMin, int* vMax)
{
	Mat roi(hsv, selection), H, S, V;
	int ch1[] = { 0, 0 }, ch2[] = { 1, 0, }, ch3[] = { 2, 0 };      //auxiliary variables
	double hmin, hmax, smin, smax, vmin, vmax;

	H.create(roi.size(), hsv.depth());
	S.create(roi.size(), hsv.depth());
	V.create(roi.size(), hsv.depth());
	//Spliting HSV channels
	mixChannels(&roi, 1, &H, 1, ch1, 1);
	mixChannels(&roi, 1, &S, 1, ch2, 1);
	mixChannels(&roi, 1, &V, 1, ch3, 1);
	//calculating HSV boundaries for inRange function
	minMaxIdx(H, &hmin, &hmax);
	minMaxIdx(S, &smin, &smax);
	minMaxIdx(V, &vmin, &vmax);

	*hMin = hmin;
	*hMax = hmax;
	*sMin = smin;
	*sMax = smax;
	*vMin = vmin;
	*vMax = vmax;

	cout << endl << "hmin: " << hmin << "\t\t" << "hmax: " << *hMax << endl
				<< "smin: " << smin << "\t\t" << "smax: " << *sMax << endl
				<< "vmin: " << vmin << "\t\t" << "vmax: " << *vMax << endl;

}

int main(int argc, const char** argv)
{
	VideoCapture cap;
	Rect ROI;
	int mintresh = 5, maxtresh = 10;
	bool savePictureSet = false;        //Trigger for caturing set of images
	int nr = 0;                         //Picture counter
	int  hmin, hmax, smin, smax, vmin, vmax;
	int camNum = 0;                     //Id of user camera

	help();

	cap.open(camNum);

	if (!cap.isOpened())
	{
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: " << camNum << endl;
		return -1;
	}

	namedWindow("Object Tracker", CV_WINDOW_AUTOSIZE);
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	namedWindow("Result", WINDOW_NORMAL);
	namedWindow("Canny", WINDOW_NORMAL); 
	namedWindow("HSV", WINDOW_NORMAL);

	setMouseCallback("Object Tracker", onMouse, 0);

	//Used to tune HSV values
	namedWindow("Control", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("LowH", "Control", &hmin, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &hmax, 179);
	cvCreateTrackbar("LowS", "Control", &smin, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &smax, 255);
	cvCreateTrackbar("LowV", "Control", &vmin, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &vmax, 255);
	cvCreateTrackbar("mintresh", "Canny", &mintresh, 1024);
	cvCreateTrackbar("maxtresh", "Canny", &maxtresh, 1024);

	Mat frame, hsv, edges, mask;

	//vector<int> hull;
	bool paused = false, giveAnswer = false;


	//ANN_MLP
	Ptr<ml::ANN_MLP> nnetwork = Algorithm::load<ml::ANN_MLP>(NAME_OF_USED_NEURAL_NETWORK);

	list<int> answers;
	clock_t timeHolder=0;

	for (;;)
	{
		if (!paused)        //Reading next frame
		{
			cap >> frame;
			if (frame.empty())
				break;
		}

		frame.copyTo(image);

		if (!paused)
		{
			//change of colorspace to HSV
			cvtColor(image, hsv, CV_BGR2HSV);
			

			if (trackObject == CALCULATE_HSV_OF_ROI)             //Part applied when region is selected. 
			{
				calculateHSVOfROI(hsv, &hmin, &hmax, &smin, &smax, &vmin, &vmax);

				trackObject = TRACKIN_MODE;        //change to tracking mode

			}

			if (trackObject == TRACKIN_MODE)
			{
				//mask creation
				inRange(hsv, Scalar(hmin, smin, vmin),
					Scalar(hmax, smax, vmax), mask);

				//morphological opening (remove small objects from the foreground)
				erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
				dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

				dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
				erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

				imshow("HSV", mask);
				//counturs container
				vector<vector<Point> > contours;

				//find edges
				Canny(mask, edges, mintresh, maxtresh);         
				imshow("Canny", edges);
				//find contours
				findContours(edges, contours, noArray(), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);     

				//convex hull container
				vector<vector<Point>> hull(contours.size());

				//calculate convexHull of contours
				for (int i = 0; i < contours.size(); i++)
				{
					convexHull(Mat(contours[i]), hull[i], false);
				}
				
				//find biggest one
				int k = 0;
				for (size_t i = 0; i < contours.size(); i++)
				{
					if (contourArea(hull[k]) < contourArea(hull[i]))
					{
						k = i;
					}
				}

				Mat drawing = Mat::zeros(edges.size(), CV_8UC1);
				drawContours(drawing, contours, contours.size()-1, Scalar(255, 0, 0), CV_FILLED);       //draw contour on "drawing"
				imshow("Contours", drawing);

				if (!contours.empty())
				{
					ROI = boundingRect(contours[k]);        //calculate bounding rectangle of biggest contour

					rectangle(image, ROI, Scalar(0, 0, 255), 2, CV_AA);     //draw it on "image"

					Mat MaskRoi(mask, ROI);
					//MaskRoi.convertTo(MaskRoi, CV_32FC1);

					resize(MaskRoi, MaskRoi, Size(32, 32), INTER_AREA);
					threshold(MaskRoi, MaskRoi, 40, 255, THRESH_BINARY);
					imshow("Result", MaskRoi);

					//ANN_MLP
					Mat MaskRoi_bin = MaskRoi.clone();
					threshold(MaskRoi, MaskRoi_bin, 40, 1, THRESH_BINARY);
					MaskRoi_bin = MaskRoi_bin.reshape(1, 1);
					MaskRoi_bin.convertTo(MaskRoi_bin, CV_32FC1);
					//PREDICT RETURN NUMBER OF CLASS STARTING FROM 0!!!!!!
					float answer = nnetwork->predict(MaskRoi_bin);

					if (timeHolder == 0)
					{
						timeHolder = clock();
					}
					
					//average answer
					//fill answers table
					if (answers.size() < 35)
					{
						answers.push_back(answer);
					}
					else if (answers.size() == 35)
					{
						answers.pop_front();
						answers.push_back(answer);
					}
					//every 0.5s
					if (((clock() - timeHolder) / CLOCKS_PER_SEC) > 0.5)
					{
						int temp[5] = { 0,0,0,0,0 };
						answer = 0;
						timeHolder = clock();

						for (list<int>::iterator iter = answers.begin(); iter != answers.end(); iter++)
						{
							switch (*iter)
							{
								case 0:
									temp[0]++;
									break;
								case 1:
									temp[1]++;
									break;
								case 2:
									temp[2]++;
									break;
								case 3:
									temp[3]++;
									break;
								case 4:
									temp[4]++;
									break;
								default:
									cout <<"answer problem" << endl;
									break;
							}
						}

						for (size_t i = 0; i < 5; i++)
						{
							if (temp[(int)answer]<temp[i])
							{
								answer = i + 1;
							}
						}
					}
					switch ((int)answer)
					{
					case 0:
					{
						putText(image, "fist", Point(ROI.x, ROI.y), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 255), 2);
						break;
					}
					case 1:
					{
						putText(image, "hi", Point(ROI.x, ROI.y), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 255), 2);
						break;
					}
					case 2:
					{
						putText(image, "ok", Point(ROI.x, ROI.y), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 255), 2);
						break;
					}
					case 3:
					{
						putText(image, "rock", Point(ROI.x, ROI.y), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 255), 2);
						break;
					}
					case 4:
					{
						putText(image, "victory", Point(ROI.x, ROI.y), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 255), 2);
						break;
					}
					}
					//

					if (savePictureSet)     //take set of pictures
					{
						if (nr < PICTURE_AMOUNT_IN_SAVED_SET)
						{
							takePicture(MaskRoi, DIR_FOR_SAVED_DATA, &nr);
							nr++;
							Sleep(5);
							cout << "\nPicture nr = " << nr;
						}
						else
						{
							savePictureSet = false;
							nr = 0;
							cout << "\nSet of pictures - DONE";
						}
					}
				}
			}
		}
		else if (trackObject == CALCULATE_HSV_OF_ROI)        //Unpause if user made selection
			paused = false;
		//Visual effect of selection
		if (selectObject && selection.width > 0 && selection.height > 0)      
		{
			Mat roi(image, selection);
			bitwise_not(roi, roi);
		}

		imshow("Object Tracker", image);

		//ESC for breaking the loop
		char c = (char)waitKey(10);     
		if (c == 27)
			break;
		//Sort of user interface
		switch (c)                      
		{
		case 's':
			trackObject = NO_TRACKING;
			break;

		case 'p':
			paused = !paused;
			break;

		case 'w':
			savePictureSet = true;
			break;

		default:
			;
		}
	}

	return 0;
}

