/////////////////////////////////////////////////////////////////////////////
//
// Detect dartboards
//
// compile with "g++ main.cpp /usr/lib64/libopencv_core.so.2.4  /usr/lib64/libopencv_highgui.so.2.4  /usr/lib64/libopencv_imgproc.so.2.4  /usr/lib64/libopencv_objdetect.so.2.4 -std=c++11"
// run with a.out "image number"
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include<string>

#include "sobel.h"
#include "hough.h"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, int imgnum );
void printFaces(std::vector<Rect> faces);
Rect vect_to_rect(vector<int> vect);
void calculate_all();
float iou_score(Rect A, Rect B);
void printRectangles(Mat &frame, vector<Rect> detected_rectangles);

/** Global variables */
String cascade_name = "dart.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv ){
	string imgnum = argv[1];
	if (imgnum == "all"){
		calculate_all();
	} else {
		string filename =  "dartPictures/dart"+imgnum+".jpg";
		Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n");
		return -1; };
		detectAndDisplay( frame, stoi(imgnum) );
		imwrite( "detected.jpg", frame );
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int imgnum){
	std::vector<Rect> detected_dartboards;
	Mat frame_grey;
	cvtColor( frame, frame_grey, CV_BGR2GRAY );
  //GaussianBlur(frame_grey, frame_grey, Size(3,3), 2); // Does this help?
	equalizeHist( frame_grey, frame_grey );
	cascade.detectMultiScale( frame_grey,
		detected_dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE,
		Size(50, 50), Size(500,500) );
		cout << "Original detected dartboards" << endl;
		printFaces(detected_dartboards);

	// Sobel filter
	Mat xConvo, yConvo, mag, dir;
	sobel(frame_grey, xConvo, yConvo, mag, dir);

	// Hough
	Mat points;
	hough(frame, mag, dir, points);

	vector<Rect> detected_rectangles;
	int inc = 0;
	for (int i = 0; i < detected_dartboards.size(); i++){

		// Create a cropped image of each voilaJones detection
		// '/2' because points is half the size of the frame
		Rect rect = Rect((detected_dartboards[i].x / 2)-1,
																     (detected_dartboards[i].y / 2)-1,
		                                 (detected_dartboards[i].width / 2)-1,
																		 (detected_dartboards[i].height / 2)-1);
		Mat violaJones = points(rect);
 		bool detected = false;
 		int center[2] = {0,0};
 		hough_detect(violaJones, &detected, center);

		if(detected) {
			int width = detected_dartboards[i].width;
			int height = detected_dartboards[i].height;
			int oldX = detected_dartboards[i].x;
			int oldY = detected_dartboards[i].y;
			int newX = oldX - width/2 + center[1];
		  int newY = oldY - height/2 + center[0];
			Rect newRect = Rect(newX, newY, width, height);

			if(inc == 0){
					detected_rectangles.push_back(newRect);
			}
			bool add = false;
		  for(int k = 0; k < detected_rectangles.size(); k++) {
				float iou = iou_score(detected_rectangles[k], newRect);
		    if(iou > 0){
					if(detected_rectangles[k].width > newRect.width){
					  add = true;
						detected_rectangles.erase(detected_rectangles.begin() + k);
						inc++;
						break;
					}else{
						add = false;
						inc++;
						break;
					}
				}else{ add = true; }
			inc++;
			}
			if(add){
				detected_rectangles.push_back(newRect);
			}
		}
	}
	printRectangles(frame, detected_rectangles);
}

float iou_score(Rect A, Rect B){
    Rect intersection = A&B;
    float iou = (float)intersection.area() /
		((float)A.area()+(float)B.area()-(float)intersection.area());
    return iou;
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
	}
}

void printRectangles(Mat &frame, vector<Rect> detected_rectangles){
	for (int i = 0; i < detected_rectangles.size(); i++){
		rectangle(frame,
							Point(detected_rectangles[i].x, detected_rectangles[i].y),
							Point(detected_rectangles[i].x + detected_rectangles[i].width,
										detected_rectangles[i].y + detected_rectangles[i].height),
							Scalar( 0, 255, 0 ), 2);
	}
}

int ***malloc3dArray(int dim1, int dim2, int dim3){
	int i, j, k;
	int ***array = (int ***) malloc(dim1 * sizeof(int **));
	for (i = 0; i < dim1; i++) {
		array[i] = (int **) malloc(dim2 * sizeof(int *));
		for (j = 0; j < dim2; j++) {
			array[i][j] = (int *) malloc(dim3 * sizeof(int));
		}
	}
	return array;
}

void calculate_all(){
	for (int imgnum = 0; imgnum<16; imgnum++){
		string filename =  "dart"+to_string(imgnum)+".jpg";
		Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); };
		detectAndDisplay( frame, imgnum);
		//imwrite( "detected"+to_string(imgnum)+".jpg", frame );
	}
}

void normaliseMatrix( Mat matrix ) {
	double min, max;
	cv::minMaxLoc(matrix, &min, &max);
	for ( int i = 0; i < matrix.rows; i++ )
	{
		for( int j = 0; j < matrix.cols; j++ )
		{
			//Normalize calculation
			matrix.at<float>(i, j) = (((matrix.at<float>(i, j) - min) * 255.0) /
			(max - min));
		}
	}
}
