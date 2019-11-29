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
void sobel(cv::Mat &input, Mat_<int> kernel, cv::Mat &convo);
void magDir(cv::Mat &inputx, cv::Mat &inputy, cv::Mat &mag, cv::Mat &dir);
void hough_circle(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m);
void hough_ellipse(Mat &mag_frame);
void normaliseMatrix( Mat matrix );
vector<vector<int> > hough_line(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m);
vector<vector<int> > hough_clustered_lines(cv::Mat &frame, vector<vector<int> > points_lines);

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
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
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
	equalizeHist( frame_grey, frame_grey );
	cascade.detectMultiScale( frame_grey,
		detected_dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE,
		Size(50, 50), Size(500,500) );
		cout << "Original detected dartboards" << endl;
		printFaces(detected_dartboards);
		//Draw detected dartboards from cascade for comparison
		for (int i = 0; i < detected_dartboards.size(); i++){
			rectangle(frame,
				Point(detected_dartboards[i].x, detected_dartboards[i].y),
				Point(detected_dartboards[i].x + detected_dartboards[i].width,
					detected_dartboards[i].y + detected_dartboards[i].height),
					Scalar( 0, 0, 255 ), 2);
		}

		// create the SOBEL kernel in 1D, Y is transpose
		Mat_<int> kernel(3,3);
		Mat_<int>kernelT(3,3);
		kernel << -1, 0, 1, -2, 0, 2, -1, 0, 1;
		kernelT << 1, 2, 1, 0, 0, 0, -1, -2, -1;

		// Sobel filter
		Mat xConvo, yConvo, mag, dir;
		sobel(frame_grey,kernel, xConvo);
		sobel(frame_grey, kernelT, yConvo);
		imwrite( "x.jpg", xConvo );
		imwrite( "y.jpg", yConvo );
		magDir( xConvo, yConvo, mag, dir);
		imwrite( "Mag.jpg", mag );
		imwrite( "Dir.jpg", dir );

		Mat points_circle;
		int x = mag.rows;
		int y = mag.cols;

		int radius = min(x,y)/2;
		int sizes_circles[] = { x, y, radius };
		Mat accu_circles(3, sizes_circles, CV_32FC1, cv::Scalar(0));
		hough_circle( frame, mag, dir, accu_circles, points_circle );

		Mat accu_lines;
		//vector<vector<int> > points_lines = hough_line( frame, mag, dir, accu_lines);
		//vector<vector<int> > points_clustered_lines = hough_clustered_lines(frame, points_lines);
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
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
		imwrite( "detected"+to_string(imgnum)+".jpg", frame );
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
			matrix.at<float>(i, j) = (((matrix.at<float>(i, j) - min) * 255.0) / (max - min));
			//std::cout << min << ',' << max << ',' << matrix.at<float>(i, j) << std::endl;
		}
	}
}
