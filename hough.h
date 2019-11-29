/////////////////////////////////////////////////////////////////////////////
//
// HOUGH header functions
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

using namespace std;
using namespace cv;

/** Function Headers */
void normaliseMatrix( Mat matrix );


void hough_circle(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m, Mat &points){
	points.create(mag_frame.size(), CV_32FC1);
	const int x = mag_frame.rows;
	const int y = mag_frame.cols;
	const int radius = min(x,y) / 2;  // can be max half the size of the smallest dimension

	// Look for strong edges
	for( int i = 0; i < mag_frame.rows; i++){
		for( int j = 0; j < mag_frame.cols; j++){
			// Pixel threshold
			if(mag_frame.at<float>(i, j) > 50) {
				// Vote for circles corresponding to point's +ve and -ve direction
				for( int r = 30; r < radius; r++){
					float d = dir_frame.at<float>(i, j);
					int a = i + r*cos(d);
					int b = j + r*sin(d);
					// Check that the radius doesn't exceed the boarder
					if((a > 0) && (b > 0) && (a < x) && (b < y)){
						accu_m.at<int>(a, b, r) += 1;
					}
					a = i - r*cos(d);
					b = j - r*sin(d);
					// Check that the radius doesn't exceed the boarder
					if((a > 0) && (b > 0) && (a < x) && (b < y)){
						accu_m.at<int>(a, b, r) += 1;
					}
				}
			}
		}
	}

	// Condensing points to (x,y) by summing r
	for( int i = 0; i < mag_frame.rows; i++){
		for( int j = 0; j < mag_frame.cols; j++){
			for( int r = 30; r < radius; r++){
				points.at<float>(i, j) += accu_m.at<int>(i,j,r);
			}
			if(points.at<float>(i, j) > 15){
				circle(frame, Point(j, i), 10 , Scalar( 0, 0, 255 ), 2);
			}
		}
	}
	normaliseMatrix( points );
	imwrite("points.jpg", points);
}

void hough_ellipse(Mat &frame){
	//Calculate hough space for ellipse
	//Threshold
	//If above threshold, return true
}

vector<vector<int> > hough_line(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m){

	vector<vector<int> > points_lines;
	const int x = mag_frame.rows/2;
	const int y = mag_frame.cols/2;
	const int d = sqrt((mag_frame.rows*mag_frame.rows) + (mag_frame.cols*mag_frame.cols)) / 2;
	const int mini = min(x, y);
	accu_m = Mat(d, 180, CV_32S, cvScalar(0));

	// Look for strong edges
	for( int i = 0; i < mag_frame.rows; i++){
		for( int j = 0; j < mag_frame.cols; j++){
			if(mag_frame.at<float>(i, j) > 100) { // Threshhold
				int xd = i-x;
				int yd = j-y;
				// If edge is strong, vote for lines corresponding to its point
				for( int r = 0; r < 180; r++){
					int p = abs((xd*cos(r*CV_PI/180)) + (yd*sin(r*CV_PI/180)));
					accu_m.at<int>(p, r)++;
				}
			}
		}
		for( int p = 0; p < d; p++) {
			for( int r = 0; r < 180; r++) {
				if(accu_m.at<int>(p, r) > 200) { // Threshold
					float a = cos(r*CV_PI/180);
					float b = sin(r*CV_PI/180);
					int x0 = x + a*p;
					int y0 = y + b*p;
					int x1 = int(x0 + mini*(-b));
					int y1 = int(y0 + mini*(a));
					int x2 = int(x0 - mini*(-b));
					int y2 = int(y0 - mini*(a));
					int points_array[] = {x1,y1,x2,y2};
					vector<int> points(points_array, points_array +sizeof(points_array)/sizeof(int));

					points_lines.push_back(points);
					line(frame, Point(y1,x1), Point(y2,x2), (0,0,255), 2);
				}
			}
		}
	}
	return points_lines;
}

vector<vector<int> > hough_clustered_lines(Mat &frame, vector<vector<int> > points_lines){
	vector<vector<int> > points_clustered_lines;
	Mat accu_m = Mat(frame.rows, frame.cols, CV_32S, cvScalar(0));
	for( int p = 0; p < points_lines.size(); p++){
		float x1 = (float)points_lines[p][0];
		float y1 = (float)points_lines[p][1];
		float x2 = (float)points_lines[p][2];
		float y2 = (float)points_lines[p][3];
		float m = (y1 - y2) / (x1 - x2);
		float c = y1 - (m*x1);

		for(int x = 0; x < frame.rows; x ++){
			int y = (int)(x*m) + c;
			//std::cout << p << ',' << points_lines.size() << '/' << m << ',' << c << ',' << '/' << x << ',' << y << std::endl;
			if( y > 0 && y < frame.rows){
				accu_m.at<float>(x, y)++;
			}
		}
	}
	normaliseMatrix(accu_m);
	imwrite("points.jpg", accu_m);
	return points_clustered_lines;
}
