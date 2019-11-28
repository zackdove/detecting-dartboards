/////////////////////////////////////////////////////////////////////////////
//
// Detect dartboards
//
// compile with "g++ detect_dart_with_hough.cpp /usr/lib64/libopencv_core.so.2.4  /usr/lib64/libopencv_highgui.so.2.4  /usr/lib64/libopencv_imgproc.so.2.4  /usr/lib64/libopencv_objdetect.so.2.4 -std=c++11"
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
//For printing correct precision
#include <iomanip>
#include <iostream>
#include<string>


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
vector<vector<int>> hough_line(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m);
vector<vector<int>> hough_clustered_lines(vector<vector<int>> points_lines);

/** Global variables */
String cascade_name = "dart.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv ){
	string imgnum = argv[1];
	if (imgnum == "all"){
		calculate_all();
	} else {
		string filename =  "dart"+imgnum+".jpg";
		Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
		detectAndDisplay( frame, stoi(imgnum) );
		imwrite( "detected.jpg", frame );
	}
	return 0;
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

void hough_circle(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m, Mat &points){
	//Calculate hough space for circles
	//Threshold
	//If above threshold, return true
        points.create(mag_frame.size(), CV_32FC1);
	const int x = mag_frame.rows;
	const int y = mag_frame.cols;
	const int radius = min(x,y) / 2;  // can be max half the size of the smallest dimension

	// Look for strong edges
	for( int i = 0; i < mag_frame.rows; i++){
	  for( int j = 0; j < mag_frame.cols; j++){
	    // Pixel threshold
	    if(mag_frame.at<float>(i, j) > 200) {
	      // If edge is strong, vote for circles corresponding to its direction
	      for( int r = 30; r < radius; r++){ 
		for (int k = -1; k < 2; k++){
		  for (int l = -1; l < 2; l++){
		    // Circle centers around selected point 
		      int d = dir_frame.at<float>(i, j);
		      int a = i + k*r*cos(d*CV_PI/180);
		      int b = j + l*r*sin(d*CV_PI/180);
		      // Check that the radius doesn't exceed the boarder
		      if((a > 0) && (b > 0) && (a < x) && (b < y)){	
			accu_m.at<float>(a, b, r) += 1;
		      }
		  } 
		}
	      }	
	    }
	  }
	}

	// Condensing points to just (x,y)
        for( int i = 0; i < mag_frame.rows; i++){
	  for( int j = 0; j < mag_frame.cols; j++){
	    for( int r = 30; r < radius; r++){ 
	      points.at<int>(i, j) += accu_m.at<float>(i,j,r);
	    }
	    if(points.at<int>(i, j) > 300){
	      std::cout << i << ',' << j << ',' << " points = " << points.at<int>(i, j) << std::endl;
	      circle(frame, Point(j, i), 10 , Scalar( 0, 0, 255 ), 2);
	    }
	  }
	} 
}

void hough_ellipse(Mat &frame){
	//Calculate hough space for ellipse
	//Threshold
	//If above threshold, return true
}

vector<vector<int>> hough_line(cv::Mat &frame, cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m){

  vector<vector<int>> points_lines;
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
    for( int k = 0; k < d; k++) {
      for( int l = 0; l < 180; l++) {
	if(accu_m.at<int>(k, l) > 200) { // Threshold
	 float a = cos(l*CV_PI/180);
	 float b = sin(l*CV_PI/180);
	 int x0 = x + a*k;
	 int y0 = y + b*k;
	 int x1 = int(x0 + mini*(-b));
	 int y1 = int(y0 + mini*(a));
	 int x2 = int(x0 - mini*(-b));
	 int y2 = int(y0 - mini*(a));
	 vector<int> points = {x1,y1,x2,y2};
	 points_lines.push_back(points);
	 line(frame, Point(y1,x1), Point(y2,x2), (0,0,255), 2);
	} 
      }		 
    }
  }
  return points_lines;
}

vector<vector<int>> hough_clustered_lines(vector<vector<int>> points_lines){
  vector<vector<int>> points_clustered_lines;

  for( int p = 0; p < points_lines.size(); p++){ 
	 int x1 = points_lines[p][0];
	 int y1 = points_lines[p][1];
	 int x2 = points_lines[p][2];
	 int y2 = points_lines[p][3];
	std::cout << x1 << ',' << x2 << ',' << y1 << ',' << y2 << std::endl;
      	}
  return points_clustered_lines;
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

	//for (int i = 0; i < detected_dartboards.size(); i++){
	//Mat dartboard_frame = frame(detected_dartboards[i]);
	//Check if face_frame cotains circle, line, ellipse etc
	//If it doesnt meet the criteria, erase it

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
	//hough_circle( frame, mag, dir, accu_circles, points_circle );

	Mat accu_lines;
	vector<vector<int>> points_lines = hough_line( frame, mag, dir, accu_lines);
        vector<vector<int>> points_clustered_lines = hough_clustered_lines(points_lines); 

	//Draw detected dartboards from cascade for comparison
	for (int i = 0; i < detected_dartboards.size(); i++){
		rectangle(frame,
			Point(detected_dartboards[i].x, detected_dartboards[i].y),
			Point(detected_dartboards[i].x + detected_dartboards[i].width, 
			      detected_dartboards[i].y + detected_dartboards[i].height),
			Scalar( 0, 0, 255 ), 2);
	}
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
	}
}

void sobel(cv::Mat &input, Mat_<int> kernel, cv::Mat &convo){
	// intialise the output using the input
	convo.create(input.size(), CV_32FC1);

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );
		for ( int i = 0; i < input.rows; i++ )
		{
		  for( int j = 0; j < input.cols; j++ )
		    {
		      float sum = 0.0;
		      for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
			  for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ){
			    // find the correct indices we are using
			    int imagex = i + m + kernelRadiusX;
			    int imagey = j + n + kernelRadiusY;
			    int kernelx = m + kernelRadiusX;
			    int kernely = n + kernelRadiusY;
			    
			    // get the values from the padded image and the kernel
			    int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
			    int kernalval = kernel.at<int>( kernelx, kernely );
			    
			    // do the multiplication
			    sum = sum + ( imageval * kernalval );
			    
			  }
			}
		      convo.at<float>(i, j) = sum;
		    }
		}
}

void magDir(cv::Mat &inputx, cv::Mat &inputy, cv::Mat &mag, cv::Mat &dir)
{
  // intialise the output using the input
  mag.create(inputx.size(), CV_32FC1);
  dir.create(inputx.size(), CV_32FC1);
  
  float magmin = 1000;
  float magmax = -1000;
  
  for ( int i = 0; i < inputx.rows; i++ )
    {	
      for( int j = 0; j < inputx.cols; j++ )
	{
	  
	  float magx = inputx.at<float>(i, j);
	  float magy = inputy.at<float>(i, j);
	  float magTemp = sqrt((magx*magx) + (magy*magy));
	  float dirTemp = atan2(magy,  magx) * 180 / CV_PI;
	  mag.at<float>(i, j) = magTemp;
	  dir.at<float>(i, j) = dirTemp;		           
	  
	  // Finding min & max values (for normalisation)
	  if(magTemp > magmax) {
	    magmax = magTemp;
	  }
	  if(magTemp < magmin) {
	    magmin = magTemp;
	  }  
	}
    }
  
  //Normalising loop
  for ( int i = 0; i < inputx.rows; i++ )
    {	
      for( int j = 0; j < inputx.cols; j++ )
	{
	  //Normalize calculation
	  float normMag = ((mag.at<float>(i, j) - magmin) * 255)/ (magmax - magmin);
	  mag.at<float>(i, j) =  normMag;
	}		
    }
}
