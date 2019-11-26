/////////////////////////////////////////////////////////////////////////////
//
// Detect dartboards
//
// g++ detect_dart_with_hough.cpp /usr/lib64/libopencv_core.so.2.4  /usr/lib64/libopencv_highgui.so.2.4  /usr/lib64/libopencv_imgproc.so.2.4  /usr/lib64/libopencv_objdetect.so.2.4 -std=c++11
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
void sobel(cv::Mat &input, Mat_<float> kernel, cv::Mat &convo);
void magDir(cv::Mat &inputx, cv::Mat &inputy, cv::Mat &mag, cv::Mat &dir);
void hough_circle(cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m);
void hough_ellipse(Mat &mag_frame);
void hough_line(Mat &mag_frame, Mat &hough_lines);
void hough_clustered_lines(Mat &mag_frame);

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

void hough_circle(cv::Mat &mag_frame, cv::Mat &dir_frame, Mat &accu_m){
	//Calculate hough space for circles
	//Threshold
	//If above threshold, return true
	const int x = mag_frame.rows;
	const int y = mag_frame.cols;
	const int radius = min(x,y) / 2;  // can be max half the size of the smallest dimension
	int ***accu;
	accu = malloc3dArray(x, y, radius);
	std::cout << x << ',' << y << ',' << radius  << std::endl;
	// Look for strong edges
	for( int i = 0; i < mag_frame.rows; i++){
		for( int j = 0; j < mag_frame.cols; j++){
			if(mag_frame.at<float>(i, j) > 245) {
			        std::cout << i << ',' << j << " SELECTED, mag value -> " << mag_frame.at<float>(i , j) << std::endl;
				// If edge is strong, try ALL circles which have THAT point on its circumference
				for( int r = 10; r < radius; r++){
				       
						// Temporary circle center (a,b)
				                int t = dir_frame.at<float>(i, j);
						int a = i - r*cos(t);
						int b = j - r*sin(t);
						// Check that the radius doesn't exceed the boarder
						if((a-r >0) && (b-r >0) && (a+r < mag_frame.rows) && (b+r < mag_frame.cols)){
							// For each temporary circle, look around circumference and sum the votes
       						int votes = 0;
	       					for( float tempT = 0; tempT < 2*CV_PI; tempT = tempT + (CV_PI/4)) {
		       					votes += mag_frame.at<float>(a + (r*cos(tempT)),
			       				b + (r*sin(tempT)));
				       		}
                                                std::cout << i << ',' << j <<'_'<< a << ',' << b <<',' <<  r << '_' << votes  << std::endl;
					       	// Threshold
						if(votes > 700) {
							
       		 
	       				       
		       					accu_m.at<float>(a, b, r)++;
			       				std::cout << accu_m.at<float>(a, b, r) << endl;
				       			circle(mag_frame, Point(b,a), r, r, 10);
				       		}
				       	}
					
				}
			}
		}
	}
	imwrite( "mag_detected.jpg", mag_frame );
	// return accu;
}



void hough_ellipse(Mat &frame){
	//Calculate hough space for ellipse
	//Threshold
	//If above threshold, return true
}

void hough_line(cv::Mat &frame, cv::Mat &hough_circles){
	//Calculate hough space for line
	//Threshold
	//If above threshold, return true
}

void hough_clustered_lines(Mat &frame){
	//Calculate hough space for lines
	//Threshold on hough value
	//Loop through detected lines
	//If more than X intersect in the same spot, return true
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
	Mat_<float> kernel(3,3);
	kernel << 1, 0, -1, 2, 0, -2, 1, 0, -1;

	// Sobel filter
	Mat xConvo, yConvo, mag, dir;
	sobel(frame_grey,kernel, xConvo);
	sobel(frame_grey, kernel.t(), yConvo);
	magDir( xConvo, yConvo, mag, dir);
	imwrite( "Mag.jpg", mag );
	imwrite( "Dir.jpg", dir );
	int x = mag.rows;
	int y = mag.cols;
	int radius = min(x,y)/2;
	int sizes[] = { x, y, radius };
	Mat accu_m(3, sizes, CV_32FC1, cv::Scalar(0));
	hough_circle( mag, dir, accu_m );
	//Draw detected dartboards from cascade for comparison
	for (int i = 0; i < detected_dartboards.size(); i++){
		rectangle(frame,
			Point(detected_dartboards[i].x, detected_dartboards[i].y),
			Point(detected_dartboards[i].x + detected_dartboards[i].width, detected_dartboards[i].y + detected_dartboards[i].height),
			Scalar( 0, 0, 255 ), 2);
	}
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
	}
}

void sobel(cv::Mat &input, Mat_<float> kernel, cv::Mat &convo){
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

		for ( int i = 1; i < input.rows; i++ )
		{
			for( int j = 1; j < input.cols; j++ )
			{
				double sum = 0.0;
				for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) // i.e -1 to 1 for radius 3
				{
					for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ){
						// find the correct indices we are using
						int imagex = i + m + kernelRadiusX;
						int imagey = j + n + kernelRadiusY;
						int kernelx = m + kernelRadiusX;
						int kernely = n + kernelRadiusY;
						// get the values from the padded image and the kernel
						int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
						double kernalval = kernel.at<double>( kernelx, kernely );
						// do the multiplication
						sum += imageval * kernalval;
						convo.at<float>(i, j) =  sum;
					}
				}
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
	float dirmin = 1000;
	float dirmax = -1000;

	for ( int i = 0; i < inputx.rows; i++ )
	{	
	  for( int j = 0; j < inputx.cols; j++ )
		{

		  float magx = inputx.at<float>(i, j);
		  float magy = inputy.at<float>(i, j);
		  float magTemp = sqrt(pow(magx, 2.0) + pow(magy, 2.0));
                  float dirTemp = atan2(magy,  magx);
		  mag.at<float>(i, j) = magTemp;
		  dir.at<float>(i, j) = dirTemp;		           

		  // Finding min & max values (for normalisation)
		  if(magTemp > magmax) {
                        magmax = magTemp;
		  }
		  if(magTemp < magmin) {
		 	magmin = magTemp;
		  }
                  if(dirTemp > dirmax) {
                        dirmax = dirTemp;
		  }
		  if(dirTemp < dirmin) {
		 	dirmin = dirTemp;
		  }
                        
		}
	}

	// Normalising loop
		for ( int i = 0; i < inputx.rows; i++ )
	  	{	
	  		for( int j = 0; j < inputx.cols; j++ )
		  	{
                                //Normalize calculation
			  	float normMag = ((mag.at<float>(i, j) - magmin) * 255)/ (magmax - magmin);
			   float normDir = ((dir.at<float>(i, j) - dirmin) * 255)/ (dirmax - dirmin);
			  	mag.at<float>(i, j) =  normMag;
			  	dir.at<float>(i, j) =  normDir;
                        }		
	       	}
}
