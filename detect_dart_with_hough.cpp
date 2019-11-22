/////////////////////////////////////////////////////////////////////////////
//
// Detect dartboards
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
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
//For printing correct precision
#include <iomanip>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, int imgnum );
void printFaces(std::vector<Rect> faces);
Rect vect_to_rect(vector<int> vect);
int calculate_all();
bool contains_circle(Mat frame);
bool contains_line(Mat frame);
bool contains_ellipse(Mat frame);
bool contains_clustered_lines(Mat frame);

/** Global variables */
String cascade_name = "frontalface.xml";
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

 int calculate_all(){
    for (int imgnum = 0; imgnum<16; imgnum++){
    string filename =  "dart"+to_string(imgnum)+".jpg";
    	Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
    	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    	detectAndDisplay( frame, imgnum);
    	imwrite( "detected"+to_string(imgnum)+".jpg", frame );
    }
    return 0;
 }

bool contains_circle(Mat frame){
    //Calculate hough space for circle
    //Threshold
    //If above threshold, return true
    return true;
}

bool contains_line(Mat frame){
    //Calculate hough space for line
    //Threshold
    //If above threshold, return true
    return true;
}

bool contains_ellipse(Mat frame){
    //Calculate hough space for ellipse
    //Threshold
    //If above threshold, return true
    return true;
}

bool contains_clustered_lines(Mat frame){
    //Calculate hough space for lines
    //Threshold on hough value
    //Loop through detected lines
    //  If more than X intersect in the same spot, return true
    return true;
}

/** @function detectAndDisplay */
	void detectAndDisplay( Mat frame, int imgnum){
	std::vector<Rect> detected_faces;
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	cascade.detectMultiScale( frame_gray, detected_faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
    cout << "Original detected faces" << endl;
	printFaces(detected_faces);
    for (int i = 0; i < detected_faces.size(); i++){
        Mat face_frame = frame(detected_faces[i]);
        //Check if face_frame cotains circle, line, ellipse etc
        //If it doesnt meet the criteria, erase it
    }
	//Draw for groud truthh
	for (int i = 0; i < detected_faces.size(); i++){
		rectangle(frame, Point(detected_faces[i].x, detected_faces[i].y), Point(detected_faces[i].x + detected_faces[i].width, detected_faces[i].y + detected_faces[i].height), Scalar( 0, 0, 255 ), 2);
	}
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
	}
}
