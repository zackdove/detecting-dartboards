/////////////////////////////////////////////////////////////////////////////
//
// Calculate the F1-Score for the faces.
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
vector<Rect> ground_truth(int filenum);
Rect vect_to_rect(vector<int> vect);
float iou(Rect A, Rect B);
int calculate_all();
float f1_score();

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;
float TP = 0;
float FP = 0;
float RP = 0;

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
	std::cout << "TPs = " << TP << std::endl;
	std::cout << "FPs = " << FP << std::endl;
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
	 float f1 = f1_score();
	 std::cout << "f1 score = " << f1 << std::endl;
	 return 0;
 }

float f1_score(){
	float precision = TP/(TP+FP);
	std::cout << "precision  = " << precision << std::endl;
	float recall  = TP/(RP);
	std::cout << "recall  = " << recall << std::endl;
	return 2*(precision*recall)/(precision+recall);
}

/** @function detectAndDisplay */
	void detectAndDisplay( Mat frame, int imgnum){
	std::vector<Rect> detected_faces;
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	cascade.detectMultiScale( frame_gray, detected_faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	vector<Rect> truth_faces = ground_truth(imgnum);
	printFaces(truth_faces);
	float threshold = 0.5;
	int true_positives = 0;
	int false_positives = 0;
	//Calculate IOU for each
	for( int i = 0; i < detected_faces.size(); i++ ){
		rectangle(frame, Point(detected_faces[i].x, detected_faces[i].y), Point(detected_faces[i].x + detected_faces[i].width, detected_faces[i].y + detected_faces[i].height), Scalar( 0, 255, 0 ), 2);
		for (int j = 0; j < truth_faces.size(); j++){
			float iou_result = iou(detected_faces[i], truth_faces[j]);
			// std::cout << "IOU for ground truth face " << truth_faces[j] << " with detected face " << detected_faces[i] << " = " << iou_result << std::endl;
			if (iou_result >= threshold){
				true_positives++;
			}
		}
	}
	TP += true_positives;
	FP += detected_faces.size()-true_positives;
	//Draw for groud truthh
	for (int i = 0; i < truth_faces.size(); i++){
		rectangle(frame, Point(truth_faces[i].x, truth_faces[i].y), Point(truth_faces[i].x + truth_faces[i].width, truth_faces[i].y + truth_faces[i].height), Scalar( 0, 0, 255 ), 2);
	}
}

void printFaces(std::vector<Rect> faces){
	for (int i = 0; i < faces.size(); i++){
		std::cout << faces[i] << std::endl;
	}
}

//Using https://www.geeksforgeeks.org/csv-file-management-using-c/
//Can optimise by storing into an array
vector<Rect> ground_truth(int filenum){
	// std::cout << filenum << std::endl;
	// File pointer
    ifstream fin("face_coords.csv");
	int file2, count = 0;
	vector<string> row;
    string line, word, temp;
	vector<int> face_coords;
	vector<Rect> face_coords_set;
	while (getline(fin, line)) {
        row.clear();
        istringstream iss(line);
        while (getline(iss, word, ',')) {
            row.push_back(word);
        }
        file2 = stoi(row[0]);
        if (file2 == filenum) {
            count = 1;
			row.erase(row.begin());
			// std::cout << "row size = " << row.size() << std::endl;
			for (int i=0; i<row.size(); i=i+4){
				RP++;
				face_coords_set.push_back(Rect(stoi(row[i]),stoi(row[i+1]),stoi(row[i+2])-stoi(row[i]),stoi(row[i+3])-stoi(row[i+1])));
			}
			return face_coords_set;
            break;
        }
    }
    if (count == 0)
        cout << "Record not found\n";
		return face_coords_set;
}

Rect vect_to_rect(vector<int> vect){
	return Rect(vect[0], vect[1], vect[2]-vect[0], vect[3]-vect[1]);
}

float iou(Rect A, Rect B){
	std::cout << "A " << A << std::endl;
	std::cout << "B " << B << std::endl;
    Rect intersection = A&B;
	std::cout << "inter " << intersection.area() << std::endl;
    float iou = (float)intersection.area() / ((float)A.area()+(float)B.area()-(float)intersection.area());
	std::cout << "iou " << std::fixed << std::setprecision(5) << iou << std::endl;
    return iou;
}
