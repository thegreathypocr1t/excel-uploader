#include <iostream>
#include <stdio.h>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

string faceXmlPath = "/home/ian/Documents/QtProjects/faceDetect/haarcascade_frontalface_alt.xml";
string eyesXmlPath = "/home/ian/Documents/QtProjects/faceDetect/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;

void detectAndShow(Mat frame);

int main() {
    VideoCapture cap;
    Mat frame;

    if(!faceCascade.load(faceXmlPath))
        cout<<"Cannot load face cascade xml file"<<endl;
    if(!eyesCascade.load(eyesXmlPath))
        cout<<"Cannot load eyes cascade xml file"<<endl;

    cap.open(0);
    if(!cap.isOpened()) {
        cout<<"Could not open camera."<<endl;
    }
    while(cap.read(frame)) {
        if(frame.empty()) {
            cout<<"Current frame is empty"<<endl;
            break;
        }

        detectAndShow(frame);
        int c = waitKey(10);
        if((char)c == 27) {
            break;
        }
    }
    return 0;
}

void detectAndShow(Mat frame) {
    Mat grayFrame;
    std::vector<Rect>  faces;

    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    equalizeHist(grayFrame, grayFrame);

    faceCascade.detectMultiScale(grayFrame, faces, 1.2, 3, 0, Size(30, 30));
    for(unsigned int i = 0; i < faces.size(); i++) {
        rectangle(frame, faces[i], Scalar(0, 255, 0), 2, 8, 0);

        Mat faceROI = grayFrame(faces[i]);
        std::vector<Rect> eyes;

        eyesCascade.detectMultiScale(faceROI, eyes, 1.2, 3, 0, Size(30, 30));
        for(unsigned int j = 0; j < eyes.size(); j++) {

            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    namedWindow("Face Detection");
    imshow("Face Detection", frame);
}
