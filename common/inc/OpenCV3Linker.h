#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#pragma region OPENCV3_LIBRARY_LINKER
#ifdef _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif
#define CV_VER  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
//#pragma comment(lib, "opencv_calib3d" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_core" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_features2d" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_flann" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_hal" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_highgui" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_imgcodecs" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_imgproc" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_ml" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_objdetect" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_photo" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_shape" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_stitching" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_superres" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_ts" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_video" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_videoio" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_videostab" CV_VER CV_EXT)
//#pragma comment(lib, "opencv_viz" CV_VER CV_EXT)
#pragma comment(lib, "opencv_world" CV_VER CV_EXT)
#pragma endregion

#pragma region MACRO

//cv::Matのピクセル値拾得用マクロ
#define matB(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 0])
#define matG(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 1])
#define matR(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 2])
#define matGRAY(IMG,X,Y)	matB(IMG,X,Y)
#define matBf(IMG,X,Y)		(((cv::Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].x)
#define matGf(IMG,X,Y)		(((cv::Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].y)
#define matRf(IMG,X,Y)		(((cv::Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].z)
#define matBd(IMG,X,Y)		(((cv::Point3d*)((IMG).data + (IMG).step.p[0] * Y))[X].x)
#define matGd(IMG,X,Y)		(((cv::Point3d*)((IMG).data + (IMG).step.p[0] * Y))[X].y)
#define matRd(IMG,X,Y)		(((cv::Point3d*)((IMG).data + (IMG).step.p[0] * Y))[X].z)

#pragma endregion