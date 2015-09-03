
#include "PCLAdapter.h"
#include "OpenCV3Linker.h"
#include "Kinect2WithOpenCVWrapper.h"
#include "ColorTable.h"

//using namespace cv;

int main(void)
{
	Kinect2WithOpenCVWrapper kinect;
	kinect.enableColorFrame();
	kinect.enableDepthFrame();
	
	ColorTable table;
	table.generate16bitPalette();
	cv::imshow("Color Palette", table.miniColorTable);

	while (1)
	{
		cv::Mat colorImg, depthMat, depthMatd, depthImg, depthError;
		kinect.getColorFrame(colorImg);
		kinect.getDepthFrame(depthMat);
		depthError = cv::Mat(depthMat.size(), CV_8UC1);
		depthMat.convertTo(depthMatd, CV_32F);
		cv::threshold(depthMatd, depthError, 1, 1, cv::THRESH_BINARY);
		if(!depthMat.empty()) table.lookup16UC1to8UC3(depthMat, depthImg);
		//depthMat.convertTo(depthImg, CV_8U, -255.0f / 8000.0f, 255.0f);		//	ãﬂÇ¢ÇŸÇ«îíÇ¢

		//	ÉtÉåÅ[ÉÄÇÃï`âÊ
		cv::imshow("Color Image", colorImg);
		cv::imshow("Depth Image", depthImg);
		cv::imshow("Depth Error", depthError);

		if (cv::waitKey(20) == VK_ESCAPE) break;
	}
	kinect.releaseAllInterface();
}