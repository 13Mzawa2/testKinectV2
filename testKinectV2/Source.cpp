#include "OpenCV3Linker.h"
#include <Kinect.h>
#pragma comment(lib, "kinect20.lib")
using namespace cv;
using namespace std;

template<class Interface>
inline void safeRelease(Interface *&interfacePointer)
{
	if (interfacePointer != NULL)
	{
		interfacePointer->Release();
		interfacePointer = NULL;
	}
}
inline void safeRelease(IKinectSensor *&kinectSensor)
{
	if (kinectSensor) kinectSensor->Close();
	if (kinectSensor != NULL)
	{
		kinectSensor->Release();
		kinectSensor = NULL;
	}
}

int main(void)
{
	IKinectSensor *kinect;		//	Kinectセンサの用意
	//	DefaultのKinectを取得
	if (FAILED(GetDefaultKinectSensor(&kinect)))
	{
		cerr << "Error: GetDefaultKinectSensor" << endl;
		return 1;
	}
	//	Kinectを開く
	if (FAILED(kinect->Open()))
	{
		cerr << "Error: IKinectSensor::Open" << endl;
		return 2;
	}
	//	KinectのColorフレームを読みだすための準備
	IColorFrameSource *colorSource;		//	Colorフレームを取得するためのSourceインターフェース
	if (FAILED(kinect->get_ColorFrameSource(&colorSource)))
	{
		cerr << "Error: IKinectSensor::get_ColorFrameSource" << endl;
		return 3;
	}
	IColorFrameReader *colorReader;		//	ColorFrameSourceからColorFrameを読みだすためのReaderインターフェース
	if (FAILED(colorSource->OpenReader(&colorReader)))
	{
		cerr << "Error: IColorFrameSource::OpenReader" << endl;
		return 4;
	}
	//	OpenCVのColorフレームの準備
	IFrameDescription *colorDescription;	//	Colorフレームの詳細
	if (FAILED(colorSource->get_FrameDescription(&colorDescription)))
	{
		cerr << "Error: IColorFrameSource::get_FrameDescription" << endl;
		return 5;
	}
	//	OpenCV側のバッファ作成
	Size bufferSize;
	colorDescription->get_Width(&bufferSize.width);
	colorDescription->get_Height(&bufferSize.height);
	Mat bufferImg(bufferSize, CV_8UC4);
	unsigned int bufferBlockSize = bufferImg.rows * bufferImg.cols * bufferImg.channels() * sizeof(uchar);
	//	表示用画像
	Mat colorImg(bufferImg.rows/2, bufferImg.cols/2, CV_8UC4);
	namedWindow("Color Image");
	while (1)
	{
		IColorFrame *colorFrame = nullptr;
		//	Colorフレーム取得
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	ColorフレームからデータをOpenCV側バッファにコピー
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				bufferBlockSize, reinterpret_cast<BYTE*>(bufferImg.data), ColorImageFormat_Bgra)))
				resize(bufferImg, colorImg, Size(), 0.5, 0.5);
		}
		//	Colorフレームの解放（次の取得命令の待機）
		safeRelease(colorFrame);
		//	Colorフレームの描画
		imshow("Color Image", colorImg);
		if (waitKey(1) == VK_ESCAPE) break;
	}
	//	全インターフェースの解放
	safeRelease(colorSource);
	safeRelease(colorReader);
	safeRelease(colorDescription);
	safeRelease(kinect);
	cv::destroyAllWindows();

	return 0;
}