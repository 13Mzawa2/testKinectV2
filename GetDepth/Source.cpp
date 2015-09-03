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
	//	KinectのDepthフレームを読みだすための準備
	IDepthFrameSource *depthSource;		//	Depthフレームを取得するためのSourceインターフェース
	if (FAILED(kinect->get_DepthFrameSource(&depthSource)))
	{
		cerr << "Error: IKinectSensor::get_DepthFrameSource" << endl;
		return 3;
	}
	IDepthFrameReader *depthReader;		//	DepthFrameSourceからDepthFrameを読みだすためのReaderインターフェース
	if (FAILED(depthSource->OpenReader(&depthReader)))
	{
		cerr << "Error: IDepthFrameSource::OpenReader" << endl;
		return 4;
	}
	//	OpenCVのDepthフレームの準備
	IFrameDescription *depthDescription;	//	Depthフレームの詳細
	if (FAILED(depthSource->get_FrameDescription(&depthDescription)))
	{
		cerr << "Error: IDepthFrameSource::get_FrameDescription" << endl;
		return 5;
	}
	//	OpenCV側のバッファ作成
	Size bufferSize;
	depthDescription->get_Width(&bufferSize.width);
	depthDescription->get_Height(&bufferSize.height);
	Mat bufferImg(bufferSize, CV_16UC1);			//	Depthバッファは16bit
	UINT bufferBlockSize = bufferImg.rows * bufferImg.cols * bufferImg.channels() * sizeof(ushort);
	//	表示用画像
	Mat depthImg(bufferSize, CV_8UC1);
	namedWindow("Depth Image");
	while (1)
	{
		IDepthFrame *depthFrame = nullptr;
		//	Depthフレーム取得
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	DepthフレームからデータをOpenCV側バッファにコピー
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(&bufferBlockSize, reinterpret_cast<UINT16**>(&bufferImg.data))))
				bufferImg.convertTo(depthImg, CV_8U, -255.0f/8000.0f, 128.0f);		//	近いほど白い
		}
		//	Depthフレームの解放（次の取得命令の待機）
		safeRelease(depthFrame);
		//	Depthフレームの描画
		imshow("Depth Image", depthImg);
		if (waitKey(1) == VK_ESCAPE) break;
	}
	//	全インターフェースの解放
	safeRelease(depthSource);
	safeRelease(depthReader);
	safeRelease(depthDescription);
	safeRelease(kinect);
	cv::destroyAllWindows();

	return 0;
}