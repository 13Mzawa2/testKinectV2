#pragma once

#include <opencv2\opencv.hpp>
#include <Kinect.h>
#pragma comment(lib, "kinect20.lib")

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

class Kinect2WithOpenCVWrapper
{
protected:
	//	KinectSDK2 Interface
	IKinectSensor *kinect;
	//	for color buffer
	IColorFrameSource *colorSource;
	IColorFrameReader *colorReader;
	IFrameDescription *colorDescription;
	//	for depth buffer
	IDepthFrameSource *depthSource;
	IDepthFrameReader *depthReader;
	IFrameDescription *depthDescription;
	//	mapping engine
	ICoordinateMapper *mapper;
	//	OpenCV Buffer Size
	cv::Size colorBufferSize;
	cv::Size depthBufferSize;
	UINT colorBufferBlockSize;
	UINT depthBufferBlockSize;

public:
	//	OpenCV Image Buffer
	cv::Mat colorBuffer;		//	1920 x 1080, 8UC4
	cv::Mat depthBuffer;		//	512 x 424, 16UC1
	

	Kinect2WithOpenCVWrapper()
	{
		//	DefaultのKinectを取得
		if (FAILED(GetDefaultKinectSensor(&kinect)))
		{
			std::cerr << "Error: GetDefaultKinectSensor" << std::endl;
			exit(-1);
		}
		//	Kinectを開く
		if (FAILED(kinect->Open()))
		{
			std::cerr << "Error: IKinectSensor::Open" << std::endl;
			exit(-1);
		}
	}

	//------------------------
	//	Color Frame Open
	//------------------------
	inline void enableColorFrame()
	{
		//	KinectのColorフレームを読みだすための準備
		if (FAILED(kinect->get_ColorFrameSource(&colorSource)))
		{
			std::cerr << "Error: IKinectSensor::get_ColorFrameSource" << std::endl;
			exit(-1);
		}
		if (FAILED(colorSource->OpenReader(&colorReader)))
		{
			std::cerr << "Error: IColorFrameSource::OpenReader" << std::endl;
			exit(-1);
		}
		//	OpenCVのColorフレームの準備
		if (FAILED(colorSource->get_FrameDescription(&colorDescription)))
		{
			std::cerr << "Error: IColorFrameSource::get_FrameDescription" << std::endl;
			exit(-1);
		}
		//	OpenCV側のバッファ作成
		//	this.colorBufferを用意
		colorDescription->get_Width(&colorBufferSize.width);
		colorDescription->get_Height(&colorBufferSize.height);
		colorBuffer = cv::Mat(colorBufferSize, CV_8UC4);
		colorBufferBlockSize = colorBuffer.total() * colorBuffer.channels() * sizeof(uchar);
	}

	//--------------------------
	//	Depth Frame Open
	//--------------------------
	inline void enableDepthFrame()
	{
		//	KinectのDepthフレームを読みだすための準備
		if (FAILED(kinect->get_DepthFrameSource(&depthSource)))
		{
			std::cerr << "Error: IKinectSensor::get_DepthFrameSource" << std::endl;
			exit(-1);
		}
		if (FAILED(depthSource->OpenReader(&depthReader)))
		{
			std::cerr << "Error: IDepthFrameSource::OpenReader" << std::endl;
			exit(-1);
		}
		//	OpenCVのDepthフレームの準備
		if (FAILED(depthSource->get_FrameDescription(&depthDescription)))
		{
			std::cerr << "Error: IDepthFrameSource::get_FrameDescription" << std::endl;
			exit(-1);
		}
		//	OpenCV側のバッファ作成
		depthDescription->get_Width(&depthBufferSize.width);
		depthDescription->get_Height(&depthBufferSize.height);
		depthBuffer = cv::Mat(depthBufferSize, CV_16UC1);			//	Depthバッファは16bit
		depthBufferBlockSize = depthBuffer.total() * depthBuffer.channels() * sizeof(ushort);
	}

	//------------------------------
	//	Coordinate Mapper
	//------------------------------
	inline void enableCoordinateMapper()
	{
		if (FAILED(kinect->get_CoordinateMapper(&mapper)))
		{
			std::cerr << "Error: IKinectSensor::get_CoordinateMapper" << std::endl;
			exit(-1);
		}
	}

	//------------------------------------------------
	//	colorBufferに保存し，ミラーリングを直して渡す
	//------------------------------------------------
	inline void getColorFrame(cv::Mat &colorImg)
	{
		colorImg = cv::Mat(colorBufferSize, CV_8UC4);
		IColorFrame *colorFrame = nullptr;
		//	Colorフレーム取得
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	ColorフレームからデータをOpenCV側バッファにコピー
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				colorBufferBlockSize,
				reinterpret_cast<BYTE*>(colorBuffer.data),
				ColorImageFormat_Bgra)))
			{
				flip(colorBuffer, colorImg, 1);		//	左右反転
			}
		}
		safeRelease(colorFrame);
	}

	//------------------------------------------------
	//	depthBufferに保存し，ミラーリングを直して渡す
	//------------------------------------------------
	inline void getDepthFrame(cv::Mat &depthImg)
	{
		depthImg = cv::Mat(depthBufferSize, CV_16UC1);
		IDepthFrame *depthFrame = nullptr;
		//	Depthフレーム取得
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	DepthフレームからデータをOpenCV側バッファにコピー
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(
				&depthBufferBlockSize,
				reinterpret_cast<UINT16**>(&depthBuffer.data))))
			{
				flip(depthBuffer, depthImg, 1);		//	左右反転
			}
		}
		safeRelease(depthFrame);
	}


	virtual ~Kinect2WithOpenCVWrapper()
	{
		releaseAllInterface();
	}

	inline void releaseAllInterface()
	{
		//	全インターフェースの解放
		safeRelease(depthSource);
		safeRelease(depthReader);
		safeRelease(depthDescription);
		safeRelease(colorSource);
		safeRelease(colorReader);
		safeRelease(colorDescription);
		safeRelease(mapper);
		safeRelease(kinect);
	}
};

