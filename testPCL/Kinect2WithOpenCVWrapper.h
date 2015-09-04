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
	IColorFrame *colorFrame;
	//	for depth buffer
	IDepthFrameSource *depthSource;
	IDepthFrameReader *depthReader;
	IFrameDescription *depthDescription;
	IDepthFrame *depthFrame;
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
	cv::Mat xyzBuffer;			//	512 x 424, 32FC3
	cv::Mat coordColorBuffer;	//	512 x 424, 8UC4

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
		colorFrame = nullptr;
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
	}

	//------------------------------------------------
	//	depthBufferに保存し，ミラーリングを直して渡す
	//------------------------------------------------
	inline void getDepthFrame(cv::Mat &depthImg)
	{
		depthImg = cv::Mat(depthBufferSize, CV_16UC1);
		depthFrame = nullptr;
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
	}

	//------------------------------------------------
	//	mapperによってずれを修正したcolorImgを渡す
	//------------------------------------------------
	inline void getCoordinatedColorFrame(cv::Mat &coordColorImg)
	{
		coordColorBuffer = cv::Mat(depthBufferSize, CV_8UC4);
		if (!colorBuffer.empty() && !depthBuffer.empty())
		{
			std::vector<ColorSpacePoint> colorSpacePoints(depthBuffer.total());
			if (SUCCEEDED(mapper->MapDepthFrameToColorSpace(
				depthBuffer.total(), reinterpret_cast<UINT16*>(depthBuffer.data),
				depthBuffer.total(), &colorSpacePoints[0])))
			{
				coordColorBuffer = cv::Scalar(0);
				for (int y = 0; y < depthBuffer.rows; y++)
				{
					for (int x = 0; x < depthBuffer.cols; x++)
					{
						unsigned int idx = y * depthBuffer.cols + x;
						cv::Point colorLoc(
							static_cast<int>(floor(colorSpacePoints[idx].X + 0.5)),
							static_cast<int>(floor(colorSpacePoints[idx].Y + 0.5)));		//	四捨五入のため0.5を足す
						if (colorLoc.x >= 0 && colorLoc.x < colorBuffer.cols
							&& colorLoc.y >= 0 && colorLoc.y < colorBuffer.rows)
							coordColorBuffer.at<cv::Vec4b>(y, x) = colorBuffer.at<cv::Vec4b>(colorLoc);
					}
				}
				coordColorImg = coordColorBuffer.clone();
			}
		}
	}

	//------------------------------------------------
	//	DepthカメラからXYZ座標を復元
	//------------------------------------------------
	inline void getXYZFrame(cv::Mat &xyzMat)
	{
		xyzBuffer = cv::Mat(depthBuffer.size(), CV_32FC3);
		if (!colorBuffer.empty() && !depthBuffer.empty())
		{
			std::vector<CameraSpacePoint> xyzPoints(depthBuffer.total());
			if (SUCCEEDED(mapper->MapDepthFrameToCameraSpace(
				depthBuffer.total(), reinterpret_cast<UINT16*>(depthBuffer.data),
				depthBuffer.total(), &xyzPoints[0])))
			{
				for (int y = 0; y < depthBuffer.rows; y++)
				{
					for (int x = 0; x < depthBuffer.cols; x++)
					{
						int idx = y * depthBuffer.cols + x;
						xyzBuffer.at<cv::Vec3f>(y, x)[0] = xyzPoints[idx].X;
						xyzBuffer.at<cv::Vec3f>(y, x)[1] = xyzPoints[idx].Y;
						xyzBuffer.at<cv::Vec3f>(y, x)[2] = xyzPoints[idx].Z;
					}
				}
				xyzMat = xyzBuffer.clone();
			}
		}
	}

	virtual ~Kinect2WithOpenCVWrapper()
	{
		releaseAllInterface();
	}

	//	フレームの解放（次フレームの待機）
	inline void releaseFrames()
	{
		safeRelease(depthFrame);
		safeRelease(colorFrame);
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

