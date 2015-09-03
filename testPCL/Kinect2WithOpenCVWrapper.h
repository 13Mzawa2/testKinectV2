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
		//	Default��Kinect���擾
		if (FAILED(GetDefaultKinectSensor(&kinect)))
		{
			std::cerr << "Error: GetDefaultKinectSensor" << std::endl;
			exit(-1);
		}
		//	Kinect���J��
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
		//	Kinect��Color�t���[����ǂ݂������߂̏���
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
		//	OpenCV��Color�t���[���̏���
		if (FAILED(colorSource->get_FrameDescription(&colorDescription)))
		{
			std::cerr << "Error: IColorFrameSource::get_FrameDescription" << std::endl;
			exit(-1);
		}
		//	OpenCV���̃o�b�t�@�쐬
		//	this.colorBuffer��p��
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
		//	Kinect��Depth�t���[����ǂ݂������߂̏���
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
		//	OpenCV��Depth�t���[���̏���
		if (FAILED(depthSource->get_FrameDescription(&depthDescription)))
		{
			std::cerr << "Error: IDepthFrameSource::get_FrameDescription" << std::endl;
			exit(-1);
		}
		//	OpenCV���̃o�b�t�@�쐬
		depthDescription->get_Width(&depthBufferSize.width);
		depthDescription->get_Height(&depthBufferSize.height);
		depthBuffer = cv::Mat(depthBufferSize, CV_16UC1);			//	Depth�o�b�t�@��16bit
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
	//	colorBuffer�ɕۑ����C�~���[�����O�𒼂��ēn��
	//------------------------------------------------
	inline void getColorFrame(cv::Mat &colorImg)
	{
		colorImg = cv::Mat(colorBufferSize, CV_8UC4);
		IColorFrame *colorFrame = nullptr;
		//	Color�t���[���擾
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	Color�t���[������f�[�^��OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				colorBufferBlockSize,
				reinterpret_cast<BYTE*>(colorBuffer.data),
				ColorImageFormat_Bgra)))
			{
				flip(colorBuffer, colorImg, 1);		//	���E���]
			}
		}
		safeRelease(colorFrame);
	}

	//------------------------------------------------
	//	depthBuffer�ɕۑ����C�~���[�����O�𒼂��ēn��
	//------------------------------------------------
	inline void getDepthFrame(cv::Mat &depthImg)
	{
		depthImg = cv::Mat(depthBufferSize, CV_16UC1);
		IDepthFrame *depthFrame = nullptr;
		//	Depth�t���[���擾
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	Depth�t���[������f�[�^��OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(
				&depthBufferBlockSize,
				reinterpret_cast<UINT16**>(&depthBuffer.data))))
			{
				flip(depthBuffer, depthImg, 1);		//	���E���]
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
		//	�S�C���^�[�t�F�[�X�̉��
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

