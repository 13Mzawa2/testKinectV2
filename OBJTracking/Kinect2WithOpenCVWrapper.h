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
	cv::Mat colorBuffer;		//	1920 x 1080, BGRA, 8UC4
	cv::Mat depthBuffer;		//	512 x 424, Gray, 16UC1
	cv::Mat xyzBuffer;			//	512 x 424, XYZ, 32FC3
	cv::Mat coordColorBuffer;	//	512 x 424, BGRA, 8UC4

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
		colorImg = cv::Mat(colorBufferSize, CV_8UC3);	//	�n������ 3 channels �ɕύX
		getColorFrame();
		cv::cvtColor(colorBuffer, colorImg, CV_BGRA2BGR);
		flip(colorImg, colorImg, 1);		//	���E���]
	}
	inline void getColorFrame(void)
	{
		colorFrame = nullptr;
		cv::Mat buffer(colorBufferSize, CV_8UC4);
		//	Color�t���[���擾
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	Color�t���[������f�[�^���擾�ł�����OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				colorBufferBlockSize,
				reinterpret_cast<BYTE*>(buffer.data),
				ColorImageFormat_Bgra)))
			{
				buffer.copyTo(colorBuffer);
			}
		}
	}

	//------------------------------------------------
	//	depthBuffer�ɕۑ����C�~���[�����O�𒼂��ēn��
	//------------------------------------------------
	inline void getDepthFrame(cv::Mat &depthImg)
	{
		depthImg = cv::Mat(depthBufferSize, CV_16UC1);
		getDepthFrame();
		flip(depthBuffer, depthImg, 1);		//	���E���]
	}
	inline void getDepthFrame(void)
	{
		depthFrame = nullptr;
		cv::Mat buffer(depthBufferSize, CV_16UC1);
		//	Depth�t���[���擾
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	Depth�t���[������f�[�^��OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(
				&depthBufferBlockSize,
				reinterpret_cast<UINT16**>(&buffer.data))))
			{
				buffer.copyTo(depthBuffer);
			}
		}
	}

	//------------------------------------------------
	//	mapper�ɂ���Ă�����C������colorImg��n��
	//------------------------------------------------
	inline void getCoordinatedColorFrame(cv::Mat &coordColorImg)
	{
		coordColorImg = cv::Mat(depthBufferSize, CV_8UC3);
		getCoordinatedColorFrame();
		cv::cvtColor(coordColorBuffer, coordColorImg, CV_BGRA2BGR);
	}
	inline void getCoordinatedColorFrame(void)
	{
		cv::Mat buffer = cv::Mat(depthBufferSize, CV_8UC4);
		if (!colorBuffer.empty() && !depthBuffer.empty())
		{
			std::vector<ColorSpacePoint> colorSpacePoints(depthBuffer.total());
			if (SUCCEEDED(mapper->MapDepthFrameToColorSpace(
				depthBuffer.total(), reinterpret_cast<UINT16*>(depthBuffer.data),		//	�擾�����f�v�X�o�b�t�@
				depthBuffer.total(), &colorSpacePoints[0])))							//	depth -> color�ϊ�LUT
			{
				buffer = cv::Scalar(0);
#pragma omp parallel for
				for (int y = 0; y < depthBuffer.rows; y++)
				{
#pragma omp parallel for
					for (int x = 0; x < depthBuffer.cols; x++)
					{
						unsigned int idx = y * depthBuffer.cols + x;
						cv::Point colorLoc(
							static_cast<int>(floor(colorSpacePoints[idx].X + 0.5)),
							static_cast<int>(floor(colorSpacePoints[idx].Y + 0.5)));		//	�l�̌ܓ��̂���0.5�𑫂�
						if (colorLoc.x >= 0 && colorLoc.x < colorBuffer.cols
							&& colorLoc.y >= 0 && colorLoc.y < colorBuffer.rows)
							buffer.at<cv::Vec4b>(idx) = colorBuffer.at<cv::Vec4b>(colorLoc);	//	LUT�ɏ]���o�b�t�@�ɕۑ�
					}
				}
				
				buffer.copyTo(coordColorBuffer);
			}
		}
	}

	//------------------------------------------------
	//	Depth�J��������XYZ���W�𕜌�
	//------------------------------------------------
	inline void getXYZFrame(cv::Mat &xyzMat)
	{
		xyzMat = cv::Mat(depthBufferSize, CV_32FC3);		//	(x, y, z) float���x
		getXYZFrame();
		xyzBuffer.copyTo(xyzMat);
	}
	inline void getXYZFrame(void)
	{
		cv::Mat buffer = cv::Mat(depthBuffer.size(), CV_32FC3);
		if (!depthBuffer.empty())
		{
			std::vector<CameraSpacePoint> xyzPoints(depthBuffer.total());
			if (SUCCEEDED(mapper->MapDepthFrameToCameraSpace(
				depthBuffer.total(), reinterpret_cast<UINT16*>(depthBuffer.data),
				depthBuffer.total(), &xyzPoints[0])))
			{
#pragma omp parallel for
				for (int y = 0; y < depthBuffer.rows; y++)
				{
					for (int x = 0; x < depthBuffer.cols; x++)
					{
						int idx = y * depthBuffer.cols + x;
						buffer.at<cv::Vec3f>(idx)[0] = xyzPoints[idx].X;
						buffer.at<cv::Vec3f>(idx)[1] = xyzPoints[idx].Y;
						buffer.at<cv::Vec3f>(idx)[2] = xyzPoints[idx].Z;
					}
				}
				buffer.copyTo(xyzBuffer);
			}
		}
	}

	virtual ~Kinect2WithOpenCVWrapper()
	{
		releaseAllInterface();
	}

	//	�t���[���̉���i���t���[���̑ҋ@�j
	inline void releaseFrames()
	{
		safeRelease(depthFrame);
		safeRelease(colorFrame);
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

