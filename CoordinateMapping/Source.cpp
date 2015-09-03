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
	IKinectSensor *kinect;		//	Kinect�Z���T�̗p��
	//	Default��Kinect���擾
	if (FAILED(GetDefaultKinectSensor(&kinect)))
	{
		cerr << "Error: GetDefaultKinectSensor" << endl;
		return 1;
	}
	//	Kinect���J��
	if (FAILED(kinect->Open()))
	{
		cerr << "Error: IKinectSensor::Open" << endl;
		return 2;
	}
	//------------------------
	//	Color Frame
	//------------------------
	//	Kinect��Color�t���[����ǂ݂������߂̏���
	IColorFrameSource *colorSource;		//	Color�t���[�����擾���邽�߂�Source�C���^�[�t�F�[�X
	if (FAILED(kinect->get_ColorFrameSource(&colorSource)))
	{
		cerr << "Error: IKinectSensor::get_ColorFrameSource" << endl;
		return 3;
	}
	IColorFrameReader *colorReader;		//	ColorFrameSource����ColorFrame��ǂ݂������߂�Reader�C���^�[�t�F�[�X
	if (FAILED(colorSource->OpenReader(&colorReader)))
	{
		cerr << "Error: IColorFrameSource::OpenReader" << endl;
		return 4;
	}
	//	OpenCV��Color�t���[���̏���
	IFrameDescription *colorDescription;	//	Color�t���[���̏ڍ�
	if (FAILED(colorSource->get_FrameDescription(&colorDescription)))
	{
		cerr << "Error: IColorFrameSource::get_FrameDescription" << endl;
		return 5;
	}
	//	OpenCV���̃o�b�t�@�쐬
	Size colorBufferSize;
	colorDescription->get_Width(&colorBufferSize.width);
	colorDescription->get_Height(&colorBufferSize.height);
	Mat colorBufferImg(colorBufferSize, CV_8UC4);
	UINT colorBufferBlockSize = colorBufferImg.total() * colorBufferImg.channels() * sizeof(uchar);
	//	�\���p�摜
	Mat colorImg(colorBufferSize/2, CV_8UC4);
	namedWindow("Color Image");
	//--------------------------
	//	Depth Frame
	//--------------------------
	//	Kinect��Depth�t���[����ǂ݂������߂̏���
	IDepthFrameSource *depthSource;		//	Depth�t���[�����擾���邽�߂�Source�C���^�[�t�F�[�X
	if (FAILED(kinect->get_DepthFrameSource(&depthSource)))
	{
		cerr << "Error: IKinectSensor::get_DepthFrameSource" << endl;
		return 3;
	}
	IDepthFrameReader *depthReader;		//	DepthFrameSource����DepthFrame��ǂ݂������߂�Reader�C���^�[�t�F�[�X
	if (FAILED(depthSource->OpenReader(&depthReader)))
	{
		cerr << "Error: IDepthFrameSource::OpenReader" << endl;
		return 4;
	}
	//	OpenCV��Depth�t���[���̏���
	IFrameDescription *depthDescription;	//	Depth�t���[���̏ڍ�
	if (FAILED(depthSource->get_FrameDescription(&depthDescription)))
	{
		cerr << "Error: IDepthFrameSource::get_FrameDescription" << endl;
		return 5;
	}
	//	OpenCV���̃o�b�t�@�쐬
	Size depthBufferSize;
	depthDescription->get_Width(&depthBufferSize.width);
	depthDescription->get_Height(&depthBufferSize.height);
	Mat depthBufferImg(depthBufferSize, CV_16UC1);			//	Depth�o�b�t�@��16bit
	UINT depthBufferBlockSize = depthBufferImg.total() * depthBufferImg.channels() * sizeof(ushort);
	//	�\���p�摜
	Mat depthImg(depthBufferSize, CV_8UC1);
	namedWindow("Depth Image");
	//------------------------------
	//	Coordinate Mapper
	//------------------------------
	//	Color��Depth�̈ʒu�����̂��߂̃}�b�s���O�G���W��
	ICoordinateMapper *mapper;
	if (FAILED(kinect->get_CoordinateMapper(&mapper)))
	{
		cerr << "Error: IKinectSensor::get_CoordinateMapper" << endl;
		return 6;
	}
	Mat coordinatedColorImg(depthBufferSize, CV_8UC4);			//	depth�ɂ��킹�ďk�������J���[�摜
	namedWindow("Coordinate Mapping");

	while (1)
	{
		//-------------------------
		//	Color Frame
		//-------------------------
		IColorFrame *colorFrame = nullptr;
		//	Color�t���[���擾
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	Color�t���[������f�[�^��OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				colorBufferBlockSize, reinterpret_cast<BYTE*>(colorBufferImg.data), ColorImageFormat_Bgra)))
				resize(colorBufferImg, colorImg, Size(), 0.5, 0.5);
		}

		//-------------------------
		//	Depth Frame
		//-------------------------
		IDepthFrame *depthFrame = nullptr;
		//	Depth�t���[���擾
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	Depth�t���[������f�[�^��OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(&depthBufferBlockSize, reinterpret_cast<UINT16**>(&depthBufferImg.data))))
				depthBufferImg.convertTo(depthImg, CV_8U, -255.0f / 8000.0f, 255.0f);		//	�߂��قǔ���
		}

		//------------------------
		//	Coordinate Mapping
		//	(DepthImage -> ColorImage)
		//------------------------
		if (colorFrame != nullptr && depthFrame != nullptr)
		{
			vector<ColorSpacePoint> colorSpacePoints(depthBufferImg.total());			//	Color�摜�ł�2D���W�n(float x, float y)
			if (SUCCEEDED(mapper->MapDepthFrameToColorSpace(							//	depthBufferImg��colorBufferImg�̑Ή��֌W��colorSpacePoints�ɕۑ�
				depthBufferImg.total(), reinterpret_cast<UINT16*>(depthBufferImg.data),		//	���̓f�[�^(Depth)
				depthBufferImg.total(), &colorSpacePoints[0])))								//	�o�͐�(Depth�Ɠ����傫����ColorSpacePoint�z��)�CcolorSpacePoints[dy * dwidth + dx] = (cx, cy)
			{
				coordinatedColorImg = Scalar(0, 0, 0, 0);		//	���œh��Ԃ�
				for (int y = 0; y < depthBufferImg.rows; y++)
				{
					for (int x = 0; x < depthBufferImg.cols; x++)
					{
						unsigned int idx = y * depthBufferImg.cols + x;
						unsigned short depth = matGRAY(depthBufferImg, x, y);
						Point colorLoc(
							static_cast<int>(floor(colorSpacePoints[idx].X + 0.5)),
							static_cast<int>(floor(colorSpacePoints[idx].Y + 0.5)));		//	�l�̌ܓ��̂���0.5�𑫂�
						if (colorLoc.x >= 0 && colorLoc.x < colorBufferImg.cols
							&& colorLoc.y >= 0 && colorLoc.y < colorBufferImg.rows)
							coordinatedColorImg.at<Vec4b>(y, x) = colorBufferImg.at<Vec4b>(colorLoc);
					}
				}
			}
		}
		

		//	�t���[���̉���i���̎擾���߂̑ҋ@�j
		safeRelease(colorFrame);
		safeRelease(depthFrame);
		//	�t���[���̕`��
		imshow("Color Image", colorImg);
		imshow("Depth Image", depthImg);
		imshow("Coordinate Mapping", coordinatedColorImg);

		if (waitKey(1) == VK_ESCAPE) break;
	}
	//	�S�C���^�[�t�F�[�X�̉��
	safeRelease(depthSource);
	safeRelease(depthReader);
	safeRelease(depthDescription);
	safeRelease(colorSource);
	safeRelease(colorReader);
	safeRelease(colorDescription);
	safeRelease(mapper);
	safeRelease(kinect);
	cv::destroyAllWindows();

	return 0;
}