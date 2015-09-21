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
	Size bufferSize;
	colorDescription->get_Width(&bufferSize.width);
	colorDescription->get_Height(&bufferSize.height);
	Mat bufferImg(bufferSize, CV_8UC4);
	unsigned int bufferBlockSize = bufferImg.rows * bufferImg.cols * bufferImg.channels() * sizeof(uchar);
	//	�\���p�摜
	Mat colorImg(bufferImg.rows/2, bufferImg.cols/2, CV_8UC4);
	namedWindow("Color Image");
	while (1)
	{
		IColorFrame *colorFrame = nullptr;
		//	Color�t���[���擾
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	Color�t���[������f�[�^��OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				bufferBlockSize, reinterpret_cast<BYTE*>(bufferImg.data), ColorImageFormat_Bgra)))
				resize(bufferImg, colorImg, Size(), 0.5, 0.5);
		}
		//	Color�t���[���̉���i���̎擾���߂̑ҋ@�j
		safeRelease(colorFrame);
		//	Color�t���[���̕`��
		imshow("Color Image", colorImg);
		if (waitKey(1) == VK_ESCAPE) break;
	}
	//	�S�C���^�[�t�F�[�X�̉��
	safeRelease(colorSource);
	safeRelease(colorReader);
	safeRelease(colorDescription);
	safeRelease(kinect);
	cv::destroyAllWindows();

	return 0;
}