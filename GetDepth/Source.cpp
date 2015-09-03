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
	Size bufferSize;
	depthDescription->get_Width(&bufferSize.width);
	depthDescription->get_Height(&bufferSize.height);
	Mat bufferImg(bufferSize, CV_16UC1);			//	Depth�o�b�t�@��16bit
	UINT bufferBlockSize = bufferImg.rows * bufferImg.cols * bufferImg.channels() * sizeof(ushort);
	//	�\���p�摜
	Mat depthImg(bufferSize, CV_8UC1);
	namedWindow("Depth Image");
	while (1)
	{
		IDepthFrame *depthFrame = nullptr;
		//	Depth�t���[���擾
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	Depth�t���[������f�[�^��OpenCV���o�b�t�@�ɃR�s�[
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(&bufferBlockSize, reinterpret_cast<UINT16**>(&bufferImg.data))))
				bufferImg.convertTo(depthImg, CV_8U, -255.0f/8000.0f, 128.0f);		//	�߂��قǔ���
		}
		//	Depth�t���[���̉���i���̎擾���߂̑ҋ@�j
		safeRelease(depthFrame);
		//	Depth�t���[���̕`��
		imshow("Depth Image", depthImg);
		if (waitKey(1) == VK_ESCAPE) break;
	}
	//	�S�C���^�[�t�F�[�X�̉��
	safeRelease(depthSource);
	safeRelease(depthReader);
	safeRelease(depthDescription);
	safeRelease(kinect);
	cv::destroyAllWindows();

	return 0;
}