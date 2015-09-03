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
	//------------------------
	//	Color Frame
	//------------------------
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
	Size colorBufferSize;
	colorDescription->get_Width(&colorBufferSize.width);
	colorDescription->get_Height(&colorBufferSize.height);
	Mat colorBufferImg(colorBufferSize, CV_8UC4);
	UINT colorBufferBlockSize = colorBufferImg.total() * colorBufferImg.channels() * sizeof(uchar);
	//	表示用画像
	Mat colorImg(colorBufferSize/2, CV_8UC4);
	namedWindow("Color Image");
	//--------------------------
	//	Depth Frame
	//--------------------------
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
	Size depthBufferSize;
	depthDescription->get_Width(&depthBufferSize.width);
	depthDescription->get_Height(&depthBufferSize.height);
	Mat depthBufferImg(depthBufferSize, CV_16UC1);			//	Depthバッファは16bit
	UINT depthBufferBlockSize = depthBufferImg.total() * depthBufferImg.channels() * sizeof(ushort);
	//	表示用画像
	Mat depthImg(depthBufferSize, CV_8UC1);
	namedWindow("Depth Image");
	//------------------------------
	//	Coordinate Mapper
	//------------------------------
	//	ColorとDepthの位置整合のためのマッピングエンジン
	ICoordinateMapper *mapper;
	if (FAILED(kinect->get_CoordinateMapper(&mapper)))
	{
		cerr << "Error: IKinectSensor::get_CoordinateMapper" << endl;
		return 6;
	}
	Mat coordinatedColorImg(depthBufferSize, CV_8UC4);			//	depthにあわせて縮小したカラー画像
	namedWindow("Coordinate Mapping");

	while (1)
	{
		//-------------------------
		//	Color Frame
		//-------------------------
		IColorFrame *colorFrame = nullptr;
		//	Colorフレーム取得
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	ColorフレームからデータをOpenCV側バッファにコピー
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				colorBufferBlockSize, reinterpret_cast<BYTE*>(colorBufferImg.data), ColorImageFormat_Bgra)))
				resize(colorBufferImg, colorImg, Size(), 0.5, 0.5);
		}

		//-------------------------
		//	Depth Frame
		//-------------------------
		IDepthFrame *depthFrame = nullptr;
		//	Depthフレーム取得
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	DepthフレームからデータをOpenCV側バッファにコピー
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(&depthBufferBlockSize, reinterpret_cast<UINT16**>(&depthBufferImg.data))))
				depthBufferImg.convertTo(depthImg, CV_8U, -255.0f / 8000.0f, 255.0f);		//	近いほど白い
		}

		//------------------------
		//	Coordinate Mapping
		//	(DepthImage -> ColorImage)
		//------------------------
		if (colorFrame != nullptr && depthFrame != nullptr)
		{
			vector<ColorSpacePoint> colorSpacePoints(depthBufferImg.total());			//	Color画像での2D座標系(float x, float y)
			if (SUCCEEDED(mapper->MapDepthFrameToColorSpace(							//	depthBufferImgとcolorBufferImgの対応関係をcolorSpacePointsに保存
				depthBufferImg.total(), reinterpret_cast<UINT16*>(depthBufferImg.data),		//	入力データ(Depth)
				depthBufferImg.total(), &colorSpacePoints[0])))								//	出力先(Depthと同じ大きさのColorSpacePoint配列)，colorSpacePoints[dy * dwidth + dx] = (cx, cy)
			{
				coordinatedColorImg = Scalar(0, 0, 0, 0);		//	黒で塗りつぶし
				for (int y = 0; y < depthBufferImg.rows; y++)
				{
					for (int x = 0; x < depthBufferImg.cols; x++)
					{
						unsigned int idx = y * depthBufferImg.cols + x;
						unsigned short depth = matGRAY(depthBufferImg, x, y);
						Point colorLoc(
							static_cast<int>(floor(colorSpacePoints[idx].X + 0.5)),
							static_cast<int>(floor(colorSpacePoints[idx].Y + 0.5)));		//	四捨五入のため0.5を足す
						if (colorLoc.x >= 0 && colorLoc.x < colorBufferImg.cols
							&& colorLoc.y >= 0 && colorLoc.y < colorBufferImg.rows)
							coordinatedColorImg.at<Vec4b>(y, x) = colorBufferImg.at<Vec4b>(colorLoc);
					}
				}
			}
		}
		

		//	フレームの解放（次の取得命令の待機）
		safeRelease(colorFrame);
		safeRelease(depthFrame);
		//	フレームの描画
		imshow("Color Image", colorImg);
		imshow("Depth Image", depthImg);
		imshow("Coordinate Mapping", coordinatedColorImg);

		if (waitKey(1) == VK_ESCAPE) break;
	}
	//	全インターフェースの解放
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