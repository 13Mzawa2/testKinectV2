#pragma once
#include <opencv2\opencv.hpp>

class ColorTable
{
protected:
	cv::Mat colorTable;

public:
	cv::Mat miniColorTable;

	inline void lookup16UC1to8UC3(cv::Mat &src, cv::Mat &dst)
	{
		dst = cv::Mat(src.size(), CV_8UC3);
		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				dst.at<cv::Vec3b>(j, i)[0] = colorTable.at<cv::Vec3b>(src.at<UINT16>(j, i), 0)[0];
				dst.at<cv::Vec3b>(j, i)[1] = colorTable.at<cv::Vec3b>(src.at<UINT16>(j, i), 0)[1];
				dst.at<cv::Vec3b>(j, i)[2] = colorTable.at<cv::Vec3b>(src.at<UINT16>(j, i), 0)[2];
			}
		}
	}

	//	デプスカメラ用
	inline void generate16bitPalette()
	{
		std::vector<cv::Vec3b> baseColors;
		baseColors.push_back(cv::Vec3b(255, 255, 255));		//	white
		baseColors.push_back(cv::Vec3b(0, 0, 255));		//	red
		baseColors.push_back(cv::Vec3b(0, 255, 255));		//	yellow
		baseColors.push_back(cv::Vec3b(0, 128, 0));		//	green
		baseColors.push_back(cv::Vec3b(128, 0, 0));		//	navy
		baseColors.push_back(cv::Vec3b(255, 0, 0));		//	blue
		baseColors.push_back(cv::Vec3b(255, 255, 0));	//	cyan
		baseColors.push_back(cv::Vec3b(0, 255, 0));		//	lime
		int anchor[] = {0, 300, 600, 900, 1200, 1500, 2400, 65536};
		generateColorTable(65536, baseColors, anchor);
		resize(colorTable, miniColorTable, cv::Size(80, 320));
	}

	//	指定基本色baseColors, 色のアンカーanchor[]に基づきグラデーションを生成して，総数nColorsのLUTを作成する
	inline void generateColorTable(int nColors, std::vector<cv::Vec3b> baseColors, int anchor[])
	{
		int colorNum = baseColors.size();

		//	LUT作成
		colorTable = cv::Mat(nColors, 1, CV_8UC3);
		static int idx = 0;
		for (int i = 1; i < colorNum; i++)
		{
			int width = anchor[i] - anchor[i - 1];
			for (int j = 0; j < width; j++)
			{
				colorTable.at<cv::Vec3b>(idx, 0)[0] = (baseColors[i - 1][0] * (width - j) + baseColors[i][0] * j) / width;		//	線形補間
				colorTable.at<cv::Vec3b>(idx, 0)[1] = (baseColors[i - 1][1] * (width - j) + baseColors[i][1] * j) / width;		//	線形補間
				colorTable.at<cv::Vec3b>(idx, 0)[2] = (baseColors[i - 1][2] * (width - j) + baseColors[i][2] * j) / width;		//	線形補間
				idx++;
			}
		}
	}

	ColorTable()
	{
	}

	virtual ~ColorTable()
	{
	}
};

