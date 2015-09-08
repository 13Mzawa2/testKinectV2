
#include "PCLAdapter.h"
#include "OpenCV3Linker.h"
#include "Kinect2WithOpenCVWrapper.h"
#include "ColorTable.h"

//using namespace cv;

const char filename[] = "model/miku.obj";

int main(void)
{
	Kinect2WithOpenCVWrapper kinect;
	kinect.enableColorFrame();
	kinect.enableDepthFrame();
	kinect.enableCoordinateMapper();

	ColorTable table;
	table.generate16bitPalette();
	cv::imshow("Color Palette", table.miniColorTable);

	pcl::visualization::CloudViewer viewer("Point Cloud");
	
	//	OBJ�t�@�C����ǂݍ���
	pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_pcd(new pcl::PointCloud<pcl::PointXYZ>());
	if (pcl::io::loadPolygonFileOBJ(filename, *mesh) != -1)
	{	//	PolygonMesh -> PointCloud<PointXYZ>
		pcl::fromPCLPointCloud2(mesh->cloud, *obj_pcd);
		//vtkSmartPointer<vtkPolyData> vtkmesh;
		//pcl::io::mesh2vtk(*mesh, vtkmesh);
		//pcl::io::vtkPolyDataToPointCloud(vtkmesh, *obj_pcd);
	}
	viewer.showCloud(obj_pcd);

	while (1)
	{
		cv::Mat colorImg, depthMat, depthMatd, depthImg, depthError;
		kinect.getColorFrame(colorImg);
		kinect.getDepthFrame(depthMat);
		//	����G���[�_���}�X�N�摜��
		depthError = cv::Mat(depthMat.size(), CV_8UC1);
		depthMat.convertTo(depthMatd, CV_32F);
		cv::threshold(depthMatd, depthError, 1, 255, cv::THRESH_BINARY);
		//	�f�v�X�}�b�v���J���[�摜��
		table.lookup16UC1to8UC3(depthMat, depthImg);
		//	Point Cloud
		cv::Mat coordinatedColorImg, xyzMat;
		kinect.getXYZFrame(xyzMat);
		kinect.getCoordinatedColorFrame(coordinatedColorImg);
		kinect.releaseFrames();
		// Create Point Cloud
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
		pointcloud->width = static_cast<uint32_t>(depthMat.cols);
		pointcloud->height = static_cast<uint32_t>(depthMat.rows);
		pointcloud->is_dense = false;
		for (int y = 0; y < depthMat.rows; y++)
		{
			for (int x = 0; x < depthMat.cols; x++)
			{
				pcl::PointXYZRGB point;
				point.x = matBf(xyzMat, x, y);
				point.y = matGf(xyzMat, x, y);
				point.z = matRf(xyzMat, x, y);
				point.b = matB(coordinatedColorImg, x, y);
				point.g = matG(coordinatedColorImg, x, y);
				point.r = matR(coordinatedColorImg, x, y);
				pointcloud->points.push_back(point);
			}
		}
		//	Point Cloud�̕`��
		//viewer.showCloud(pointcloud);

		//	�t���[���̕`��
		resize(colorImg, colorImg, cv::Size(), 0.5, 0.5);
		cv::imshow("Color Image", colorImg);
		cv::imshow("Depth Image", depthImg);
		cv::imshow("Depth Error", depthError);

		char c = cv::waitKey(1);
		if (c == 'q' || c == VK_ESCAPE) break;
	}
	kinect.releaseAllInterface();
}