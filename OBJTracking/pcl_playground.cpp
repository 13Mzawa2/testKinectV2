#include <omp.h>			//	OPNEMP
#include "PCLAdapter.h"
#include "OpenCV3Linker.h"
#include "Kinect2WithOpenCVWrapper.h"
#include "ColorTable.h"

//using namespace cv;

const char filename[] = "model/miku.obj";

template <typename PointT>
inline void removePlane(boost::shared_ptr<pcl::PointCloud<PointT>> &src, boost::shared_ptr<pcl::PointCloud<PointT>> &dst, float thresh);

int main(void)
{
	Kinect2WithOpenCVWrapper kinect;
	kinect.enableColorFrame();
	kinect.enableDepthFrame();
	kinect.enableCoordinateMapper();

	ColorTable table;
	table.generate16bitPalette();
	//cv::imshow("Color Palette", table.miniColorTable);

	pcl::visualization::CloudViewer viewer("Point Cloud");
	
	////	OBJ�t�@�C����ǂݍ���
	//pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
	//pcl::PointCloud<pcl::PointXYZ>::Ptr obj_pcd(new pcl::PointCloud<pcl::PointXYZ>());
	//if (pcl::io::loadPolygonFileOBJ(filename, *mesh) != -1)
	//{	//	PolygonMesh -> PointCloud<PointXYZ>
	//	pcl::fromPCLPointCloud2(mesh->cloud, *obj_pcd);
	//	//vtkSmartPointer<vtkPolyData> vtkmesh;
	//	//pcl::io::mesh2vtk(*mesh, vtkmesh);
	//	//pcl::io::vtkPolyDataToPointCloud(vtkmesh, *obj_pcd);
	//}
	//viewer.showCloud(obj_pcd);

	while (1)
	{
		cv::Mat colorImg, depthMat, depthMatd, depthImg, depthError;
		kinect.getColorFrame(colorImg);
		kinect.getDepthFrame(depthMat);
		////	����G���[�_���}�X�N�摜��
		//depthError = cv::Mat(depthMat.size(), CV_8UC1);
		//depthMat.convertTo(depthMatd, CV_32F);
		//cv::threshold(depthMatd, depthError, 1, 255, cv::THRESH_BINARY);
		//	�f�v�X�}�b�v���J���[�摜��
		table.lookup16UC1to8UC3(depthMat, depthImg);
		//	Point Cloud
		cv::Mat coordinatedColorImg, xyzMat;
		kinect.getXYZFrame(xyzMat);
		//kinect.getCoordinatedColorFrame(coordinatedColorImg);
		kinect.releaseFrames();
		// Create Point Cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
		pointcloud->width = static_cast<uint32_t>(depthMat.cols);
		pointcloud->height = static_cast<uint32_t>(depthMat.rows);
		pointcloud->is_dense = false;
#ifdef _OPENMP
#pragma omp parallel
#endif
		{
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1000)
#endif
			for (int y = 0; y < depthMat.rows; y++)
			{
				for (int x = 0; x < depthMat.cols; x++)
				{
					pcl::PointXYZ point;
					point.x = matBf(xyzMat, x, y);
					point.y = matGf(xyzMat, x, y);
					point.z = matRf(xyzMat, x, y);
					//point.b = matB(coordinatedColorImg, x, y);
					//point.g = matG(coordinatedColorImg, x, y);
					//point.r = matR(coordinatedColorImg, x, y);
					pointcloud->points.push_back(point);
				}
			}
		}
		//	�_�Q�̌y�ʉ��i�_�E���T���v�����O�j
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::ApproximateVoxelGrid <pcl::PointXYZ> avg;
		avg.setInputCloud(pointcloud);
		avg.setLeafSize(0.001f, 0.001f, 0.001f);	//	VoxelGrid�̑傫����X,Y,Z�Ŏw�� �����ł�1mm�ɂ���
		avg.filter(*cloud_filtered);

		//	���ʏ���
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented(new pcl::PointCloud<pcl::PointXYZ>());
		removePlane(cloud_filtered, cloud_segmented, 0.03);
		removePlane(cloud_segmented, cloud_segmented, 0.04);
		removePlane(cloud_segmented, cloud_segmented, 0.05);

		////	�O��l�̏���(Statistical)
		////	�ߖTn�_�Ƃ̋������W���΍���k�{�ȏゾ�����珜��
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_removed(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
		//sor.setInputCloud(cloud_segmented);
		//sor.setMeanK(20);				//	���ׂ�ߖT�_�̌�
		//sor.setStddevMulThresh(1.0);	//	�W���΍��̉��{�܂ŋ��e���邩
		////sor.setNegative(true);			//	true�Ȃ�O��l�̕��̂ݎc��
		//sor.filter(*cloud_removed);

		//	Harris�����_���o��
		pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;
		detector.setNonMaxSupression(true);
		detector.setRadius(0.01);				//	Harris�����_�̌v�Z���a(1cm)
		detector.setInputCloud(cloud_segmented);
		pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
		detector.compute(*keypoints);

		////	�N���X�^�����O
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZ>());
		//if (cloud_segmented->is_dense)
		//{
		//	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		//	tree->setInputCloud(cloud_segmented);
		//	std::vector<pcl::PointIndices> cluster_indeces;
		//	pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
		//	clustering.setClusterTolerance(0.03);
		//	clustering.setMinClusterSize(200);
		//	clustering.setMaxClusterSize(100000);
		//	clustering.setSearchMethod(tree);
		//	clustering.setInputCloud(cloud_segmented);
		//	clustering.extract(cluster_indeces);

		//	pcl::ExtractIndices<pcl::PointXYZ> extract;
		//	extract.setInputCloud(cloud_segmented);
		//	pcl::IndicesPtr indices(new std::vector<int>);
		//	*indices = cluster_indeces[0].indices;
		//	extract.setIndices(indices);
		//	extract.setNegative(false);
		//	extract.filter(*cloud_clustered);
		//}


			////	�O��l�̏���(Radius)
			////	���ar�̋ߖT������n�ȏ�_�������Ă��Ȃ���Ώ���
			////	������Ȃ�ł��x����
			//pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
			//outrem.setInputCloud(pointcloud);
			//outrem.setRadiusSearch(0.1);		//	���a0.8�ߖT
			//outrem.setMinNeighborsInRadius(3);	//	���a���ɓ����Ă���ׂ��_�̍Œ��
			////outrem.setNegative(true);
			//outrem.filter(*cloud_filtered);

			////	�S�Ă̓_�ɂ����Ė@���̐���
			//pcl::PointCloud<pcl::PointNormal>::Ptr normal(new pcl::PointCloud<pcl::PointNormal>());
			//pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
			//ne.setInputCloud(cloud_segmented);
			//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());		//	�T�����@�FKdTree
			//ne.setSearchMethod(tree);
			//ne.setRadiusSearch(0.03);		//	�T�����a
			//ne.compute(*normal);

			////	FPFH�����ʂ̌v�Z
			//pcl::PointCloud<pcl::FPFHSignature33>::Ptr features;
			//pcl::FPFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
			//fpfh.setInputCloud(cloud_removed);
			//fpfh.setInputNormals(normal);
			//fpfh.setSearchMethod(tree);
			//fpfh.setRadiusSearch(0.05);		//	�T�����a�@�@������̎��̒T�����a���傫������K�v������
			//fpfh.compute(*features);

			//	Point Cloud�̕`��
			//viewer.showCloud(pointcloud);
			//viewer.showCloud(cloud_filtered);
			//viewer.showCloud(cloud_segmented);
			//viewer.showCloud(cloud_removed);
		if (keypoints->is_dense)
			viewer.showCloud(keypoints);

		//	�t���[���̕`��
		resize(colorImg, colorImg, cv::Size(), 0.5, 0.5);
		cv::imshow("Color Image", colorImg);
		cv::imshow("Depth Image", depthImg);
		//cv::imshow("Depth Error", depthError);

		char c = cv::waitKey(1);
		if (c == 'q' || c == VK_ESCAPE) break;
	}
	kinect.releaseAllInterface();
}

template <typename PointT>
inline void removePlane(boost::shared_ptr<pcl::PointCloud<PointT>> &src, boost::shared_ptr<pcl::PointCloud<PointT>> &dst, float thresh)
{
	//	���ʂ̌��o
	pcl::SACSegmentation<PointT> seg;
	seg.setInputCloud(src);
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);		//	���ʂ����o
	seg.setMethodType(pcl::SAC_RANSAC);			//	RANSAC�Ō�Ή��_����
	seg.setDistanceThreshold(thresh);				//	���̕ϓ����܂ŋ���
	pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients());	//	���肳�ꂽ���f���� ax+by+cz+d=0
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());			//	���ʓ_�Q��Index
	seg.segment(*inliers, *coeffs);
	//	���ʂ̏���
	pcl::ExtractIndices<PointT>::Ptr extract(new pcl::ExtractIndices<PointT>());
	extract->setInputCloud(src);
	extract->setIndices(inliers);			//	���ʓ_�Q��Index
	extract->setNegative(true);			//	���ʏ���
	extract->filter(*dst);
}