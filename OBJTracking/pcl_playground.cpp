#include <omp.h>			//	OPNEMP
#include "PCLAdapter.h"
#include "OpenCV3Linker.h"
#include "Kinect2WithOpenCVWrapper.h"
#include "ColorTable.h"

//using namespace cv;

const char filename[] = "model/drop_modified_x004.obj";

template <typename PointT>
inline void removePlane(boost::shared_ptr<pcl::PointCloud<PointT>> src, boost::shared_ptr<pcl::PointCloud<PointT>> dst, float thresh);
template <typename PointT, typename PointNT>
inline void addNormal(boost::shared_ptr<pcl::PointCloud<PointT>> cloud, boost::shared_ptr<pcl::PointCloud<PointNT>> normals, float radius);
template <typename PointInT, typename PointKT>
inline void getHarrisKeypoints(boost::shared_ptr<pcl::PointCloud<PointInT>> cloud_normals, boost::shared_ptr<pcl::PointCloud<PointKT>> keypoints, float radius);
inline void getFPFHSignatureAtKeypoints(
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints,
	pcl::PointCloud<pcl::Normal>::Ptr normals,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features,
	float radius);

int main(void)
{
	Kinect2WithOpenCVWrapper kinect;
	kinect.enableColorFrame();
	kinect.enableDepthFrame();
	kinect.enableCoordinateMapper();

	ColorTable table;
	table.generate16bitPalette();
	//cv::imshow("Color Palette", table.miniColorTable);

	//pcl::visualization::CloudViewer viewer("Point Cloud");
	pcl::visualization::PCLVisualizer viewer("Point Cloud");

	//	OBJ�t�@�C����ǂݍ���
	pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_pcd(new pcl::PointCloud<pcl::PointXYZ>());				//	obj�t�@�C���̓_�Q�f�[�^
	if (pcl::io::loadPolygonFileOBJ(filename, *mesh) == -1)
	{
		return -1;
	}
	//	PolygonMesh -> PointCloud<PointXYZ>
	pcl::fromPCLPointCloud2(mesh->cloud, *obj_pcd);
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	//	�P�ʕϊ� mm -> m
	for (int i = 0; i < obj_pcd->height*obj_pcd->width; i++)
	{
		pcl::PointXYZ point;
		point.x = obj_pcd->points[i].x / 1000.0f;
		point.y = obj_pcd->points[i].y / 1000.0f;
		point.z = obj_pcd->points[i].z / 1000.0f;
		obj_cloud->push_back(point);
	}

	//------------------------------------------
	//	OBJ�f�[�^��������ʂ��v�Z
	//------------------------------------------

	//	�_�Q�̌y�ʉ��i�_�E���T���v�����O�j
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_filtered(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::ApproximateVoxelGrid <pcl::PointXYZ> avg;
	avg.setInputCloud(obj_cloud);
	avg.setLeafSize(0.001f, 0.001f, 0.001f);	//	VoxelGrid�̑傫����X,Y,Z�Ŏw�� �����ł�1mm�ɂ���
	avg.filter(*obj_filtered);

	//	�S�Ă̓_�̖@���̐���
	//	PointXYZ����Normal�y��PointNormal�𐶐�
	pcl::PointCloud<pcl::Normal>::Ptr obj_normals(new pcl::PointCloud<pcl::Normal>());		//	�@���̂�
	pcl::PointCloud<pcl::PointNormal>::Ptr obj_cloud_normals(new pcl::PointCloud<pcl::PointNormal>());	//	3�����_�Q + ���肳�ꂽ�@��
	addNormal(obj_filtered, obj_normals, 0.01f);
	pcl::concatenateFields(*obj_filtered, *obj_normals, *obj_cloud_normals);

	//	Harris�����_���o��
	//	PointNormal����PointXYZI�𐶐�
	pcl::PointCloud<pcl::PointXYZI>::Ptr obj_keypoints(new pcl::PointCloud<pcl::PointXYZI>());		//	Harris�����_�o�͌���
	getHarrisKeypoints(obj_cloud_normals, obj_keypoints, 0.01f);
	//	Harris�����_��pcl::PointCloud<pcl::PointXYZ>�ɃR�s�[
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_kpts(new pcl::PointCloud<pcl::PointXYZ>());
	obj_kpts->points.resize(obj_keypoints->points.size());
	pcl::copyPointCloud(*obj_keypoints, *obj_kpts);

	//	�����_�����FPFH�����ʂ̌v�Z
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr obj_features(new pcl::PointCloud<pcl::FPFHSignature33>());		//	FPFH������
	getFPFHSignatureAtKeypoints(obj_filtered, obj_kpts, obj_normals, obj_features, 0.04f);
	//viewer.showCloud(obj_cloud);
	

	Sleep(4000);		//	Kinect�̋N����҂�

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

		//	�t���[���̕`��
		resize(colorImg, colorImg, cv::Size(), 0.5, 0.5);
		cv::imshow("Color Image", colorImg);
		cv::imshow("Depth Image", depthImg);
		//cv::imshow("Depth Error", depthError);

		//	�L�[���͎�t
		char c = cv::waitKey(1);
		if (c == 'q' || c == VK_ESCAPE) break;
		
		// Create Point Cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
		pointcloud->width = static_cast<uint32_t>(depthMat.cols);
		pointcloud->height = static_cast<uint32_t>(depthMat.rows);
		pointcloud->is_dense = false;
#pragma omp parallel
		{
#pragma omp for schedule(dynamic, 1000)
			for (int y = 0; y < depthMat.rows; y++)
			{
				for (int x = 0; x < depthMat.cols; x++)
				{	//	PCL�ɓn������mm�P�ʂɑ�����
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

		//-----------------------------------------------------
		//	Kinect�f�[�^����ΏۂƂȂ镔�������𒊏o
		//-----------------------------------------------------

		////	NaN�_�̏���
		//std::vector<int> indices;
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::removeNaNFromPointCloud(*pointcloud, *cloud, indices);

		//	�_�Q�̌y�ʉ��i�_�E���T���v�����O�j
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::ApproximateVoxelGrid <pcl::PointXYZ> avg;
		avg.setInputCloud(pointcloud);
		//avg.setLeafSize(0.001f, 0.001f, 0.001f);	//	VoxelGrid�̑傫����X,Y,Z�Ŏw�� �����ł�1mm�ɂ���
		avg.filter(*cloud_filtered);

		//	���ʏ���
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented(new pcl::PointCloud<pcl::PointXYZ>());
		removePlane(cloud_filtered, cloud_segmented, 0.02f);
		removePlane(cloud_segmented, cloud_segmented, 0.03f);
		//removePlane(cloud_segmented, cloud_segmented, 0.04f);

		////	�O��l�̏���(Statistical)
		////	�ߖTn�_�Ƃ̋������W���΍���k�{�ȏゾ�����珜��
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_removed(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
		//sor.setInputCloud(cloud_segmented);
		//sor.setMeanK(50);				//	���ׂ�ߖT�_�̌�
		//sor.setStddevMulThresh(1.0);	//	�W���΍��̉��{�܂ŋ��e���邩
		////sor.setNegative(true);			//	true�Ȃ�O��l�̕��̂ݎc��
		//sor.filter(*cloud_removed);


		////	�N���X�^�����O
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		//std::vector<pcl::PointIndices> cluster_indeces;
		//pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
		//clustering.setClusterTolerance(0.02);
		//clustering.setMinClusterSize(1000);
		//clustering.setMaxClusterSize(3400);
		//clustering.setSearchMethod(tree);
		//clustering.setInputCloud(cloud_segmented);
		//clustering.extract(cluster_indeces);

		//if (cluster_indeces.size() > 0)
		//{
		//	cout << "clusters: " << cluster_indeces.size() << ", Max cluster size: " << cluster_indeces[0].indices.size() << "\n";
		//	pcl::ExtractIndices<pcl::PointXYZ> extract;
		//	extract.setInputCloud(cloud_segmented);
		//	pcl::IndicesPtr indices(new std::vector<int>);
		//	*indices = cluster_indeces[0].indices;
		//	extract.setIndices(indices);
		//	extract.setNegative(false);
		//	extract.filter(*cloud_clustered);

			//------------------------------------------
			//	Kinect�f�[�^��������ʂ��v�Z
			//------------------------------------------

			//	�S�Ă̓_�̖@���̐���
			//	PointXYZ����Normal�y��PointNormal�𐶐�
			pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());		//	�@���̂�
			pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>());	//	3�����_�Q + ���肳�ꂽ�@��
			addNormal(cloud_segmented, normals, 0.01f);
			pcl::concatenateFields(*cloud_segmented, *normals, *cloud_normals);

			//	Harris�����_���o��
			//	PointNormal����PointXYZI�𐶐�
			pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());		//	Harris�����_�o�͌���
			getHarrisKeypoints(cloud_normals, keypoints, 0.01f);
			//	Harris�����_��pcl::PointCloud<pcl::PointXYZ>�ɃR�s�[
			pcl::PointCloud<pcl::PointXYZ>::Ptr kpts(new pcl::PointCloud<pcl::PointXYZ>());
			kpts->points.resize(keypoints->points.size());
			pcl::copyPointCloud(*keypoints, *kpts);	

			//	�����_�����FPFH�����ʂ̌v�Z
			pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());		//	FPFH������
			getFPFHSignatureAtKeypoints(cloud_segmented, kpts, normals, features, 0.04f);
		

			//------------------------------------------
			//	�����Ή��t��
			//------------------------------------------
			std::vector<int> correspondences;		//	�Ή��t���̂��߂̃C���f�b�N�X
			//	Kd-tree�Ń}�b�`���O�J�n
			//	OBJ�̓����� -> �V�[���̓�����
			correspondences.resize(obj_features->size());
			pcl::KdTreeFLANN<pcl::FPFHSignature33> searchTree;
			searchTree.setInputCloud(features);
			std::vector<int> idx(1);
			std::vector<float> L2Distance(1);
			for (int i = 0; i < obj_features->size(); i++)
			{
				correspondences[i] = -1;		//	�C���f�b�N�X�� -1 ��������Ή��t�����Ȃ�
				if (isnan(obj_features->points[i].histogram[0])) continue;		//	���̓_�̓����ʂ��󂾂������΂�
				searchTree.nearestKSearch(*obj_features, i, 1, idx, L2Distance);	//	�T����������_�C�T������ߖT�_�̌��C�ߖT�_�̃C���f�b�N�X�C�ߖT�_�܂ł̋���
				correspondences[i] = idx[0];		//	�ŋߖT�_�̃C���f�b�N�X��ۑ�
			}
			//	�V�[���̓����� -> OBJ�t�@�C���̓����ʂ̃}�b�`���O�𒲂ׂ�
			pcl::CorrespondencesPtr pCorrespondences(new pcl::Correspondences);		//	�Ή��_�̃C���f�b�N�X�̑Ή��\
			int nCorr = 0;		//	�Ή��_�̐�
			for (int i = 0; i < correspondences.size(); i++)
			{	//	-1�̓_�������đΉ��_�̐��𒲂ׂ�
				if (correspondences[i] >= 0) nCorr++;
			}
			pCorrespondences->resize(nCorr);
			for (int i = 0, j = 0; i < correspondences.size(); i++)
			{
				if (correspondences[i] > 0)
				{
					(*pCorrespondences)[j].index_query = i;			//	i�Ԗڂ̃\�[�X�̃C���f�b�N�X
					(*pCorrespondences)[j].index_match = correspondences[i];	//	i�Ԗڂ̃^�[�Q�b�g�̃C���f�b�N�X
					j++;
				}
			}
			//	RANSAC�ɂ���Ή��_����
			pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector;
			rejector.setInputSource(obj_kpts);		//	Harris�����_��XYZ���W
			rejector.setInputTarget(kpts);
			rejector.setInputCorrespondences(pCorrespondences);
			rejector.getCorrespondences(*pCorrespondences);		//	��̑Ή��\���㏑��

			//------------------------------------------
			//	�����ʒu���v�Z
			//------------------------------------------
			Eigen::Matrix4f initialTransMat;
			pcl::registration::TransformationEstimation<pcl::PointXYZ, pcl::PointXYZ>::Ptr 
				transEst(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>());		//	�ʒu�p������G���W��
			transEst->estimateRigidTransformation(*obj_kpts, *kpts, *pCorrespondences, initialTransMat);		//	�ϊ����𐄒�
			pcl::PointCloud<pcl::PointXYZ>::Ptr obj_transformed(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::transformPointCloud(*obj_filtered, *obj_transformed, initialTransMat);


			//------------------------------------------
			//	ICP�A���S���Y���ɂ�鍂���x�ʒu���킹
			//------------------------------------------
			pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr icp(new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>());
			icp->setInputSource(obj_transformed);
			icp->setInputTarget(cloud_filtered);
			//icp->setTransformationEpsilon(1e-6);
			//icp->setMaxCorrespondenceDistance(5.0f);
			//icp->setMaximumIterations(200);
			//icp->setEuclideanFitnessEpsilon(1.0f);
			//icp->setRANSACOutlierRejectionThreshold(1.0);
			pcl::PointCloud<pcl::PointXYZ>::Ptr obj_final(new pcl::PointCloud<pcl::PointXYZ>());
			icp->align(*obj_final);

			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(obj_transformed, 255, 255, 0);
			viewer.addPointCloud(obj_transformed, source_color, "transformed");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "transformed");
			if (icp->hasConverged())
			{
				pcl::transformPointCloud(*obj_transformed, *obj_final, icp->getFinalTransformation());
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> icp_color(obj_final, 255, 0, 0);
				viewer.addPointCloud(obj_final, icp_color, "final");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "final");
			}
			

		//}

			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> field_color(cloud_segmented, 255, 255, 255);
			viewer.addPointCloud<pcl::PointXYZ>(cloud_segmented, field_color, "cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

			//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(cloud_clustered, 0, 255, 255);
			//viewer.addPointCloud(cloud_clustered, target_color, "cluster");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cluster");

			viewer.spinOnce();
			viewer.removeAllPointClouds();

		//viewer.showCloud(cloud_clustered);

		//	Point Cloud�̕`��
		//viewer.showCloud(pointcloud);
		//viewer.showCloud(cloud_filtered);
		//viewer.showCloud(cloud_segmented);
		//viewer.showCloud(cloud_removed);

	}

	kinect.releaseAllInterface();
}

//---------------------------------------------
//	���ʂ̏���
//---------------------------------------------
template <typename PointT>
inline void removePlane(boost::shared_ptr<pcl::PointCloud<PointT>> src, boost::shared_ptr<pcl::PointCloud<PointT>> dst, float thresh)
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

//---------------------------------------------
//	�@���̐���
//---------------------------------------------
template <typename PointT, typename PointNT>
inline void addNormal(boost::shared_ptr<pcl::PointCloud<PointT>> cloud, boost::shared_ptr<pcl::PointCloud<PointNT>> normals, float radius)
{
	//	�_�Q����@���𐄒�
	pcl::NormalEstimationOMP<PointT, PointNT> ne;
	ne.setInputCloud(cloud);
	pcl::search::Search<PointT>::Ptr normalTree(new pcl::search::KdTree<PointT>());		//	�T�����@�FKdTree
	ne.setSearchMethod(normalTree);
	ne.setRadiusSearch(radius);
	//ne.setKSearch(10);			//	�ߖT10�_�𐄒�ɗ��p
	ne.setNumberOfThreads(100);
	ne.compute(*normals);
}


template <typename PointInT, typename PointKT>
inline void getHarrisKeypoints(boost::shared_ptr<pcl::PointCloud<PointInT>> cloud_normals, boost::shared_ptr<pcl::PointCloud<PointKT>> keypoints, float radius)
{
	pcl::HarrisKeypoint3D<PointInT, PointKT>::Ptr harrisDetector(new pcl::HarrisKeypoint3D<PointInT, PointKT>());
	harrisDetector->setNonMaxSupression(true);
	harrisDetector->setRadius(0.01f);				//	Harris�����_�̌v�Z���a(1cm)
	//harrisDetector->setKSearch(15);
	harrisDetector->setInputCloud(cloud_normals);
	harrisDetector->setSearchSurface(cloud_normals);
	harrisDetector->setMethod(pcl::HarrisKeypoint3D<PointInT, PointKT>::CURVATURE); // HARRIS, NOBLE, LOWE, TOMASI, CURVATURE 
	pcl::search::Search<PointInT>::Ptr harrisTree(new pcl::search::KdTree<PointInT>());		//	�T�����@�FKdTree
	harrisDetector->setSearchMethod(harrisTree);
	harrisDetector->compute(*keypoints);
}

inline void getFPFHSignatureAtKeypoints(
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints,
	pcl::PointCloud<pcl::Normal>::Ptr normals,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features,
	float radius)
{
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr fpfh(new pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>());		//	FPFH�����ʌv�Z��
	fpfh->setRadiusSearch(radius);								//	�T�����a�@�@������̎��̒T�����a���傫������K�v������
	//fpfh->setKSearch(25);
	pcl::search::Search<pcl::PointXYZ>::Ptr fpfhTree(new pcl::search::KdTree<pcl::PointXYZ>());		//	�T�����@�FKdTree
	fpfh->setSearchMethod(fpfhTree);
	fpfh->setNumberOfThreads(100);
	//	�_�Q�f�[�^�̓o�^
	fpfh->setInputCloud(keypoints);				//	�T���_
	fpfh->setSearchSurface(cloud);				//	�T���ɗ��p����T�[�t�F�X
	fpfh->setInputNormals(normals);				//	�@��
	fpfh->compute(*features);
}
