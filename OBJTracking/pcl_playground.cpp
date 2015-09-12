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
	
	////	OBJファイルを読み込む
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
		////	測定エラー点をマスク画像化
		//depthError = cv::Mat(depthMat.size(), CV_8UC1);
		//depthMat.convertTo(depthMatd, CV_32F);
		//cv::threshold(depthMatd, depthError, 1, 255, cv::THRESH_BINARY);
		//	デプスマップをカラー画像化
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
		//	点群の軽量化（ダウンサンプリング）
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::ApproximateVoxelGrid <pcl::PointXYZ> avg;
		avg.setInputCloud(pointcloud);
		avg.setLeafSize(0.001f, 0.001f, 0.001f);	//	VoxelGridの大きさをX,Y,Zで指定 ここでは1mmにした
		avg.filter(*cloud_filtered);

		//	平面除去
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented(new pcl::PointCloud<pcl::PointXYZ>());
		removePlane(cloud_filtered, cloud_segmented, 0.03);
		removePlane(cloud_segmented, cloud_segmented, 0.04);
		removePlane(cloud_segmented, cloud_segmented, 0.05);

		////	外れ値の除去(Statistical)
		////	近傍n点との距離が標準偏差のk倍以上だったら除去
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_removed(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
		//sor.setInputCloud(cloud_segmented);
		//sor.setMeanK(20);				//	調べる近傍点の個数
		//sor.setStddevMulThresh(1.0);	//	標準偏差の何倍まで許容するか
		////sor.setNegative(true);			//	trueなら外れ値の方のみ残す
		//sor.filter(*cloud_removed);

		//	Harris特徴点検出器
		pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;
		detector.setNonMaxSupression(true);
		detector.setRadius(0.01);				//	Harris特徴点の計算半径(1cm)
		detector.setInputCloud(cloud_segmented);
		pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
		detector.compute(*keypoints);

		////	クラスタリング
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


			////	外れ値の除去(Radius)
			////	半径rの近傍球内にn個以上点が入っていなければ除去
			////	いくらなんでも遅すぎ
			//pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
			//outrem.setInputCloud(pointcloud);
			//outrem.setRadiusSearch(0.1);		//	半径0.8近傍
			//outrem.setMinNeighborsInRadius(3);	//	半径内に入っているべき点の最低個数
			////outrem.setNegative(true);
			//outrem.filter(*cloud_filtered);

			////	全ての点において法線の推定
			//pcl::PointCloud<pcl::PointNormal>::Ptr normal(new pcl::PointCloud<pcl::PointNormal>());
			//pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
			//ne.setInputCloud(cloud_segmented);
			//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());		//	探索方法：KdTree
			//ne.setSearchMethod(tree);
			//ne.setRadiusSearch(0.03);		//	探索半径
			//ne.compute(*normal);

			////	FPFH特徴量の計算
			//pcl::PointCloud<pcl::FPFHSignature33>::Ptr features;
			//pcl::FPFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
			//fpfh.setInputCloud(cloud_removed);
			//fpfh.setInputNormals(normal);
			//fpfh.setSearchMethod(tree);
			//fpfh.setRadiusSearch(0.05);		//	探索半径　法線推定の時の探索半径より大きくする必要がある
			//fpfh.compute(*features);

			//	Point Cloudの描画
			//viewer.showCloud(pointcloud);
			//viewer.showCloud(cloud_filtered);
			//viewer.showCloud(cloud_segmented);
			//viewer.showCloud(cloud_removed);
		if (keypoints->is_dense)
			viewer.showCloud(keypoints);

		//	フレームの描画
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
	//	平面の検出
	pcl::SACSegmentation<PointT> seg;
	seg.setInputCloud(src);
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);		//	平面を検出
	seg.setMethodType(pcl::SAC_RANSAC);			//	RANSACで誤対応点除去
	seg.setDistanceThreshold(thresh);				//	この変動幅まで許す
	pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients());	//	推定されたモデル式 ax+by+cz+d=0
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());			//	平面点群のIndex
	seg.segment(*inliers, *coeffs);
	//	平面の除去
	pcl::ExtractIndices<PointT>::Ptr extract(new pcl::ExtractIndices<PointT>());
	extract->setInputCloud(src);
	extract->setIndices(inliers);			//	平面点群のIndex
	extract->setNegative(true);			//	平面除去
	extract->filter(*dst);
}