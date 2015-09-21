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

	//	OBJファイルを読み込む
	pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_pcd(new pcl::PointCloud<pcl::PointXYZ>());				//	objファイルの点群データ
	if (pcl::io::loadPolygonFileOBJ(filename, *mesh) == -1)
	{
		return -1;
	}
	//	PolygonMesh -> PointCloud<PointXYZ>
	pcl::fromPCLPointCloud2(mesh->cloud, *obj_pcd);
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	//	単位変換 mm -> m
	for (int i = 0; i < obj_pcd->height*obj_pcd->width; i++)
	{
		pcl::PointXYZ point;
		point.x = obj_pcd->points[i].x / 1000.0f;
		point.y = obj_pcd->points[i].y / 1000.0f;
		point.z = obj_pcd->points[i].z / 1000.0f;
		obj_cloud->push_back(point);
	}

	//------------------------------------------
	//	OBJデータから特徴量を計算
	//------------------------------------------

	//	点群の軽量化（ダウンサンプリング）
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_filtered(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::ApproximateVoxelGrid <pcl::PointXYZ> avg;
	avg.setInputCloud(obj_cloud);
	avg.setLeafSize(0.001f, 0.001f, 0.001f);	//	VoxelGridの大きさをX,Y,Zで指定 ここでは1mmにした
	avg.filter(*obj_filtered);

	//	全ての点の法線の推定
	//	PointXYZからNormal及びPointNormalを生成
	pcl::PointCloud<pcl::Normal>::Ptr obj_normals(new pcl::PointCloud<pcl::Normal>());		//	法線のみ
	pcl::PointCloud<pcl::PointNormal>::Ptr obj_cloud_normals(new pcl::PointCloud<pcl::PointNormal>());	//	3次元点群 + 推定された法線
	addNormal(obj_filtered, obj_normals, 0.01f);
	pcl::concatenateFields(*obj_filtered, *obj_normals, *obj_cloud_normals);

	//	Harris特徴点検出器
	//	PointNormalからPointXYZIを生成
	pcl::PointCloud<pcl::PointXYZI>::Ptr obj_keypoints(new pcl::PointCloud<pcl::PointXYZI>());		//	Harris特徴点出力結果
	getHarrisKeypoints(obj_cloud_normals, obj_keypoints, 0.01f);
	//	Harris特徴点をpcl::PointCloud<pcl::PointXYZ>にコピー
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_kpts(new pcl::PointCloud<pcl::PointXYZ>());
	obj_kpts->points.resize(obj_keypoints->points.size());
	pcl::copyPointCloud(*obj_keypoints, *obj_kpts);

	//	特徴点周りのFPFH特徴量の計算
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr obj_features(new pcl::PointCloud<pcl::FPFHSignature33>());		//	FPFH特徴量
	getFPFHSignatureAtKeypoints(obj_filtered, obj_kpts, obj_normals, obj_features, 0.04f);
	//viewer.showCloud(obj_cloud);
	

	Sleep(4000);		//	Kinectの起動を待つ

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

		//	フレームの描画
		resize(colorImg, colorImg, cv::Size(), 0.5, 0.5);
		cv::imshow("Color Image", colorImg);
		cv::imshow("Depth Image", depthImg);
		//cv::imshow("Depth Error", depthError);

		//	キー入力受付
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
				{	//	PCLに渡す時はmm単位に揃える
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
		//	Kinectデータから対象となる部分だけを抽出
		//-----------------------------------------------------

		////	NaN点の除去
		//std::vector<int> indices;
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::removeNaNFromPointCloud(*pointcloud, *cloud, indices);

		//	点群の軽量化（ダウンサンプリング）
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::ApproximateVoxelGrid <pcl::PointXYZ> avg;
		avg.setInputCloud(pointcloud);
		//avg.setLeafSize(0.001f, 0.001f, 0.001f);	//	VoxelGridの大きさをX,Y,Zで指定 ここでは1mmにした
		avg.filter(*cloud_filtered);

		//	平面除去
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented(new pcl::PointCloud<pcl::PointXYZ>());
		removePlane(cloud_filtered, cloud_segmented, 0.02f);
		removePlane(cloud_segmented, cloud_segmented, 0.03f);
		//removePlane(cloud_segmented, cloud_segmented, 0.04f);

		////	外れ値の除去(Statistical)
		////	近傍n点との距離が標準偏差のk倍以上だったら除去
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_removed(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
		//sor.setInputCloud(cloud_segmented);
		//sor.setMeanK(50);				//	調べる近傍点の個数
		//sor.setStddevMulThresh(1.0);	//	標準偏差の何倍まで許容するか
		////sor.setNegative(true);			//	trueなら外れ値の方のみ残す
		//sor.filter(*cloud_removed);


		////	クラスタリング
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
			//	Kinectデータから特徴量を計算
			//------------------------------------------

			//	全ての点の法線の推定
			//	PointXYZからNormal及びPointNormalを生成
			pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());		//	法線のみ
			pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>());	//	3次元点群 + 推定された法線
			addNormal(cloud_segmented, normals, 0.01f);
			pcl::concatenateFields(*cloud_segmented, *normals, *cloud_normals);

			//	Harris特徴点検出器
			//	PointNormalからPointXYZIを生成
			pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());		//	Harris特徴点出力結果
			getHarrisKeypoints(cloud_normals, keypoints, 0.01f);
			//	Harris特徴点をpcl::PointCloud<pcl::PointXYZ>にコピー
			pcl::PointCloud<pcl::PointXYZ>::Ptr kpts(new pcl::PointCloud<pcl::PointXYZ>());
			kpts->points.resize(keypoints->points.size());
			pcl::copyPointCloud(*keypoints, *kpts);	

			//	特徴点周りのFPFH特徴量の計算
			pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());		//	FPFH特徴量
			getFPFHSignatureAtKeypoints(cloud_segmented, kpts, normals, features, 0.04f);
		

			//------------------------------------------
			//	初期対応付け
			//------------------------------------------
			std::vector<int> correspondences;		//	対応付けのためのインデックス
			//	Kd-treeでマッチング開始
			//	OBJの特徴量 -> シーンの特徴量
			correspondences.resize(obj_features->size());
			pcl::KdTreeFLANN<pcl::FPFHSignature33> searchTree;
			searchTree.setInputCloud(features);
			std::vector<int> idx(1);
			std::vector<float> L2Distance(1);
			for (int i = 0; i < obj_features->size(); i++)
			{
				correspondences[i] = -1;		//	インデックスが -1 だったら対応付けがない
				if (isnan(obj_features->points[i].histogram[0])) continue;		//	その点の特徴量が空だったら飛ばす
				searchTree.nearestKSearch(*obj_features, i, 1, idx, L2Distance);	//	探索する特徴点，探索する近傍点の個数，近傍点のインデックス，近傍点までの距離
				correspondences[i] = idx[0];		//	最近傍点のインデックスを保存
			}
			//	シーンの特徴量 -> OBJファイルの特徴量のマッチングを調べる
			pcl::CorrespondencesPtr pCorrespondences(new pcl::Correspondences);		//	対応点のインデックスの対応表
			int nCorr = 0;		//	対応点の数
			for (int i = 0; i < correspondences.size(); i++)
			{	//	-1の点を除いて対応点の数を調べる
				if (correspondences[i] >= 0) nCorr++;
			}
			pCorrespondences->resize(nCorr);
			for (int i = 0, j = 0; i < correspondences.size(); i++)
			{
				if (correspondences[i] > 0)
				{
					(*pCorrespondences)[j].index_query = i;			//	i番目のソースのインデックス
					(*pCorrespondences)[j].index_match = correspondences[i];	//	i番目のターゲットのインデックス
					j++;
				}
			}
			//	RANSACによる誤対応点除去
			pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector;
			rejector.setInputSource(obj_kpts);		//	Harris特徴点のXYZ座標
			rejector.setInputTarget(kpts);
			rejector.setInputCorrespondences(pCorrespondences);
			rejector.getCorrespondences(*pCorrespondences);		//	先の対応表を上書き

			//------------------------------------------
			//	初期位置を計算
			//------------------------------------------
			Eigen::Matrix4f initialTransMat;
			pcl::registration::TransformationEstimation<pcl::PointXYZ, pcl::PointXYZ>::Ptr 
				transEst(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>());		//	位置姿勢推定エンジン
			transEst->estimateRigidTransformation(*obj_kpts, *kpts, *pCorrespondences, initialTransMat);		//	変換式を推定
			pcl::PointCloud<pcl::PointXYZ>::Ptr obj_transformed(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::transformPointCloud(*obj_filtered, *obj_transformed, initialTransMat);


			//------------------------------------------
			//	ICPアルゴリズムによる高精度位置合わせ
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

		//	Point Cloudの描画
		//viewer.showCloud(pointcloud);
		//viewer.showCloud(cloud_filtered);
		//viewer.showCloud(cloud_segmented);
		//viewer.showCloud(cloud_removed);

	}

	kinect.releaseAllInterface();
}

//---------------------------------------------
//	平面の除去
//---------------------------------------------
template <typename PointT>
inline void removePlane(boost::shared_ptr<pcl::PointCloud<PointT>> src, boost::shared_ptr<pcl::PointCloud<PointT>> dst, float thresh)
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

//---------------------------------------------
//	法線の推定
//---------------------------------------------
template <typename PointT, typename PointNT>
inline void addNormal(boost::shared_ptr<pcl::PointCloud<PointT>> cloud, boost::shared_ptr<pcl::PointCloud<PointNT>> normals, float radius)
{
	//	点群から法線を推定
	pcl::NormalEstimationOMP<PointT, PointNT> ne;
	ne.setInputCloud(cloud);
	pcl::search::Search<PointT>::Ptr normalTree(new pcl::search::KdTree<PointT>());		//	探索方法：KdTree
	ne.setSearchMethod(normalTree);
	ne.setRadiusSearch(radius);
	//ne.setKSearch(10);			//	近傍10点を推定に利用
	ne.setNumberOfThreads(100);
	ne.compute(*normals);
}


template <typename PointInT, typename PointKT>
inline void getHarrisKeypoints(boost::shared_ptr<pcl::PointCloud<PointInT>> cloud_normals, boost::shared_ptr<pcl::PointCloud<PointKT>> keypoints, float radius)
{
	pcl::HarrisKeypoint3D<PointInT, PointKT>::Ptr harrisDetector(new pcl::HarrisKeypoint3D<PointInT, PointKT>());
	harrisDetector->setNonMaxSupression(true);
	harrisDetector->setRadius(0.01f);				//	Harris特徴点の計算半径(1cm)
	//harrisDetector->setKSearch(15);
	harrisDetector->setInputCloud(cloud_normals);
	harrisDetector->setSearchSurface(cloud_normals);
	harrisDetector->setMethod(pcl::HarrisKeypoint3D<PointInT, PointKT>::CURVATURE); // HARRIS, NOBLE, LOWE, TOMASI, CURVATURE 
	pcl::search::Search<PointInT>::Ptr harrisTree(new pcl::search::KdTree<PointInT>());		//	探索方法：KdTree
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
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr fpfh(new pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>());		//	FPFH特徴量計算器
	fpfh->setRadiusSearch(radius);								//	探索半径　法線推定の時の探索半径より大きくする必要がある
	//fpfh->setKSearch(25);
	pcl::search::Search<pcl::PointXYZ>::Ptr fpfhTree(new pcl::search::KdTree<pcl::PointXYZ>());		//	探索方法：KdTree
	fpfh->setSearchMethod(fpfhTree);
	fpfh->setNumberOfThreads(100);
	//	点群データの登録
	fpfh->setInputCloud(keypoints);				//	探索点
	fpfh->setSearchSurface(cloud);				//	探索に利用するサーフェス
	fpfh->setInputNormals(normals);				//	法線
	fpfh->compute(*features);
}
