#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;

//通过Brief模板文件对图像特征点计算Brief描述子
class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};



class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);

	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	//寻找并建立关键帧与回环帧之间的匹配关系,返回True即为确定构成回环
	bool findConnection(KeyFrame* old_kf);
	//计算窗口中所有特征点的描述子
	void computeWindowBRIEFPoint();
	//检测新的特征点并计算所有特征点的描述子，为了回环检测
	void computeBRIEFPoint();
	//void extractBrief();
	//计算两个描述子之间的汉明距离
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);

	//关键帧中某个特征点的描述子与回环帧的所有描述子匹配,得到最佳匹配点(像素和归一化plane)
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	//将此关键帧对象与某个回环帧进行BRIEF描述子匹配
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	//通过RANSAC的基本矩阵检验去除匹配异常的点
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	//通过RANSAC的PNP检验去除匹配异常的点
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);

	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();



	double time_stamp; //关键帧的时间戳
	int index; //关键帧的全局索引
	int local_index;//此关键帧在 两个回环帧之间的local index，比如 有1-10帧，当前是第5帧，4和10帧构成闭环，那么当前帧的local index为1，在optimize4DoF里赋值
	Eigen::Vector3d vio_T_w_i; //
	Eigen::Matrix3d vio_R_w_i; 
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;

	cv::Mat image;//关键帧的图像
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d; //关键帧能观测到的3d点
	vector<cv::Point2f> point_2d_uv;//关键帧相机平面的2d点
	vector<cv::Point2f> point_2d_norm;//关键帧上3d点归一化后的坐标
	vector<double> point_id;//特征点id
	vector<cv::KeyPoint> keypoints;//新提取的fast角点
	vector<cv::KeyPoint> keypoints_norm; //新提取的fast角点： 归一化平面 
	vector<cv::KeyPoint> window_keypoints;//此关键帧原本特征点（point_2d_uv）
	vector<BRIEF::bitset> brief_descriptors;//新提取角点计算的描述子
	vector<BRIEF::bitset> window_brief_descriptors;//window_keypoints计算的描述子
	bool has_fast_point;
	int sequence; //图像序列，默认为1，pose_graph_node.cpp中创建关键帧的时候传入

	bool has_loop;//当前帧是否存在闭环帧
	int loop_index;//闭环真的idx
	Eigen::Matrix<double, 8, 1 > loop_info;//与闭环帧之间的信息：relative_t(3) relative_r(4) relative_yaw(1)
};

