#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts; // 每一帧中新提取的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts; 
    vector<cv::Point2f> prev_un_pts, cur_un_pts; // 去畸变之后的点(归一化平面)
    vector<cv::Point2f> pts_velocity;   // 去畸变之后feature的速度
    vector<int> ids; //feature id
    vector<int> track_cnt; //feature 被跟踪的次数
    map<int, cv::Point2f> cur_un_pts_map; // // id, 去畸变之后的点(归一化平面)
    map<int, cv::Point2f> prev_un_pts_map; //
    camodocal::CameraPtr m_camera; //相机模型(cata鱼眼、针孔..)
    double cur_time; // 计算feature速度用的， pre-cur = dt
    double prev_time;

    static int n_id;
};
