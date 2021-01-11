/*
 * @Description: 
 * @Author: julian
 * @E-mail: 1546450025@qq.com
 * @Date: 2019-12-03 16:56:31
 * @LastEditTime: 2021-01-11 12:00:08
 */
#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> // 残差是2维、 pose_i、 pose_j、 外参、逆深度
{
  public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;  // i 和 j 时刻的 特征点的 归一化坐标
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t; 
};
