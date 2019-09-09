#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks; // 优化变量数据
    std::vector<int> drop_set;  //待marg优化变量的ID

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;  // 残差，视觉:2x1, IMU:15x1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};



class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info); // 加残差块相关信息(优化变量，待marg的变量)
    void preMarginalize(); // 计算每个残差的jacobian, 跟新parameter_block_data
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; // 所有观测项
    int m, n;  // m: marg , n: reserve
    std::unordered_map<long, int> parameter_block_size; //global size， 
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size ，需要marg的变量
    std::unordered_map<long, double *> parameter_block_data; // 优化变量地址，数据

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};




//
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
