#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

// corres : 匹配的归一化坐标平面点
// delta_q_imu : 两帧之间imu的相对旋转
// calib_ric_result: 估计的外参数
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;  // 当前有几组 旋转约束
    Rc.push_back(solveRelativeR(corres));
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric); //将 imu系下的相对旋转-> 相机系下, ric初始化 I

    Eigen::MatrixXd A(frame_count * 4, 4);  // 利用旋转约束构建的方程系数，后面对它做SVD分解, 
    A.setZero();
    int sum_ok = 0; //useless
    for (int i = 1; i <= frame_count; i++) // 从第1个开始，略去第0个
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);  // r1 和 r2 理论上是相等的，这里得到角度差值
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;   // Huber 核函数
        ++sum_ok;
        Matrix4d L, R; // 

        // 计算 camera系下，R_ij的左乘算子
        double w = Quaterniond(Rc[i]).w();
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        // 计算 imu系 R_ij的右乘算子
        Quaterniond R_ij(Rimu[i]); // imu相对旋转
        w = R_ij.w();  // 实部
        q = R_ij.vec(); // 虚部
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R); // 构建方程系数矩阵
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV); 
    Matrix<double, 4, 1> x = svd.matrixV().col(3);  // 对A进行SVD分解，得到特征值最小的特征向量
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse(); // 得到最小特征向量对应的旋转矩阵,取逆得到RIC
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25) // 最后的条件 ： 1，已经估计了窗口的次数，2，倒数第二的特征值要>0.25，意味着最后一维度接近0空间的
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}



// corres 匹配的的点(归一化平面)
// 得到两帧之间的相对旋转R
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)  // 需要足够的点
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++) //取归一化点前两维
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat E = cv::findFundamentalMat(ll, rr); // 得到 FundmantalMatrix矩阵
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);

        if (determinant(R1) + 1.0 < 1e-09) //if R1的行列式  -1
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        // 利用三角化检测哪个R t是正确的
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j); // cv2eigen
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<float>(3);

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}



// 根据E 分解得到 四组解(R t 亮亮配对)
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);

    //绕 z 轴旋转90°
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);

    //绕 z 轴旋转-90°
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
