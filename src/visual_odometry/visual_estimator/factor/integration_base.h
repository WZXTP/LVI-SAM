#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;

/**
* @class IntegrationBase IMU pre-integration class
* @Description 
*/

/*实现了IMU（惯性测量单元）的预积分算法，用于计算两个关键帧之间的旋转、速度和位移变化量，
并维护雅可比矩阵和协方差矩阵，用于优化时的误差计算。
*/
class IntegrationBase
{
  public:

    double dt;//时间间隔
    Eigen::Vector3d acc_0, gyr_0;//当前的加速度和角速度
    Eigen::Vector3d acc_1, gyr_1;//上一时刻的加速度和角速度

    const Eigen::Vector3d linearized_acc, linearized_gyr;//线性化的加速度计和陀螺仪
    Eigen::Vector3d linearized_ba, linearized_bg;//加速度计和陀螺仪的线性化偏差（bias）

    Eigen::Matrix<double, 15, 15> jacobian;//雅可比矩阵
    Eigen::Matrix<double, 15, 15> covariance;//协方差矩阵，表示误差的传播
    Eigen::Matrix<double, 18, 18> noise;

    double sum_dt;
    //位移、四元数旋转和速度的预积分增量。
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    // saves all the IMU measurements and time difference between two image frames，保存所有的IMU测量和两个图像帧之间的时间差
    std::vector<double> dt_buf;//保存两个图像帧之间的时间差
    std::vector<Eigen::Vector3d> acc_buf;//IMU的加速度测量值
    std::vector<Eigen::Vector3d> gyr_buf;//IMU的陀螺仪测量值

    IntegrationBase() = delete;

    IntegrationBase(const Eigen::Vector3d &_acc_0, 
                    const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, 
                    const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, 
          gyr_0{_gyr_0}, 
          linearized_acc{_acc_0}, 
          linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, 
          linearized_bg{_linearized_bg},
          jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, 
          covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, 
          delta_p{Eigen::Vector3d::Zero()}, 
          delta_q{Eigen::Quaterniond::Identity()}, 
          delta_v{Eigen::Vector3d::Zero()}

    {
        //噪声矩阵 noise 的初始化
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    // after optimization, repropagate pre-integration using the updated bias。优化后，使用更新后的偏置重新预积分
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;//更新偏置值
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();//重置雅可比矩阵为单位矩阵
        covariance.setZero();//重置协方差矩阵为零矩阵
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)//重新传播IMU数据
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    /**
    * @brief   IMU预积分传播方程
    * @Description  积分计算两个关键帧之间IMU测量的变化量： 
    *               旋转delta_q 速度delta_v 位移delta_p
    *               加速度的biaslinearized_ba 陀螺仪的Bias linearized_bg
    *               同时维护更新预积分的Jacobian和Covariance,计算优化时必要的参数
    * @param[in]   _dt 时间间隔
    * @param[in]   _acc_1 线加速度
    * @param[in]   _gyr_1 角速度
    * @return  void
    */
    //实现了一个IMU（惯性测量单元）数据传播函数，用于更新状态变量（位置、姿态、速度、偏置）在时间步长 dt 内的变化。
    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;//积分后的位移变化
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        //中点积分法进行状态更新
        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //更新状态变量
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();//姿态归一化
        //时间步长累积和更新传感器值
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }

    /**
    * @brief   IMU预积分中采用中值积分递推Jacobian和Covariance
    *          构造误差的线性化递推方程，得到Jacobian和Covariance递推公式-> Paper 式9、10、11
    * @return  void
    */
    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;//平均角速度，去除了陀螺仪偏置 linearized_bg 的影响
            Vector3d a_0_x = _acc_0 - linearized_ba;//去除了加速度偏置 linearized_ba 后的加速度
            Vector3d a_1_x = _acc_1 - linearized_ba;//加速度
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;//分别表示角速度和两个加速度向量的反对称矩阵（也称作叉乘矩阵）

            //反对称矩阵
            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            //状态转移矩阵 F 的构建
            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();//位置
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;//位置对姿态和加速度的影响
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;//位置对速度的影响
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;//姿态部分的更新
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;//姿态对陀螺仪偏置的影响
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;//速度部分的更新
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;//速度对加速度偏置的影响
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;//速度对加速度偏置的影响
            F.block<3, 3>(9, 9) = Matrix3d::Identity();// 加速度偏置保持不变
            F.block<3, 3>(12, 12) = Matrix3d::Identity();// 陀螺仪偏置保持不变
            //cout<<"A"<<endl<<A<<endl;

            //过程噪声矩阵 V 的构建
            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            //雅可比矩阵和协方差的更新
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }

    
    // calculate residuals for ceres optimization, used in imu_factor.h
    // paper equation 24
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;//残差向量，维度15x1，包括了位置、姿态、速度、加速度偏置和陀螺仪偏置的误差。

        //雅可比矩阵提取，分别表示位置、姿态、速度相对于加速度偏置（BA）和陀螺仪偏置（BG）的偏导。
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        //加速度偏置和陀螺仪偏置的偏差。
        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        //修正后的预积分分量
        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);//修正后的姿态变化量（四元数），通过陀螺仪偏置修正。
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;//修正后的速度变化量，考虑了加速度和陀螺仪偏置。
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;//修正后的位置变化量，同样考虑了偏置影响。

        //残差计算
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }
};
