#pragma once
#include <memory>

#include "icp_cuda.h"

namespace mvfr
{
    template<Point3D PointSource, PointWithNormal3D PointTarget, FloatingType Scalar = double>
    class IterativeClosestPointWithNormalsCuda: public IterativeClosestPointCuda<PointSource,PointTarget,Scalar>
    {
    public:
        using Ptr = std::shared_ptr<IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>>;
        using ConstPtr = std::shared_ptr<const IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>>;

        using IterativeClosestPointCuda = IterativeClosestPointCuda<PointSource,PointTarget,Scalar>;
        using PointCloudSource = typename IterativeClosestPointCuda::PointCloudSource;
        using PointCloudSourcePtr = typename IterativeClosestPointCuda::PointCloudSourcePtr;
        using PointCloudSourceConstPtr = typename IterativeClosestPointCuda::PointCloudSourceConstPtr;
        using PointCloudTarget = typename IterativeClosestPointCuda::PointCloudTarget;
        using PointCloudTargetPtr = typename IterativeClosestPointCuda::PointCloudTargetPtr;
        using PointCloudTargetConstPtr = typename IterativeClosestPointCuda::PointCloudTargetConstPtr;

        using Matrix4 = typename IterativeClosestPointCuda::Matrix4;
        using Search = typename IterativeClosestPointCuda::Search;
        using SearchPtr = typename IterativeClosestPointCuda::SearchPtr;
        using KdTree = typename IterativeClosestPointCuda::KdTree;      // as the uniform interface with pcl::Registration::KdTree
        using KdTreePtr = typename IterativeClosestPointCuda::KdTreePtr;

        using UpdateVisualizerCallbackSignature = typename IterativeClosestPointCuda::UpdateVisualizerCallbackSignature;


        IterativeClosestPointWithNormalsCuda()
        {
            reg_name_ = "IterativeClosestPointWithNormalsCuda";
        }

        ~IterativeClosestPointWithNormalsCuda() override = default;

        /// 设置源点云
        using IterativeClosestPointCuda::setInputSource;

        /// 同时设置CPU与GPU源点云
        using IterativeClosestPointCuda::setInputSourceDevice;

        /// 设置目标点云
        virtual inline void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

        /// 获取目标点云
        using IterativeClosestPointCuda::getInputTarget;

        /// 同时设置CPU与GPU目标点云
        virtual void setInputTargetDevice(const PointCloudTargetConstPtr& cloud,const CloudDevice& cloud_device) override;

        virtual void setInputTargetDevice(const PointCloudTargetConstPtr& cloud,const CloudDevice& cloud_device, const NormalDevice& normal_device);


        /// 获取GPU目标点云
        using IterativeClosestPointCuda::getInputTargetDevice;

        inline NormalDevice& const getTargetNormalDevice() const
        {
            return target_normal_device_;
        }

        /**
         *  @brief 执行配准.
         *  @param output 变换后的源点云
         *  @param guess  初始位姿
         */
        virtual inline void align(PointCloudSource& output,const Matrix4& guess) override;

        using IterativeClosestPointCuda::align;

        using IterativeClosestPointCuda::getClassName;


        /// 配准初始化函数
        virtual bool initCompute() override;
        using IterativeClosestPointCuda::deinitCompute;

    protected:
        // --------------------------------- 基本成员变量 --------------------------------
        using IterativeClosestPointCuda::reg_name_;
        using IterativeClosestPointCuda::input_;
        using IterativeClosestPointCuda::indices_;
        using IterativeClosestPointCuda::target_;
        using IterativeClosestPointCuda::source_device_;
        using IterativeClosestPointCuda::target_device_;
        using IterativeClosestPointCuda::aligned_cloud_device_;
        using IterativeClosestPointCuda::update_visualizer_;     //!< PCLVisualier 可视化回调函数
        NormalDevice target_normal_device_;

        // ----------------------- CorrespondencesEstimation -----------------------
        using IterativeClosestPointCuda::tree_;
        using IterativeClosestPointCuda::corr_dist_threshold_;
        using IterativeClosestPointCuda::target_cloud_updated_;
        using IterativeClosestPointCuda::source_cloud_updated_;
        //using IterativeClosestPointCuda::force_no_recompute_;   //!< 是否禁止更新搜索树

        using IterativeClosestPointCuda::correspondences_;
        using IterativeClosestPointCuda::correspondences_device_;
        using IterativeClosestPointCuda::min_number_correspondences_;   //!< 计算刚性变换所需的最小对应关系数量

        // -------------------------- TransformEstimation --------------------------
        using IterativeClosestPointCuda::final_transformation_;   //!< ICPCuda计算的最终变换矩阵
        using IterativeClosestPointCuda::transformation_;     //!< 本次迭代计算的变换矩阵
        using IterativeClosestPointCuda::previous_transformation_;    //!< 上次迭代计算的变换矩阵

        // -------------------------- ConvergenceCriteria --------------------------
        using IterativeClosestPointCuda::convergence_criteria_;
        using IterativeClosestPointCuda::nr_iterations_;     //!< 当前迭代次数
        using IterativeClosestPointCuda::max_iterations_;   //!< 最大迭代次数
        using IterativeClosestPointCuda::transformation_epsilon_;   //!< 相邻两次迭代平移向量的最大误差（参考 pcl::registration::ConvergenceCriteria）
        using IterativeClosestPointCuda::transformation_rotation_epsilon_;  //!< 相邻两次迭代旋转矩阵的最大误差
        using IterativeClosestPointCuda::euclidean_fitness_epsilon_;   //!< 相邻两次迭代欧式距离（近邻点距离均值）的最大误差
        using IterativeClosestPointCuda::converged_;    //!< ICP是否收敛

        void constructTransformationMatrix(const double& alpha,const double& beta,const double& gamma,const double& tx,const double& ty,const double& tz,Matrix4& transformation_matrix) const;
    };
}

#include "nicp_cuda.hpp"
