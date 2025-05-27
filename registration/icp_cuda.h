/**
 *
 *  @file      icp_cuda.h
 *  @brief     CUDA implementation of point-to-point ICP
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/pcl_base.h>
#include <pcl/correspondence.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/default_convergence_criteria.h>

#include "../search/search_cuda.h"
#include "../search/octree_cuda.h"
#include "../utilities/types.h"

namespace mvfr
{
    /**
     *  @class   IterativeClosestPointCuda
     *  @brief   CUDA加速ICP
     *  @details 基于SearchCuda近邻点搜索（默认实例化为OcTreeCuda派生类），对 point-to-point icp 算法进行加速
     *  @tparam  PointSource 源点云数据类型
     *  @tparam  PointTarget 目标点云数据类型
     *  @tparam  Scalar      计算精度
     *
     *  @note 由于本算法实质上是 point-to-point icp 算法，因此算法仅要求目标点云与源点云中存在 xyz 字段. 如果存在其它字段的信息，则输出结果时保持不变
     */
    template<typename PointSource, typename PointTarget, typename Scalar = double>
    class IterativeClosestPointCuda : public pcl::PCLBase<PointSource>
    {
    public:
        using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;

        using Ptr = std::shared_ptr<IterativeClosestPointCuda<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const IterativeClosestPointCuda<PointSource, PointTarget, Scalar>>;

        using PointCloudSource = pcl::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
        using PointCloudTarget = pcl::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

        using Search = SearchCuda<PointTarget>;
        using SearchPtr = typename Search::Ptr;
        using KdTree = OcTreeCuda<PointTarget>;      // as the uniform interface with pcl::Registration::KdTree

        using TransformationEstimation = typename pcl::registration::TransformationEstimation<PointSource, PointTarget, Scalar>;
        using TransformationEstimationPtr = typename TransformationEstimation::Ptr;
        using TransformationEstimationConstPtr = typename TransformationEstimation::ConstPtr;

        using UpdateVisualizerCallbackSignature = void(const pcl::PointCloud<PointSource>&,
            const pcl::Indices&,
            const pcl::PointCloud<PointTarget>&,
            const pcl::Indices&);

        /**
         *  @brief IterativeClosestPointCuda object constructor
         */
        IterativeClosestPointCuda() :
            tree_(new OcTreeCuda<PointTarget>),
            reg_name_("IterativeClosestPointCuda"),
            target_(),
            source_device_(nullptr, 0),
            target_device_(nullptr, 0),
            final_transformation_(Matrix4::Identity()),
            transformation_(Matrix4::Identity()),
            previous_transformation_(Matrix4::Identity()),
            euclidean_fitness_epsilon_(-std::numeric_limits<double>::max()),
            corr_dist_threshold_(std::sqrt(std::numeric_limits<double>::max())),
            // transformation_estimation_(),
            correspondences_(new pcl::Correspondences)
        {
            convergence_criteria_.reset(
                new pcl::registration::DefaultConvergenceCriteria<Scalar>(nr_iterations_, transformation_, *correspondences_));
        }

        /**
         *  @brief IterativeClosestPointCuda object destructor
         */
        virtual ~IterativeClosestPointCuda() override = default;

        /// 设置源点云
        virtual void setInputSource(const PointCloudSourceConstPtr& cloud);

        /// 获取源点云
        inline PointCloudSourceConstPtr& const getInputSource() const
        {
            return (input_);
        }

        /// 同时设置CPU与GPU源点云
        virtual void setInputSourceDevice(const PointCloudSourceConstPtr& cloud, const CloudDevice& cloud_device);

        /// 获取GPU源点云
        inline CloudDevice& const getInputSourceDevice() const
        {
            return source_device_;
        }

        /// 设置目标点云
        virtual inline void setInputTarget(const PointCloudTargetConstPtr& cloud);

        /// 获取目标点云
        inline PointCloudTargetConstPtr& const getInputTarget() const
        {
            return (target_);
        }

        /// 同时设置CPU与GPU目标点云
        virtual void setInputTargetDevice(const PointCloudTargetConstPtr& cloud, const CloudDevice& cloud_device);

        /// 获取GPU目标点云
        inline CloudDevice& const getInputTargetDevice() const
        {
            return target_device_;
        }

        /**
         * @brief 设置目标点云搜索树.
         *
         * @param tree 目标点云搜索树
         * @param force_no_recompute 是否禁止通过输入目标点云更新搜索树
         *
         * @note 当 force_no_recompute == true 时，必须确保输入参数 \c tree 内存在正确的目标点云与设备点云
         *
         * @warning 注意 OcTreeCuda 设备点云的手动设置
         */
        inline void setSearchMethodTarget(const SearchPtr& tree, bool force_no_recompute = false)
        {
            tree_ = tree;
            force_no_recompute_ = force_no_recompute;
            // Since we just set a new tree, we need to check for updates
            target_cloud_updated_ = true;
        }

        /// 获取搜索树（目标点云）
        inline SearchPtr& getSearchMethodTarget() const
        {
            return (tree_);
        }

        /// 设置对应关系距离阈值（小于该阈值的对应关系才会参与变换矩阵计算）
        inline void setMaxCorrespondenceDistance(double distance_threshold)
        {
            corr_dist_threshold_ = distance_threshold;
        }

        /// 获取对应关系距离阈值
        inline double getMaxCorrespondenceDistance() const
        {
            return (corr_dist_threshold_);
        }

        ///// 设置ICPCuda的目标函数（变换矩阵计算方法）
        //void setTransformationEstimation(const TransformationEstimationPtr& te)
        //{
        //    transformation_estimation_ = te;
        //}

        /// 获取ICPCuda最终计算的变换矩阵
        inline const Matrix4& getFinalTransformation() const
        {
            return (final_transformation_);
        }

        /// 获取ICPCuda本次迭代计算的变换矩阵
        inline const Matrix4& getLastIncrementalTransformation() const
        {
            return (transformation_);
        }

        /// 设置最大迭代次数 （若迭代次数大于该阈值，则迭代视为收敛）
        inline void setMaximumIterations(int nr_iterations)
        {
            max_iterations_ = nr_iterations;
        }

        /// 获取最大迭代次数
        inline int getMaximumIterations() const
        {
            return (max_iterations_);
        }

        /// 设置变换矩阵误差阈值（若相邻两次迭代计算的变换矩阵小于该阈值，则迭代视为收敛）
        inline void setTransformationEpsilon(double epsilon)
        {
            transformation_epsilon_ = epsilon;
        }

        /// 获取变换矩阵误差阈值
        inline double getTransformationEpsilon() const
        {
            return (transformation_epsilon_);
        }

        /// 设置旋转矩阵误差阈值（若相邻两次迭代计算的旋转矩阵小于该阈值，则迭代视为收敛）
        inline void setTransformationRotationEpsilon(double epsilon)
        {
            transformation_rotation_epsilon_ = epsilon;
        }

        /// 获取旋转矩阵误差阈值
        inline double getTransformationRotationEpsilon() const
        {
            return (transformation_rotation_epsilon_);
        }

        /// 设置对应点欧式距离阈值（若相邻两次迭代过程中对应点的平均欧式距离小于该阈值，则迭代视为收敛）
        inline void setEuclideanFitnessEpsilon(double epsilon)
        {
            euclidean_fitness_epsilon_ = epsilon;
        }

        /// 获取对应关系欧式距离阈值
        inline double getEuclideanFitnessEpsilon() const
        {
            return (euclidean_fitness_epsilon_);
        }

        /// 设置PCLVisualizer可视化回调函数
        inline bool registerVisualizationCallback(std::function<UpdateVisualizerCallbackSignature>& visualizerCallback)
        {
            if (visualizerCallback) {
                update_visualizer_ = visualizerCallback;
                pcl::Indices indices;
                update_visualizer_(*input_, indices, *target_, indices);
                return (true);
            }
            return (false);
        }

        /// 获取ICP迭代敛散性判断器
        inline typename pcl::registration::DefaultConvergenceCriteria<Scalar>::Ptr
            getConvergeCriteria()
        {
            return convergence_criteria_;
        }

        /// 判断迭代是否收敛
        inline bool hasConverged() const
        {
            return (converged_);
        }


        /// 获取最终迭代的次数
        inline int getFinalIterations(void) const
        {
            if (!converged_)
            {
                PCL_ERROR("[% s::getFinalIterations] ICPCuda 未计算或迭代未收敛，无法获得最终的迭代次数! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
                return 0;
            }
            return nr_iterations_;
        }


        /**
         *  @brief  比较计算得到的变化矩阵与实际变换矩阵 \c ground_th 间的差异
         *  @param  ground_th 实际变换矩阵
         *  @retval           （arccos((tr(Rg*Rc')-1)/2), ||tg - tc||）, Rg, Rc分别为实际旋转矩阵与计算得到的旋转矩阵, tg, tc为实际平移向量与计算平移向量
         */
        inline std::pair<double, double> getFitnessScore(const Matrix4& ground_th);


        /**
         *  @brief  计算配准后对应点的平均欧式距离.
         *  @param  max_range
         *  @retval
         */
        inline double getFitnessScore(double max_range = std::numeric_limits<double>::max());


        /**
         *  @brief 执行配准（初始位姿为单位矩阵）.
         *  @param output 变换后的源点云
         */
        inline void align(PointCloudSource& output);

        /**
         *  @brief 执行配准.
         *  @param output 变换后的源点云
         *  @param guess  初始位姿
         */
        inline void align(PointCloudSource& output, const Matrix4& guess);

        /// 获取GPU上配准后的源点云
        inline CloudDevice& const getAlignedCloudDevice(void) const
        {
            if (!hasConverged)
            {
                PCL_ERROR("[% s::getAlignedCloudDevice] 迭代未收敛，无法获得配准设备点云! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
                return CloudDevice(nullptr, 0);
            }

            return aligned_cloud_device_;
        }

        /// 获取类名
        inline const std::string& getClassName() const
        {
            return (reg_name_);
        }

        /// 配准初始化函数
        bool initCompute();
        using pcl::PCLBase<PointSource>::deinitCompute;

    protected:
        // -------------------------------- 导入基类成员变量 -------------------------------
        using pcl::PCLBase<PointSource>::input_;
        using pcl::PCLBase<PointSource>::indices_;

        // --------------------------------- 基本成员变量 --------------------------------
        std::string reg_name_;
        PointCloudTargetConstPtr target_;
        CloudDevice source_device_;
        CloudDevice target_device_;
        CloudDevice aligned_cloud_device_;
        std::function<UpdateVisualizerCallbackSignature> update_visualizer_;     //!< PCLVisualier 可视化回调函数

        // ----------------------- CorrespondencesEstimation -----------------------
        SearchPtr tree_;
        double corr_dist_threshold_;
        bool target_cloud_updated_{ true };
        bool source_cloud_updated_{ true };
        bool force_no_recompute_{ false };   //!< 是否禁止更新搜索树

        pcl::CorrespondencesPtr correspondences_;
        CorrespondencesDevice correspondences_device_;
        unsigned int min_number_correspondences_{ 3 };   //!< 计算刚性变换所需的最小对应关系数量

        // -------------------------- TransformEstimation --------------------------
        //TransformationEstimationPtr transformation_estimation_;
        Matrix4 final_transformation_;   //!< ICPCuda计算的最终变换矩阵
        Matrix4 transformation_;     //!< 本次迭代计算的变换矩阵
        Matrix4 previous_transformation_;    //!< 上次迭代计算的变换矩阵

        // -------------------------- ConvergenceCriteria --------------------------
        typename pcl::registration::DefaultConvergenceCriteria<Scalar>::Ptr convergence_criteria_;
        int nr_iterations_{ 0 };     //!< 当前迭代次数
        int max_iterations_{ 10 };   //!< 最大迭代次数
        double transformation_epsilon_{ 0.0 };   //!< 相邻两次迭代平移向量的最大误差（参考 pcl::registration::ConvergenceCriteria）
        double transformation_rotation_epsilon_{ 0.0 };  //!< 相邻两次迭代旋转矩阵的最大误差
        double euclidean_fitness_epsilon_;   //!< 相邻两次迭代欧式距离（近邻点距离均值）的最大误差
        bool converged_{ false };    //!< ICP是否收敛
    };
}

#include "icp_cuda.hpp"
