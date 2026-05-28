#pragma once

#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "nicp_cuda.h"

namespace mvfr{
    template<Point3D PointSource,PointWithNormal3D PointTarget,FloatingType Scalar>
    void IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>::setInputTarget(const PointCloudTargetConstPtr & cloud)
    {
        IterativeClosestPointCuda::setInputTarget(cloud);
        target_normal_device_ = NormalDevice(nullptr,0);
    }
    
    template<Point3D PointSource,PointWithNormal3D PointTarget,FloatingType Scalar>
    void IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>::setInputTargetDevice(const PointCloudTargetConstPtr & cloud,const CloudDevice & cloud_device)
    {
        IterativeClosestPointCuda::setInputTargetDevice(cloud,cloud_device);
        target_normal_device_ = NormalDevice(nullptr,0);
    }
    
    
    template<Point3D PointSource,PointWithNormal3D PointTarget,FloatingType Scalar>
    void IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>::setInputTargetDevice(const PointCloudTargetConstPtr & cloud,const CloudDevice & cloud_device,const NormalDevice & normal_device)
    {
        IterativeClosestPointWithNormalsCuda::setInputTargetDevice(cloud,cloud_device);
        if(normal_device.first == nullptr || normal_device.second != cloud_device.second)
        {
            PCL_ERROR("[% s::setInputSourceDevice] 输入目标点云法向量为空或点云与法向量数量不匹配! %s(%d)\n",getClassName().c_str(),__FILE__,__LINE__);
            return;
        }
        target_normal_device_ = normal_device;
    }


    template<Point3D PointSource,PointWithNormal3D PointTarget,FloatingType Scalar>
    void IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>::align(PointCloudSource & output,const Matrix4 & guess)
    {
        if(!initCompute())
            return;

        // 重置参数
        aligned_cloud_device_ = CloudDevice(nullptr,0);
        nr_iterations_ = 0;
        correspondences_device_ = CorrespondencesDevice(nullptr,nullptr,nullptr,0);
        transformation_ = previous_transformation_ = Matrix4::Identity();
        final_transformation_ = guess;
        converged_ = false;
        convergence_criteria_->setMaximumIterations(max_iterations_);
        convergence_criteria_->setRelativeMSE(euclidean_fitness_epsilon_);
        convergence_criteria_->setTranslationThreshold(transformation_epsilon_);
        if(transformation_rotation_epsilon_ > 0)
            convergence_criteria_->setRotationThreshold(transformation_rotation_epsilon_);

        // 初始化设备源点云 aligned_cloud_device_
        pcl::PointXYZ* cloud_device_temp_ptr;
        cudaSafeCall(cudaMalloc(&cloud_device_temp_ptr,source_device_.second * sizeof(pcl::PointXYZ)));
        cudaSafeCall(cudaMemcpy(cloud_device_temp_ptr,source_device_.first.get(),source_device_.second * sizeof(pcl::PointXYZ),cudaMemcpyDeviceToDevice));
        aligned_cloud_device_.first = getCudaSharedPtr(cloud_device_temp_ptr);
        aligned_cloud_device_.second = source_device_.second;
        if(guess != Matrix4::Identity())		// 若初始位姿不为单位矩阵，则对源点云进行变换
            transformCloudDevice(aligned_cloud_device_.first.get(),aligned_cloud_device_.second,guess);

        // 分配近邻点搜索索引、近邻点距离、对应关系GPU内存
        pcl::index_t* indices_device_temp_ptr;
        cudaSafeCall(cudaMalloc(&indices_device_temp_ptr,source_device_.second * sizeof(pcl::index_t)));
        IndicesDevice indices_device(getCudaSharedPtr(indices_device_temp_ptr),source_device_.second);

        cudaSafeCall(cudaMalloc(&indices_device_temp_ptr,source_device_.second * sizeof(pcl::index_t)));
        std::get<0>(correspondences_device_) = getCudaSharedPtr(indices_device_temp_ptr);

        cudaSafeCall(cudaMalloc(&indices_device_temp_ptr,source_device_.second * sizeof(pcl::index_t)));
        std::get<1>(correspondences_device_) = getCudaSharedPtr(indices_device_temp_ptr);


        float* distances_device_temp_ptr;
        cudaSafeCall(cudaMalloc(&distances_device_temp_ptr,source_device_.second * sizeof(float)));
        DistancesDevice distances_device(getCudaSharedPtr(distances_device_temp_ptr),source_device_.second);

        cudaSafeCall(cudaMalloc(&distances_device_temp_ptr,source_device_.second * sizeof(float)));
        std::get<2>(correspondences_device_) = getCudaSharedPtr(distances_device_temp_ptr);
        std::get<3>(correspondences_device_) = source_device_.second;

        // 分配主机对应关系CPU内存
        std::shared_ptr<pcl::index_t> corr_src_host_ptr(new pcl::index_t[std::get<3>(correspondences_device_)],std::default_delete<pcl::index_t[]>());
        std::shared_ptr<pcl::index_t> corr_tgt_host_ptr(new pcl::index_t[std::get<3>(correspondences_device_)],std::default_delete<pcl::index_t[]>());
        std::shared_ptr<float> corr_dis_host_ptr(new float[std::get<3>(correspondences_device_)],std::default_delete<float[]>());


        // 迭代直至收敛
        do {
            // 保存上次迭代的计算的变换矩阵
            previous_transformation_ = transformation_;

            // 计算对应关系
            tree_->approxNearestSearch(aligned_cloud_device_,indices_device,distances_device);
            //tree_->nearestKSearch(aligned_cloud_device_, 1, indices_device, distances_device);
            const unsigned corr_num = correspondencesDeviceRejector(indices_device,distances_device,
                correspondences_device_,corr_dist_threshold_ * corr_dist_threshold_);

            // 检查对应关系是否充足
            if(corr_num < min_number_correspondences_) {
                PCL_ERROR("[% s::align] 源点云与目标点云对应关系不足! %s(%d)\n",getClassName().c_str(),__FILE__,__LINE__);
                convergence_criteria_->setConvergenceState(pcl::registration::DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES);
                converged_ = false;
                break;
            }

            // 更新 correspoondences_
            cudaSafeCall(cudaMemcpy(corr_src_host_ptr.get(),std::get<0>(correspondences_device_).get(),corr_num * sizeof(pcl::index_t),cudaMemcpyDeviceToHost));
            cudaSafeCall(cudaMemcpy(corr_tgt_host_ptr.get(),std::get<1>(correspondences_device_).get(),corr_num * sizeof(pcl::index_t),cudaMemcpyDeviceToHost));
            cudaSafeCall(cudaMemcpy(corr_dis_host_ptr.get(),std::get<2>(correspondences_device_).get(),corr_num * sizeof(float),cudaMemcpyDeviceToHost));

            correspondences_->resize(corr_num);
            for(int i = 0; i < corr_num; ++i)
                (*correspondences_)[i] = pcl::Correspondence(corr_src_host_ptr.get()[i],corr_tgt_host_ptr.get()[i],corr_dis_host_ptr.get()[i]);

            // 生成本次迭代计算的变换矩阵 transformation_
            //Eigen::Matrix<double,6,6> ATA = Eigen::Matrix<double,6,6>::Zero();
            //Eigen::Vector<double,6> ATb = Eigen::Vector<double,6>::Zero();
            //computeNICPMatrixDevice(aligned_cloud_device_,target_device_,target_normal_device_,correspondences_device_,
            //    corr_num,ATA,ATb);
            //Eigen::Vector<double,6> x = static_cast<Eigen::Vector<double,6>>(ATA.inverse() * ATb);    // 求解 Ax=b
            //constructTransformationMatrix(x(0),x(1),x(2),x(3),x(4),x(5),transformation_);
            {
                pcl::registration::TransformationEstimationPointToPlaneLLS<PointSource,PointTarget,Scalar> trans_est;
                pcl::PointCloud<PointSource> input_transformed;
                pcl::transformPointCloudWithNormals(*input_,input_transformed,final_transformation_);
                trans_est.estimateRigidTransformation(input_transformed,*target_,*correspondences_,transformation_);
            }


            // 基于 transformation_ 更新GPU源点云 aligned_cloud_device_
            transformCloudDevice(aligned_cloud_device_.first.get(),aligned_cloud_device_.second,transformation_);

            // 更新final_transformation_ (transformation是基于全局坐标系求得的，所以用左乘)
            final_transformation_ = transformation_ * final_transformation_;

            /// @todo 异步更新 update_visualizer
            // 更新PCLVisualizer可视化回调函数
            if(update_visualizer_ != nullptr) {
                pcl::Indices source_indices_good,target_indices_good;
                for(const pcl::Correspondence& corr : *correspondences_) {
                    source_indices_good.emplace_back(corr.index_query);
                    target_indices_good.emplace_back(corr.index_match);
                }
                pcl::PointCloud<PointSource> input_transformed;
                pcl::transformPointCloudWithNormals(*input_,input_transformed,final_transformation_);
                update_visualizer_(
                    input_transformed,source_indices_good,*target_,target_indices_good);
            }

            ++nr_iterations_;
            converged_ = static_cast<bool>((*convergence_criteria_));
        } while(convergence_criteria_->getConvergenceState() ==
            pcl::registration::DefaultConvergenceCriteria<
            Scalar>::CONVERGENCE_CRITERIA_NOT_CONVERGED);

        // 结果输出
        PCL_DEBUG("ICPCuda计算的变换矩阵"
            "为:\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%"
            "5f\t%5f\t%5f\t%5f\n",
            final_transformation_(0,0),
            final_transformation_(0,1),
            final_transformation_(0,2),
            final_transformation_(0,3),
            final_transformation_(1,0),
            final_transformation_(1,1),
            final_transformation_(1,2),
            final_transformation_(1,3),
            final_transformation_(2,0),
            final_transformation_(2,1),
            final_transformation_(2,2),
            final_transformation_(2,3),
            final_transformation_(3,0),
            final_transformation_(3,1),
            final_transformation_(3,2),
            final_transformation_(3,3));


        // 计算源点云配准结果
        pcl::transformPointCloudWithNormals(*input_,output,final_transformation_);


        deinitCompute();
    }


    template<Point3D PointSource,PointWithNormal3D PointTarget,FloatingType Scalar>
    bool IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>::initCompute()
    {
        if(IterativeClosestPointCuda::initCompute())
        {
            if(target_normal_device_.first == nullptr)
            {
                pcl::PointCloud<pcl::Normal> normals_temp;
                pcl::copyPointCloud(*target_,normals_temp);

                pcl::Normal* device_ptr;
                cudaSafeCall(cudaMalloc(&device_ptr,normals_temp.size()*sizeof(pcl::Normal)));
                cudaSafeCall(cudaMemcpy(device_ptr,normals_temp.data(),normals_temp.size()*sizeof(pcl::Normal),cudaMemcpyHostToDevice));
                target_normal_device_ = NormalDevice(getCudaSharedPtr(device_ptr),normals_temp.size());
            }
            return true;
        }

        return false;
    }

    template<Point3D PointSource,PointWithNormal3D PointTarget,FloatingType Scalar>
    void IterativeClosestPointWithNormalsCuda<PointSource,PointTarget,Scalar>::constructTransformationMatrix(const double & alpha,const double & beta,const double & gamma,const double & tx,const double & ty,const double & tz, Matrix4 & transformation_matrix) const
    {
        transformation_matrix = Eigen::Matrix<Scalar,4,4>::Zero();
        transformation_matrix(0,0) = static_cast<Scalar>(std::cos(gamma) * std::cos(beta));
        transformation_matrix(0,1) = static_cast<Scalar>(
            -sin(gamma) * std::cos(alpha) + std::cos(gamma) * sin(beta) * sin(alpha));
        transformation_matrix(0,2) = static_cast<Scalar>(
            sin(gamma) * sin(alpha) + std::cos(gamma) * sin(beta) * std::cos(alpha));
        transformation_matrix(1,0) = static_cast<Scalar>(sin(gamma) * std::cos(beta));
        transformation_matrix(1,1) = static_cast<Scalar>(
            std::cos(gamma) * std::cos(alpha) + sin(gamma) * sin(beta) * sin(alpha));
        transformation_matrix(1,2) = static_cast<Scalar>(
            -std::cos(gamma) * sin(alpha) + sin(gamma) * sin(beta) * std::cos(alpha));
        transformation_matrix(2,0) = static_cast<Scalar>(-sin(beta));
        transformation_matrix(2,1) = static_cast<Scalar>(std::cos(beta) * sin(alpha));
        transformation_matrix(2,2) = static_cast<Scalar>(std::cos(beta) * std::cos(alpha));

        transformation_matrix(0,3) = static_cast<Scalar>(tx);
        transformation_matrix(1,3) = static_cast<Scalar>(ty);
        transformation_matrix(2,3) = static_cast<Scalar>(tz);
        transformation_matrix(3,3) = static_cast<Scalar>(1);
    }
}

