#pragma once


#include "icp_cuda.h"
#include "registration_cuda.cuh"

namespace mvfr
{
	template<typename PointSource, typename PointTarget, typename Scalar>
	void IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::setInputSource(const PointCloudSourceConstPtr& cloud)
	{
		if (cloud->points.empty())
		{
			PCL_ERROR("[% s::setInputSource] 输入源点云为空! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return;
		}
		pcl::PCLBase<PointSource>::setInputCloud(cloud);
		source_device_ = CloudDevice(nullptr, 0);
		source_cloud_updated_ = true;
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	void IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::setInputSourceDevice(const PointCloudSourceConstPtr& cloud, const CloudDevice& cloud_device)
	{
		if (cloud->points.empty() || cloud_device.first == nullptr || cloud->size() != cloud_device.second)
		{
			PCL_ERROR("[% s::setInputSourceDevice] 输入源点云为空或CPU与GPU点云不匹配! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return;
		}
		pcl::PCLBase<PointSource>::setInputCloud(cloud);
		source_device_ = cloud_device;
		source_cloud_updated_ = true;
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	void IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::setInputTarget(const PointCloudTargetConstPtr& cloud)
	{
		if (cloud->points.empty())
		{
			PCL_ERROR("[% s::setInputTarget] 输入目标点云为空! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return;
		}
		target_ = cloud;
		target_device_ = CloudDevice(nullptr, 0);
		target_cloud_updated_ = true;
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	void IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::setInputTargetDevice(const PointCloudTargetConstPtr& cloud, const CloudDevice& cloud_device)
	{
		if (cloud->points.empty() || cloud_device.first == nullptr || cloud->size() != cloud_device.second)
		{
			PCL_ERROR("[% s::setInputTargetDevice] 输入目标点云为空! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return;
		}
		target_ = cloud;
		target_device_ = cloud_device;
		target_cloud_updated_ = true;
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	std::pair<double, double> IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::getFitnessScore(const Matrix4& ground_th)
	{
		if (!converged_)
		{
			PCL_ERROR("[% s::getFitnessScore] ICPCuda 未计算或迭代未收敛，无法计算拟合分数! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return std::make_pair(0.0, 0.0);
		}

		Eigen::Transform<Scalar, 3, Eigen::Isometry> calcu_trans(final_transformation_);
		Eigen::Transform<Scalar, 3, Eigen::Isometry> indeed_trans(ground_th);

		return { std::acos(((indeed_trans.rotation() * calcu_trans.rotation().inverse()).trace() - 1) / 2),
			(indeed_trans.translation() - calcu_trans.translation()).norm() };
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	double IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::getFitnessScore(double max_range)
	{
		// 判断ICP迭代是否收敛
		if (!converged_)
		{
			PCL_ERROR("[% s::getFitnessScore] ICPCuda 未计算或迭代未收敛，无法计算拟合分数! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return 0.0;
		}

		double fitness_score = 0.0;
		for (const auto& corr : *correspondences_)
		{
			if (corr.distance < +max_range)
				fitness_score += corr.distance;
		}
		return fitness_score / correspondences_->size();
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	void IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::align(PointCloudSource& output)
	{
		align(output, Matrix4::Identity());
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	void IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::align(PointCloudSource& output, const Matrix4& guess)
	{
		if (!initCompute())
			return;

		// 重置参数
		aligned_cloud_device_ = CloudDevice(nullptr, 0);
		nr_iterations_ = 0;
		correspondences_device_ = CorrespondencesDevice(nullptr, nullptr, nullptr, 0);
		transformation_ = previous_transformation_ = Matrix4::Identity();
		final_transformation_ = guess;
		converged_ = false;
		convergence_criteria_->setMaximumIterations(max_iterations_);
		convergence_criteria_->setRelativeMSE(euclidean_fitness_epsilon_);
		convergence_criteria_->setTranslationThreshold(transformation_epsilon_);
		if (transformation_rotation_epsilon_ > 0)
			convergence_criteria_->setRotationThreshold(transformation_rotation_epsilon_);

		// 初始化设备源点云 aligned_cloud_device_
		pcl::PointXYZ* cloud_device_temp_ptr;
		cudaSafeCall(cudaMalloc(&cloud_device_temp_ptr, source_device_.second * sizeof(pcl::PointXYZ)));
		cudaSafeCall(cudaMemcpy(cloud_device_temp_ptr, source_device_.first.get(), source_device_.second * sizeof(pcl::PointXYZ), cudaMemcpyDeviceToDevice));
		aligned_cloud_device_.first = getCudaSharedPtr(cloud_device_temp_ptr);
		aligned_cloud_device_.second = source_device_.second;
		if (guess != Matrix4::Identity())		// 若初始位姿不为单位矩阵，则对源点云进行变换
			transformCloudDevice(aligned_cloud_device_.first.get(), aligned_cloud_device_.second, guess);

		// 分配近邻点搜索索引、近邻点距离、对应关系GPU内存
		pcl::index_t* indices_device_temp_ptr;
		cudaSafeCall(cudaMalloc(&indices_device_temp_ptr, source_device_.second * sizeof(pcl::index_t)));
		IndicesDevice indices_device(getCudaSharedPtr(indices_device_temp_ptr), source_device_.second);

		cudaSafeCall(cudaMalloc(&indices_device_temp_ptr, source_device_.second * sizeof(pcl::index_t)));
		std::get<0>(correspondences_device_) = getCudaSharedPtr(indices_device_temp_ptr);

		cudaSafeCall(cudaMalloc(&indices_device_temp_ptr, source_device_.second * sizeof(pcl::index_t)));
		std::get<1>(correspondences_device_) = getCudaSharedPtr(indices_device_temp_ptr);


		float* distances_device_temp_ptr;
		cudaSafeCall(cudaMalloc(&distances_device_temp_ptr, source_device_.second * sizeof(float)));
		DistancesDevice distances_device(getCudaSharedPtr(distances_device_temp_ptr), source_device_.second);

		cudaSafeCall(cudaMalloc(&distances_device_temp_ptr, source_device_.second * sizeof(float)));
		std::get<2>(correspondences_device_) = getCudaSharedPtr(distances_device_temp_ptr);
		std::get<3>(correspondences_device_) = source_device_.second;

		// 分配主机对应关系CPU内存
		std::shared_ptr<pcl::index_t> corr_src_host_ptr(new pcl::index_t[std::get<3>(correspondences_device_)], std::default_delete<pcl::index_t[]>());
		std::shared_ptr<pcl::index_t> corr_tgt_host_ptr(new pcl::index_t[std::get<3>(correspondences_device_)], std::default_delete<pcl::index_t[]>());
		std::shared_ptr<float> corr_dis_host_ptr(new float[std::get<3>(correspondences_device_)], std::default_delete<float[]>());


		// 迭代直至收敛
		do {
			// 保存上次迭代的计算的变换矩阵
			previous_transformation_ = transformation_;

			// 计算对应关系
			tree_->approxNearestSearch(aligned_cloud_device_, indices_device, distances_device);
			//tree_->nearestKSearch(aligned_cloud_device_, 1, indices_device, distances_device);
			const unsigned corr_num = correspondencesDeviceRejector(indices_device, distances_device,
				correspondences_device_, corr_dist_threshold_ * corr_dist_threshold_);

			// 检查对应关系是否充足
			if (corr_num < min_number_correspondences_) {
				PCL_ERROR("[% s::align] 源点云与目标点云对应关系不足! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				convergence_criteria_->setConvergenceState(pcl::registration::DefaultConvergenceCriteria<Scalar>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES);
				converged_ = false;
				break;
			}

			/// @todo 异步更新  correspoondences_ 和  update_visualizer
			// 更新 correspoondences_
			cudaSafeCall(cudaMemcpy(corr_src_host_ptr.get(), std::get<0>(correspondences_device_).get(), corr_num * sizeof(pcl::index_t), cudaMemcpyDeviceToHost));
			cudaSafeCall(cudaMemcpy(corr_tgt_host_ptr.get(), std::get<1>(correspondences_device_).get(), corr_num * sizeof(pcl::index_t), cudaMemcpyDeviceToHost));
			cudaSafeCall(cudaMemcpy(corr_dis_host_ptr.get(), std::get<2>(correspondences_device_).get(), corr_num * sizeof(float), cudaMemcpyDeviceToHost));

			correspondences_->resize(corr_num);
			for (int i = 0; i < corr_num; ++i)
				(*correspondences_)[i] = pcl::Correspondence(corr_src_host_ptr.get()[i], corr_tgt_host_ptr.get()[i], corr_dis_host_ptr.get()[i]);

			//// 更新PCLVisualizer可视化回调函数
			//if (update_visualizer_ != nullptr) {
			//	pcl::Indices source_indices_good, target_indices_good;
			//	for (const Correspondence& corr : *correspondences_) {
			//		source_indices_good.emplace_back(corr.index_query);
			//		target_indices_good.emplace_back(corr.index_match);
			//	}
			//	update_visualizer_(
			//		*input_transformed, source_indices_good, *target_, target_indices_good);
			//}


			// 计算 correlation matrix H = corr_src_points * corr_tgt_points'
			Eigen::Matrix<Scalar, 4, 1> centroid_src, centroid_tgt;
			Eigen::Matrix<Scalar, 3, 3> H;
			computeHmatrixDevice(aligned_cloud_device_, target_device_, correspondences_device_, corr_num, centroid_src, centroid_tgt, H);

			// 对H矩阵进行奇异值分解
			Eigen::JacobiSVD<Eigen::Matrix<Scalar, 3, 3>> svd(
				H, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix<Scalar, 3, 3> u = svd.matrixU();
			Eigen::Matrix<Scalar, 3, 3> v = svd.matrixV();

			// 计算旋转矩阵R
			if (u.determinant() * v.determinant() < 0) {
				for (int x = 0; x < 3; ++x)
					v(x, 2) *= -1;
			}
			Eigen::Matrix<Scalar, 3, 3> R = v * u.transpose();

			// 生成本次迭代计算的变换矩阵 transformation_
			transformation_.topLeftCorner(3, 3) = R;
			const Eigen::Matrix<Scalar, 3, 1> Rc(R * centroid_src.template head<3>());
			transformation_.template block<3, 1>(0, 3) = centroid_tgt.template head<3>() - Rc;

			// 基于 transformation_ 更新GPU源点云 aligned_cloud_device_
			transformCloudDevice(aligned_cloud_device_.first.get(), aligned_cloud_device_.second, transformation_);

			// 更新final_transformation_ (transformation是基于全局坐标系求得的，所以用左乘)
			final_transformation_ = transformation_ * final_transformation_;

			++nr_iterations_;
			converged_ = static_cast<bool>((*convergence_criteria_));
		} while (convergence_criteria_->getConvergenceState() ==
			pcl::registration::DefaultConvergenceCriteria<
			Scalar>::CONVERGENCE_CRITERIA_NOT_CONVERGED);


		// 结果输出
		PCL_DEBUG("ICPCuda计算的变换矩阵"
			"为:\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%"
			"5f\t%5f\t%5f\t%5f\n",
			final_transformation_(0, 0),
			final_transformation_(0, 1),
			final_transformation_(0, 2),
			final_transformation_(0, 3),
			final_transformation_(1, 0),
			final_transformation_(1, 1),
			final_transformation_(1, 2),
			final_transformation_(1, 3),
			final_transformation_(2, 0),
			final_transformation_(2, 1),
			final_transformation_(2, 2),
			final_transformation_(2, 3),
			final_transformation_(3, 0),
			final_transformation_(3, 1),
			final_transformation_(3, 2),
			final_transformation_(3, 3));


		// 计算CPU配准源点云
		pcl::copyPointCloud(*input_, output);		// 复制源点云至输出点云（配准后点云）,主要是为了获得非xyz字段信息
		pcl::PointCloud<pcl::PointXYZ> cloud_temp(input_->size(), 1);	// 初始化临时点云，用来获取GPU中变换后的源点云的xyz字段信息
		cudaSafeCall(cudaMemcpy(cloud_temp.data(), aligned_cloud_device_.first.get(),
			aligned_cloud_device_.second * sizeof(pcl::PointXYZ), cudaMemcpyDeviceToHost));
		pcl::concatenateFields(output, cloud_temp, output);


		deinitCompute();
	}

	template<typename PointSource, typename PointTarget, typename Scalar>
	bool IterativeClosestPointCuda<PointSource, PointTarget, Scalar>::initCompute()
	{
		if (!target_)
		{
			PCL_ERROR("[% s::initCompute] 未给定目标点云! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return (false);
		}

		if (!pcl::PCLBase<PointSource>::initCompute())
			return false;


		// 更新GPU源点云
		if (source_cloud_updated_)
		{
			if (!source_device_.first)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp;
				if constexpr (std::is_same_v<PointCloudSource, pcl::PointXYZ>)
					cloud_temp = input_;
				else
				{
					cloud_temp.reset(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::copyPointCloud(*input_, *cloud_temp);
				}

				pcl::PointXYZ* device_ptr;
				cudaSafeCall(cudaMalloc(&device_ptr, cloud_temp->size() * sizeof(pcl::PointXYZ)));
				cudaSafeCall(cudaMemcpy(device_ptr, cloud_temp->data(), cloud_temp->size() * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
				source_device_.first = getCudaSharedPtr(device_ptr);
				source_device_.second = cloud_temp->size();
			}
			source_cloud_updated_ = false;
		}

		// 更新目标点云搜索树及GPU目标点云
		if (target_cloud_updated_)
		{
			if (!target_device_.first)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp;
				if constexpr (std::is_same_v<PointCloudSource, pcl::PointXYZ>)
					cloud_temp = target_;
				else
				{
					cloud_temp.reset(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::copyPointCloud(*target_, *cloud_temp);
				}

				pcl::PointXYZ* device_ptr;
				cudaSafeCall(cudaMalloc(&device_ptr, cloud_temp->size() * sizeof(pcl::PointXYZ)));
				cudaSafeCall(cudaMemcpy(device_ptr, cloud_temp->data(), cloud_temp->size() * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
				target_device_.first = getCudaSharedPtr(device_ptr);
				target_device_.second = cloud_temp->size();
			}

			if (!force_no_recompute_)
				tree_->setInputCloud(target_device_);

			target_cloud_updated_ = false;
		}

		return true;
	}
}
