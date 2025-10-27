/**
 *
 *  @file      utilities.h
 *  @brief     Some non-core but useful functions
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <numbers>
#include <cmath>
#include <filesystem>
#include <utility>
#include <random>
#include <functional>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/filter.h>

#include "types.h"

namespace mvfr
{
	namespace fs = std::filesystem;

	// ---------------------------------------------------------------------------------------------------
	// ---------------------------------------- 文件加载 --------------------------------------------------
	// ---------------------------------------------------------------------------------------------------

	/**
	 *  @brief  从给定路径 \c file_path 中读取点云文件.
	 *  @tparam PointT    点云数据格式
	 *  @param  file_path 文件路径
	 *  @param  cloud     点云
	 *  @retval           0 读取成功，<0 读取失败
	 */
	template<typename PointT>
	int loadCloud(const std::string& file_path, typename pcl::PointCloud<PointT>::Ptr& cloud)
	{
		// 判断路径是否指向文件
		fs::path path_temp(file_path);
		if (!fs::is_regular_file(path_temp))
		{
			PCL_ERROR("[loadcloud] 给定路径(%s)没有指向文件! %s(%d)\n", file_path.c_str(), __FILE__, __LINE__);
			return -1;
		}

		// 初始化
		int status = 0;
		if (!cloud)
			cloud.reset(new pcl::PointCloud<PointT>);
		else
			cloud->clear();

		// 根据文件类型读取点云
		if (path_temp.extension().string() == ".txt")
		{
			std::ifstream ifs(file_path, std::ios::in);
			if (!ifs.is_open())		// 判断文件是否可以正常打开
			{
				PCL_ERROR("[loadcloud] 无法打开文件(%s)! %s(%d)\n", file_path.c_str(), __FILE__, __LINE__);
				return -1;
			}
			while (!ifs.eof())	// 读取点云数据
			{
				PointT* clouldtemp(new PointT);
				ifs >> clouldtemp->x >> clouldtemp->y >> clouldtemp->z;
				if (clouldtemp->x == clouldtemp->y && clouldtemp->x == clouldtemp->z)
				{
					continue;
				}
				cloud->push_back(*clouldtemp);
				delete clouldtemp;
			}
			ifs.close();             //关闭文件输入流
			// 去除点云中的无穷值并初始化点云头
			pcl::Indices indices_temp;
			pcl::removeNaNFromPointCloud(*cloud, *cloud, indices_temp);
			cloud->width = cloud->points.size();
			cloud->height = 1;
			cloud->is_dense = true;
		}
		else if (path_temp.extension().string() == ".pcd")
			status = pcl::io::loadPCDFile<PointT>(file_path, *cloud);
		else if (path_temp.extension().string() == ".ply")
			status = pcl::io::loadPLYFile<PointT>(file_path, *cloud);
		else
		{
			PCL_ERROR("[loadcloud] 扩展名为%s的目标文件(%s)未定义读取方式! %s(%d)\n",
				path_temp.extension().string().c_str(), file_path.c_str(), __FILE__, __LINE__);
			status = -1;
		}

		return status;
	}


	/**
	 *  @brief  读取多个点云文件
	 *  @tparam PointT     点云数据格式
	 *  @param  file_paths 文件路径集
	 *  @param  clouds     点云集
	 *
	 *  @note 输入的路径数量应大于等于2
	 */
	template<typename PointT>
	void loadClouds(const std::vector<std::string>& file_paths, std::vector<typename pcl::PointCloud<PointT>::Ptr>& clouds)
	{
		if (file_paths.size() < 2)
		{
			PCL_ERROR("[loadClouds] 文件路径路径过少! %s(%d)\n", __FILE__, __LINE__);
			return;
		}

		clouds.resize(file_paths.size());
		for (auto& cloud_ : clouds)
			cloud_.reset(new pcl::PointCloud<PointT>);

		for (int i = 0; i < file_paths.size(); ++i)
			loadCloud<PointT>(file_paths[i], clouds[i]);
	}


	// ---------------------------------------------------------------------------------------------------
	// -------------------------------------------- 其它 --------------------------------------------------
	// ---------------------------------------------------------------------------------------------------

	/**
	 *  @brief  Generate a rotaion matrix(homogeneous) randomly.
	 *  @tparam Scalar          data type
	 *  @param  angle_magnitude The rotation angle(degree) is limited to [-angle_magnitude, angle_magnitude)
	 *  @param  rng             Random Number Generator
	 *  @param  fixed_angle		Only randomly generate axis if true, the rotation angle is \c angle_magnitude
	 *  @retval                 Rotation Matrix
	 */
	template<FloatingType Scalar, typename RNG = std::default_random_engine>
		requires requires{requires std::uniform_random_bit_generator<std::decay_t<RNG>>; }
	Eigen::Matrix4<Scalar> createSO3Transformation(
		const Scalar angle_magnitude,
		RNG&& rng = std::default_random_engine{
			static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
		},
		bool fixed_angle = false)
	{
		Eigen::Matrix4<Scalar> rotation = Eigen::Matrix4<Scalar>::Identity();
		auto uniform_real_dis = std::bind(
			std::uniform_real<Scalar>(-1, 1),
			std::ref(rng));

		// Generate a random 3D unit vector with uniform spherical distribution
		// inspired by Generate a random 3D unit vector with uniform spherical distribution
		auto phi = (uniform_real_dis() + 1) * std::numbers::pi_v<Scalar>;
		auto theta = acos(uniform_real_dis());
		auto x = sin(theta) * cos(phi);
		auto y = sin(theta) * sin(phi);
		auto z = cos(theta);
		Eigen::Vector3<Scalar> rot_axis = Eigen::Vector3<Scalar>(x, y, z).normalized();

		if (fixed_angle)
		{
			rotation.topLeftCorner<3, 3>() = Eigen::AngleAxis<Scalar>(
				angle_magnitude / 180 * std::numbers::pi_v<Scalar>,
				rot_axis).matrix();
		}
		else
		{
			rotation.topLeftCorner<3, 3>() = Eigen::AngleAxis<Scalar>(
				uniform_real_dis() * angle_magnitude / 180 * std::numbers::pi_v<Scalar>,
				rot_axis).matrix();
		}
		return rotation;
	}

	/**
	 *  @brief  Generate a iosmetry matrix randomly
	 *  @tparam Scalar          data type
	 *  @param  rotation_mag    The rotation angle(degree) is limited to [-angle_magnitude, angle_magnitude)
	 *  @param  translation_mag The translation along a certain axis is limited to [0, translation_mag)
	 *  @param  fixed_mag		Use fixed rotation angle( \c rotation_mag ) and translation magnitude( \c translation_mag ), if true
	 *  @param  rng             Random Number Generator
	 *  @retval                 Isometry Matrix
	 */
	template<FloatingType Scalar, typename RNG = std::default_random_engine>
		requires requires{requires std::uniform_random_bit_generator<std::decay_t<RNG>>; }
	Eigen::Matrix4<Scalar> createSE3Transformation(
		const Scalar rotation_mag, const Scalar translation_mag,
		RNG&& rng = std::default_random_engine{ 
			static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) 
		},
		bool fixed_mag = false)
	{
		Eigen::Matrix4<Scalar> transformation = createSO3Transformation(rotation_mag, rng, fixed_mag);

		auto random_real_dis = std::bind(
			std::uniform_real<Scalar>(-1, 1),
			std::ref(rng));

		Eigen::Vector3<Scalar> translate_direction =
			Eigen::Vector3<Scalar>(random_real_dis(), random_real_dis(), random_real_dis()).normalized();

		
		if (fixed_mag)
		{
			transformation.topRightCorner<3, 1>() = translate_direction * translation_mag;
		}
		else
		{
			transformation.topRightCorner<3, 1>() = translate_direction * (((random_real_dis() + 1) / 2) * translation_mag);
		}
		return transformation;
	}


	/// 计算给定点云内点的平均间距
	template<typename PointT>
	double calcuCloudAvgInterval(const typename pcl::PointCloud<PointT>::ConstPtr& cloud)
	{
		std::vector<pcl::Indices> indices_temp;
		std::vector<std::vector<float>> distances_temp;

		pcl::search::KdTree<PointT> tree;
		tree.setInputCloud(cloud);
		tree.nearestKSearch(*cloud, pcl::Indices(), 2, indices_temp, distances_temp);

		double dis = 0.0;
		for (const auto& d : distances_temp)
			dis += d[1];
		return std::sqrt(dis / cloud->size());
	}


	/// 计算给定点云组内每片点云的平均间距
	template<typename PointT>
	void calcuCloudAvgInterval(const std::vector<typename pcl::PointCloud<PointT>::Ptr>& clouds, std::vector<double>& intervals)
	{
		intervals.clear();
		for (const auto& cloud : clouds)
			intervals.push_back(calcuCloudAvgInterval<PointT>(cloud));
	}


	/**
	 *  @brief  比较计算得到的变化矩阵 \c cal 与实际变换矩阵 \c gth 间的差异
	 *  @tparam Scalar 数据类型
	 *  @param  gth    实际变换
	 *  @param  cal    计算变换
	 *  @retval        (旋转误差, 平移误差) == (arccos((tr(Rg*Rc')-1)/2), ||tg - tc||). 其中Rg, Rc分别为实际旋转矩阵与计算得到的旋转矩阵, tg, tc为实际平移向量与计算平移向量
	 */
	template<typename Scalar>
	std::pair<double, double> calcuIsometry3DError(const Eigen::Matrix<Scalar, 4, 4>& gth, const Eigen::Matrix<Scalar, 4, 4>& cal)
	{
		Eigen::Transform<Scalar, 3, Eigen::Isometry> calcu_trans(cal);
		Eigen::Transform<Scalar, 3, Eigen::Isometry> indeed_trans(gth);

		return { std::acos(((indeed_trans.rotation() * calcu_trans.rotation().transpose()).trace() - 1) / 2),
			(indeed_trans.translation() - calcu_trans.translation()).norm() };
	}


	/**
	 *  @brief	批量处理点云数据
	 *
	 *	@details 基于处理函数 \c func 及函数参数 \c args 批量处理原始点云 \c raw_clouds，最后将结果保存至 \c processed_clouds 内.
	 *	参考示例：
	 *	@code
	 *	// 声明并加载原始点云 raw_clouds
	 *  std::vector<pcl::PointXYZ> raw_clouds, process_clouds;
	 *	// 批量执行半径滤波 radiusOutlierRemoval（注意传入模板实参）
	 *	processCloudsBatch<pcl::PointXYZ>(rawclouds, process_clouds, radiusOutlierRemoval<pcl::PointXYZ>, 5.0, 50);
	 *	@endcode
	 *
	 *  @tparam PointT           点数据类型
	 *  @tparam F                处理函数类型
	 *  @tparam Args             函数参数类型
	 *  @param  raw_clouds       原始点云
	 *  @param  processed_clouds 处理后点云
	 *  @param  func             处理函数
	 *  @param  args             函数参数
	 *  @retval                  true则成功全部点云；否则，则存在未处理成功的点云，或没有输入点云
	 *
	 *	@note 处理函数 \c func 的声明应为 <b>template<typename PointT> bool func(const pcl::PointCloud<PointT>::Ptr&, pcl::PointCloud<PointT>::Ptr&, Args...);<\b>
	 */
	template<typename PointT, typename F, typename... Args>
	bool processCloudsBatch(const std::vector<typename pcl::PointCloud<PointT>::Ptr>& raw_clouds,
		std::vector<typename pcl::PointCloud<PointT>::Ptr>& processed_clouds, F func, Args... args)
	{
		if (raw_clouds.size() == 0)
		{
			processed_clouds.clear();
			return false;
		}

		bool status = true;
		processed_clouds.resize(raw_clouds.size());

		for (int i = 0; i < raw_clouds.size(); ++i)
		{
			processed_clouds[i].reset(new pcl::PointCloud<PointT>);
			status &= func(raw_clouds[i], processed_clouds[i], std::forward<Args>(args)...);
		}

		return status;
	}

}
