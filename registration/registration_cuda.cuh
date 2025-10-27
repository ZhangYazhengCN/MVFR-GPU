/**
 *
 *  @file      multiview_registration.h
 *  @brief     some functions that need to be compiled by nvcc and used in point cloud fine registration
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/types.h>

#include "../utilities/types.h"


#ifdef MVFR_GPU_EXPORTS
#define MVFR_GPU_API __declspec(dllexport)
#else
#define MVFR_GPU_API __declspec(dllimport)
#endif


namespace mvfr
{
	/**
	 *  @brief  根据变换矩阵 \c trans 对 \c cloud_device_ptr 指向的大小为 \c cloud_device_size GPU点云进行坐标变换
	 *  @tparam Scalar 数据类型
	 *  @tparam StorageOrder trans矩阵内存顺序
	 *  @param[in out]  cloud_device_ptr 设备点云指针
	 *  @param[in]  cloud_device_size 设备点云大小
	 *  @param[in]  trans 变换矩阵
	 */
	template<typename Scalar = double, int StorageOrder = Eigen::ColMajor>
	void transformCloudDevice(pcl::PointXYZ* const cloud_device_ptr, const std::size_t cloud_device_size, const Eigen::Matrix<Scalar, 4, 4, StorageOrder>& trans);

	/**
	 *  @brief  基于距离阈值 \c th 对计算的对应关系进行筛选
	 *  @param[in]  indices_device   源点云在目标点云中的近邻点索引
	 *  @param[in]  distances_device 源点云在目标点云中的近邻点距离的平方
	 *  @param[out] corr_device      保存筛选后的对应关系
	 *  @param[in]  th               距离平方的阈值
	 *  @retval						 筛选后对应关系的数量
	 */
    MVFR_GPU_API unsigned correspondencesDeviceRejector(const IndicesDevice& indices_device,const DistancesDevice& distances_device,
        CorrespondencesDevice& corr_device,const float th);

	/**
	 *  @brief  基于源点云与目标点云的对应关系计算H矩阵，\f(H = (source-centroid_{src})[corr_{src}]*((target-centroid_{tgt})[corr_{tgt}])^{T}\f)
	 *  @tparam Scalar        数据类型
	 *  @tparam StorageOrder  H矩阵的内存顺序
	 *  @param[in]  source_device GPU源点云
	 *  @param[in]  target_device GPU目标点云
	 *  @param[in]  corr_device   源点云与目标点云对应关系
	 *  @param[in]  valid_corr    对应关系的数量
	 *  @param[out] centroid_src  源点云质心
	 *  @param[out] centroid_tgt  目标点云质心
	 *  @param[out] H             H矩阵
	 */
	template<typename Scalar = double, int StorageOrder = Eigen::ColMajor>
	void computeHmatrixDevice(const CloudDevice& source_device, const CloudDevice& target_device,
		const CorrespondencesDevice& corr_device, const unsigned valid_corr,
		Eigen::Matrix<Scalar, 4, 1>& centroid_src,
		Eigen::Matrix<Scalar, 4, 1>& centroid_tgt,
		Eigen::Matrix<Scalar, 3, 3, StorageOrder>& H);

}

#undef MVFR_GPU_API
