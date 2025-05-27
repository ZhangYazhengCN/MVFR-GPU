#pragma once
#include <tuple>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "octree_cuda.h"
#include "search_cuda.cuh"

namespace mvfr
{
	template<typename PointT>
	void OcTreeCuda<PointT>::nearestKSearch(const CloudDevice& query_cloud_device, unsigned k, IndicesDevice& indices, DistancesDevice& distances) const
	{
		if (k != 1)
		{
			PCL_ERROR("[% s::nearestKSearch] 仅支持k=1的KNN搜索! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return;
		}

		if (query_cloud_device.second < search_in_cuda_th_)
			PCL_WARN("[% s::nearestKSearch] 查询点数量过少，建议在CPU上执行近邻点搜索! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);

		if (!const_cast<OcTreeCuda<PointT>*>(this)->initSearchCuda())
		{
			PCL_ERROR("[% s::nearestKSearch] 近邻点搜索计算初始化失败! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return;
		}

		// 为 indices 和 distances 分配GPU内存
		unsigned elem_size = k * query_cloud_device.second;
		if (!indices.first || indices.second != elem_size)
		{
			using DataType_ = typename IndicesDevice::first_type::element_type;
			DataType_* temp_ptr;
			cudaSafeCall(cudaMalloc(&temp_ptr, elem_size * sizeof(DataType_)));
			indices.first = getCudaSharedPtr(temp_ptr);
			indices.second = elem_size;
		}
		if (!distances.first || distances.second != elem_size)
		{
			using DataType_ = typename DistancesDevice::first_type::element_type;
			DataType_* temp_ptr;
			cudaSafeCall(cudaMalloc(&temp_ptr, elem_size * sizeof(DataType_)));
			distances.first = getCudaSharedPtr(temp_ptr);
			distances.second = elem_size;
		}

		// 执行近邻点搜索
		pcl::gpu::NeighborIndices indices_temp;
		indices_temp.data = decltype(indices_temp.data)(indices.first.get(), indices.second);
		pcl::gpu::Octree::ResultSqrDists distances_temp(distances.first.get(), distances.second);
		tree_->nearestKSearchBatch(
			pcl::gpu::Octree::Queries(query_cloud_device.first.get(), query_cloud_device.second),
			k, indices_temp, distances_temp);

		const_cast<OcTreeCuda<PointT>*>(this)->deinitSearchCuda();
	}

	template<typename PointT>
	void OcTreeCuda<PointT>::radiusSearch(const CloudDevice& query_cloud_device, const double radius, IndicesDevice& indices, DistancesDevice& distances, unsigned max_nn) const
	{
		if (query_cloud_device.second < search_in_cuda_th_)
			PCL_WARN("[% s::radiusSearch] 查询点数量过少，建议在CPU上执行近邻点搜索! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);

		if (!const_cast<OcTreeCuda<PointT>*>(this)->initSearchCuda())
		{
			PCL_ERROR("[% s::radiusSearch] 近邻点搜索计算初始化失败! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return;
		}

		// 为 indices 和 distances 分配GPU内存
		unsigned elem_size = (max_nn == 0 ? cloud_device_.second : max_nn) * query_cloud_device.second;
		if (!indices.first || indices.second != elem_size)
		{
			using DataType_ = typename IndicesDevice::first_type::element_type;
			DataType_* temp_ptr;
			cudaSafeCall(cudaMalloc(&temp_ptr, elem_size * sizeof(DataType_)));
			indices.first = getCudaSharedPtr(temp_ptr);
			indices.second = elem_size;
		}
		if (!distances.first || distances.second != elem_size)
		{
			using DataType_ = typename DistancesDevice::first_type::element_type;
			DataType_* temp_ptr;
			cudaSafeCall(cudaMalloc(&temp_ptr, elem_size * sizeof(DataType_)));
			distances.first = getCudaSharedPtr(temp_ptr);
			distances.second = elem_size;
		}

		//// 将索引初始化为默认值 -1
		using DataType_ = IndicesDevice::first_type::element_type;
		thrust::host_vector<DataType_> init_use_temp(indices.second, -1);
		cudaSafeCall(cudaMemcpy(indices.first.get(), init_use_temp.data(), indices.second * sizeof(DataType_), cudaMemcpyHostToDevice));

		// 执行近邻点搜索
		pcl::gpu::NeighborIndices indices_temp;
		indices_temp.data = decltype(indices_temp.data)(indices.first.get(), indices.second);
		tree_->radiusSearch(
			pcl::gpu::Octree::Queries(query_cloud_device.first.get(), query_cloud_device.second),
			radius, max_nn, indices_temp);

		calcuDistances(query_cloud_device.first.get(), cloud_device_.first.get(),
			indices_temp.data, indices_temp.sizes, distances.first.get(),
			query_cloud_device.second, (max_nn == 0 ? cloud_device_.second : max_nn), sorted_results_);

		const_cast<OcTreeCuda<PointT>*>(this)->deinitSearchCuda();
	}

	template<typename PointT>
	void OcTreeCuda<PointT>::approxNearestSearch(const CloudDevice& query_cloud_device, IndicesDevice& indices, DistancesDevice& distances) const
	{
		if (query_cloud_device.second < search_in_cuda_th_)
			PCL_WARN("[% s::approxNearestSearch] 查询点数量过少，建议在CPU上执行近邻点搜索! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);

		if (!const_cast<OcTreeCuda<PointT>*>(this)->initSearchCuda())
		{
			PCL_ERROR("[% s::approxNearestSearch] 近邻点搜索计算初始化失败! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return;
		}

		// 为 indices 和 distances 分配GPU内存
		unsigned elem_size = query_cloud_device.second;
		if (!indices.first || indices.second != elem_size)
		{
			using DataType_ = typename IndicesDevice::first_type::element_type;
			DataType_* temp_ptr;
			cudaSafeCall(cudaMalloc(&temp_ptr, elem_size * sizeof(DataType_)));
			indices.first = getCudaSharedPtr(temp_ptr);
			indices.second = elem_size;
		}
		if (!distances.first || distances.second != elem_size)
		{
			using DataType_ = typename DistancesDevice::first_type::element_type;
			DataType_* temp_ptr;
			cudaSafeCall(cudaMalloc(&temp_ptr, elem_size * sizeof(DataType_)));
			distances.first = getCudaSharedPtr(temp_ptr);
			distances.second = elem_size;
		}

		// 执行 approxNearestSearch
		pcl::gpu::NeighborIndices indices_temp;
		indices_temp.data = decltype(indices_temp.data)(indices.first.get(), indices.second);
		pcl::gpu::Octree::ResultSqrDists distances_temp(distances.first.get(), distances.second);
		tree_->approxNearestSearch(pcl::gpu::Octree::Queries(query_cloud_device.first.get(), query_cloud_device.second),
			indices_temp, distances_temp);

		const_cast<OcTreeCuda<PointT>*>(this)->deinitSearchCuda();
	}


	template<typename PointT>
	bool OcTreeCuda<PointT>::initCompute(void)
	{
		if (!tree_)
		{
			PCL_ERROR("[% s::initCompute] 搜索树为空! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return false;
		}

		// 输入点云已更新，重新建树
		if (cloud_update_)
		{
			cloud_device_octree_ = pcl::gpu::Octree::PointCloud(cloud_device_.first.get(), cloud_device_.second);
			tree_->setCloud(cloud_device_octree_);
			tree_->build();
		}
		return true;
	}
}