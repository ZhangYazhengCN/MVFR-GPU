#pragma once
#include "search_cuda.h"

#include <pcl/common/io.h>

#include "search_cuda.cuh"
#include "../utilities/cuda_utilities.h"

namespace mvfr
{
	template<Point3D PointT>
	bool SearchCuda<PointT>::setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices)
	{
		if (!cloud || cloud->empty())
		{
			PCL_ERROR("[% s::setInputCloud] 输入主机点云为空! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return false;
		}
		input_ = cloud;
		indices_ = indices;
		cloud_device_.first.reset();
		cloud_device_.second = 0;

		cloud_update_ = true;
		return true;
	}

	template<Point3D PointT>
	bool SearchCuda<PointT>::setInputCloud(const CloudDevice& cloud_device)
	{
		if (!cloud_device.first || cloud_device.second == 0)
		{
			PCL_ERROR("[% s::setInputCloud] 输入设备点云指针为空或大小为0! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return false;
		}
		cloud_device_ = cloud_device;
		input_.reset();
		indices_.reset();

		cloud_update_ = true;
		return true;
	}

	template<Point3D PointT>
	bool SearchCuda<PointT>::setInputCloud(const PointCloudConstPtr& cloud, const CloudDevice& cloud_device, const IndicesConstPtr& indices)
	{
		if ((!cloud || cloud->empty()) || (!cloud_device.first || cloud_device.second == 0))
		{
			PCL_ERROR("[% s::setInputCloud] 输入点云为空! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return false;
		}

		if (cloud->size() != cloud_device.second)
		{
			PCL_ERROR("[% s::setInputCloud] 设备点云与主机点云大小不匹配! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return false;
		}

		input_ = cloud;
		indices_ = indices;
		cloud_device_ = cloud_device;

		cloud_update_ = true;
		return true;
	}

	template<Point3D PointT>
	int SearchCuda<PointT>::nearestKSearch(const PointT& point, int k, pcl::Indices& k_indices, std::vector<float>& k_sqr_distances) const
	{
		if (!const_cast<SearchCuda<PointT>*>(this)->initSearchCuda())
		{
			PCL_ERROR("[% s::nearestKSearch] 初始化失败! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return -1;
		}

		// 分配GPU内存
		pcl::PointXYZ* point_device_ptr;
		pcl::index_t* indices_device_ptr;
		float* distances_device_ptr;

		cudaSafeCall(cudaMalloc(&point_device_ptr, sizeof(pcl::PointXYZ)));
		if constexpr (std::is_same_v<PointT, pcl::PointXYZ>)
			cudaSafeCall(cudaMemcpy(point_device_ptr, &point, sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
		else
		{
			pcl::PointXYZ point_temp(point.x, point.y, point.z);
			cudaSafeCall(cudaMemcpy(point_device_ptr, &point_temp, sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
		}
		CloudDevice cloud_device(getCudaSharedPtr(point_device_ptr), 1);

		cudaSafeCall(cudaMalloc(&indices_device_ptr, k * sizeof(pcl::index_t)));
		cudaSafeCall(cudaMalloc(&distances_device_ptr, k * sizeof(float)));
		IndicesDevice indices_device(getCudaSharedPtr(indices_device_ptr), k);
		DistancesDevice disatnces_device(getCudaSharedPtr(distances_device_ptr), k);

		// 计算近邻点
		nearestKSearch(cloud_device, k, indices_device, disatnces_device);

		// 生成近邻索引与距离
		moveIndicesAndDistances(indices_device.first.get(), disatnces_device.first.get(), indices_device.second, k_indices, k_sqr_distances);

		if (const_cast<SearchCuda<PointT>*>(this)->deinitSearchCuda()) return k_indices.size(); else return -1;
	}

	template<Point3D PointT>
	void SearchCuda<PointT>::nearestKSearch(const PointCloud& cloud, const pcl::Indices& indices, int k, std::vector<pcl::Indices>& k_indices, std::vector<std::vector<float>>& k_sqr_distances) const
	{
		if (!const_cast<SearchCuda<PointT>*>(this)->initSearchCuda())
		{
			PCL_ERROR("[% s::nearestKSearch] 初始化失败! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return;
		}

		pcl::PointCloud<pcl::PointXYZ>* cloud_temp_ptr;
		if constexpr (std::is_same_v<PointT, pcl::PointXYZ>)
		{
			if (indices.empty())
				cloud_temp_ptr = const_cast<PointCloud*>(&cloud);
			else
			{
				cloud_temp_ptr = new pcl::PointCloud<pcl::PointXYZ>;
				pcl::copyPointCloud(cloud, indices, *cloud_temp_ptr);
			}
		}
		else
		{
			cloud_temp_ptr = new pcl::PointCloud<pcl::PointXYZ>;
			if (indices.empty())
				pcl::copyPointCloud(cloud, *cloud_temp_ptr);
			else
				pcl::copyPointCloud(cloud, indices, *cloud_temp_ptr);
		}


		// 分配GPU内存
		pcl::PointXYZ* cloud_device_ptr;
		pcl::index_t* indices_device_ptr;
		float* distances_device_ptr;

		cudaSafeCall(cudaMalloc(&cloud_device_ptr, cloud_temp_ptr->size() * sizeof(pcl::PointXYZ)));
		cudaSafeCall(cudaMemcpy(cloud_device_ptr, cloud_temp_ptr->data(), cloud_temp_ptr->size() * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
		CloudDevice cloud_device(getCudaSharedPtr(cloud_device_ptr), cloud_temp_ptr->size());

		cudaSafeCall(cudaMalloc(&indices_device_ptr, cloud_temp_ptr->size() * k * sizeof(pcl::index_t)));
		cudaSafeCall(cudaMalloc(&distances_device_ptr, cloud_temp_ptr->size() * k * sizeof(float)));
		IndicesDevice indices_device(getCudaSharedPtr(indices_device_ptr), cloud_temp_ptr->size() * k);
		DistancesDevice disatnces_device(getCudaSharedPtr(distances_device_ptr), cloud_temp_ptr->size() * k);

		// 释放指针
		if (!(std::is_same_v<PointT, pcl::PointXYZ> && indices.empty()))
			delete cloud_temp_ptr;

		// 计算近邻点
		nearestKSearch(cloud_device, k, indices_device, disatnces_device);

		// 生成近邻索引与距离
		moveIndicesAndDistances(indices_device.first.get(), disatnces_device.first.get(), indices_device.second, k_indices, k_sqr_distances, k);

		const_cast<SearchCuda<PointT>*>(this)->deinitSearchCuda();
	}

	template<Point3D PointT>
	int SearchCuda<PointT>::radiusSearch(const PointT& point, double radius, pcl::Indices& k_indices, std::vector<float>& k_sqr_distances, unsigned int max_nn) const
	{
		if (!const_cast<SearchCuda<PointT>*>(this)->initSearchCuda())
		{
			PCL_ERROR("[% s::radiusSearch] 初始化失败! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return -1;
		}

		unsigned stride = max_nn == 0 ? cloud_device_.second : max_nn;

		// 分配GPU内存
		pcl::PointXYZ* point_device_ptr;
		pcl::index_t* indices_device_ptr;
		float* distances_device_ptr;

		cudaSafeCall(cudaMalloc(&point_device_ptr, sizeof(pcl::PointXYZ)));
		if constexpr (std::is_same_v<PointT, pcl::PointXYZ>)
			cudaSafeCall(cudaMemcpy(point_device_ptr, &point, sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
		else
		{
			pcl::PointXYZ point_temp(point.x, point.y, point.z);
			cudaSafeCall(cudaMemcpy(point_device_ptr, &point_temp, sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
		}
		CloudDevice cloud_device(getCudaSharedPtr(point_device_ptr), 1);

		cudaSafeCall(cudaMalloc(&indices_device_ptr, stride * sizeof(pcl::index_t)));
		cudaSafeCall(cudaMalloc(&distances_device_ptr, stride * sizeof(float)));
		IndicesDevice indices_device(getCudaSharedPtr(indices_device_ptr), stride);
		DistancesDevice disatnces_device(getCudaSharedPtr(distances_device_ptr), stride);

		// 计算近邻点
		radiusSearch(cloud_device, radius, indices_device, disatnces_device, max_nn);

		// 生成近邻索引与距离
		moveIndicesAndDistances(indices_device.first.get(), disatnces_device.first.get(), indices_device.second, k_indices, k_sqr_distances);


		if (const_cast<SearchCuda<PointT>*>(this)->deinitSearchCuda()) return k_indices.size(); else return -1;
	}

	template<Point3D PointT>
	void SearchCuda<PointT>::radiusSearch(const PointCloud& cloud, const pcl::Indices& indices, double radius, std::vector<pcl::Indices>& k_indices, std::vector<std::vector<float>>& k_sqr_distances, unsigned int max_nn) const
	{
		if (!const_cast<SearchCuda<PointT>*>(this)->initSearchCuda())
		{
			PCL_ERROR("[% s::radiusSearch] 初始化失败! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return;
		}

		unsigned stride = max_nn == 0 ? cloud_device_.second : max_nn;

		pcl::PointCloud<pcl::PointXYZ>* cloud_temp_ptr;
		if constexpr (std::is_same_v<PointT, pcl::PointXYZ>)
		{
			if (indices.empty())
				cloud_temp_ptr = const_cast<PointCloud*>(&cloud);
			else
			{
				cloud_temp_ptr = new pcl::PointCloud<pcl::PointXYZ>;
				pcl::copyPointCloud(cloud, indices, *cloud_temp_ptr);
			}
		}
		else
		{
			cloud_temp_ptr = new pcl::PointCloud<pcl::PointXYZ>;
			if (indices.empty())
				pcl::copyPointCloud(cloud, *cloud_temp_ptr);
			else
				pcl::copyPointCloud(cloud, indices, *cloud_temp_ptr);
		}

		// 分配GPU内存
		pcl::PointXYZ* cloud_device_ptr;
		pcl::index_t* indices_device_ptr;
		float* distances_device_ptr;

		cudaSafeCall(cudaMalloc(&cloud_device_ptr, cloud_temp_ptr->size() * sizeof(pcl::PointXYZ)));
		cudaSafeCall(cudaMemcpy(cloud_device_ptr, cloud_temp_ptr->data(), cloud_temp_ptr->size() * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice));
		CloudDevice cloud_device(getCudaSharedPtr(cloud_device_ptr), cloud_temp_ptr->size());

		cudaSafeCall(cudaMalloc(&indices_device_ptr, cloud_temp_ptr->size() * stride * sizeof(pcl::index_t)));
		cudaSafeCall(cudaMalloc(&distances_device_ptr, cloud_temp_ptr->size() * stride * sizeof(float)));
		IndicesDevice indices_device(getCudaSharedPtr(indices_device_ptr), cloud_temp_ptr->size() * stride);
		DistancesDevice disatnces_device(getCudaSharedPtr(distances_device_ptr), cloud_temp_ptr->size() * stride);

		// 释放指针
		if (!(std::is_same_v<PointT, pcl::PointXYZ> && indices.empty()))
			delete cloud_temp_ptr;

		// 计算近邻点
		radiusSearch(cloud_device, radius, indices_device, disatnces_device, max_nn);

		// 生成近邻索引与距离
		moveIndicesAndDistances(indices_device.first.get(), disatnces_device.first.get(), indices_device.second, k_indices, k_sqr_distances, stride);

		const_cast<SearchCuda<PointT>*>(this)->deinitSearchCuda();
	}

	template<Point3D PointT>
	inline bool SearchCuda<PointT>::initSearchCuda(void)
	{
		if (!input_ && !cloud_device_.first)
		{
			PCL_ERROR("[% s::initCompute] 未指定主机点云和设备点云! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			return false;
		}

		// 输入点云更新，且不存在设备点云，则更新设备点云
		if (cloud_update_ && !cloud_device_.first)
		{

			// 将 input_ 中的有效点云迁移至GPU显存
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp;

			if constexpr (std::is_same_v<PointT, pcl::PointXYZ>)	// 模板实参与pcl::PointXYZ类型相同且input_内所有点有效，则不必执行copy操作
			{
				cloud_temp = std::const_pointer_cast<pcl::PointCloud<pcl::PointXYZ>>(input_);
			}
			else
			{
				cloud_temp.reset(new pcl::PointCloud<pcl::PointXYZ>);
				if (indices_)
					pcl::copyPointCloud(*input_, *indices_, *cloud_temp);
				else
					pcl::copyPointCloud(*input_, *cloud_temp);
			}

			pcl::PointXYZ* device_ptr;
			cudaSafeCall(cudaMalloc(&device_ptr, cloud_temp->size() * sizeof(pcl::PointXYZ)));
			cudaSafeCall(cudaMemcpy(device_ptr, cloud_temp->data(), cloud_temp->size() * sizeof(pcl::PointXYZ), cudaMemcpyDefault));
			cloud_device_.first = getCudaSharedPtr<pcl::PointXYZ>(device_ptr);
			cloud_device_.second = cloud_temp->size();
		}

		// 首先执行派生类的初始化操作 initCompute() ，然后再将 cloud_update_ 置为false，最终返回 initCompute() 的结果
		return initCompute() && !(cloud_update_ = false);
	}

}