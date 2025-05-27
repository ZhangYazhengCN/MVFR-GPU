#include <type_traits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <cuda/std/cmath>
#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/impl/point_types.hpp>

#include "search_cuda.cuh"
#include "../utilities/cuda_utilities.h"

namespace mvfr
{

	struct is_valid_index
	{
		__host__ __device__ bool operator()(int index)
		{
			return index >= 0;
		}
	};


	// #######################################################################################################################################################################
	// #######################################################################################################################################################################
	// -------------------------------------------------------------------- SearchCuda ----------------------------------------------------------------------------------------------
	// #######################################################################################################################################################################
	// #######################################################################################################################################################################



	void moveIndicesAndDistances(pcl::index_t* indices_device, float* distances_device, const std::size_t size,
		pcl::Indices& indices_host, std::vector<float>& distances_host)
	{
		int counter = thrust::count_if(thrust::device, indices_device, indices_device + size, is_valid_index());

		indices_host.resize(counter);
		cudaSafeCall(cudaMemcpy(indices_host.data(), indices_device, counter * sizeof(pcl::index_t), cudaMemcpyDeviceToHost));
		distances_host.resize(counter);
		cudaSafeCall(cudaMemcpy(distances_host.data(), distances_device, counter * sizeof(float), cudaMemcpyDeviceToHost));
	}


	template<typename IndexType>
	__global__ void countValidIndices(IndexType* begin_ptr, IndexType* end_ptr,
		const unsigned stride, unsigned* const counters_ptr)
	{
		const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
		IndexType* current_ptr = begin_ptr + tid * stride;
		if (current_ptr < end_ptr)
		{
			counters_ptr[tid] =
				thrust::count_if(thrust::device, current_ptr, current_ptr + stride, is_valid_index());
		}
	}



	void moveIndicesAndDistances(pcl::index_t* indices_device, float* distances_device, const std::size_t size,
		std::vector<pcl::Indices>& indices_host, std::vector<std::vector<float>>& distances_host, const unsigned stride)
	{
		// indices和distances从GPU拷贝至CPU
		thrust::host_vector<pcl::index_t> indices_host_vector(size);
		thrust::host_vector<float> distances_host_vector(size);
		cudaSafeCall(cudaMemcpy(indices_host_vector.data(), indices_device, size * sizeof(pcl::index_t), cudaMemcpyDeviceToHost));
		cudaSafeCall(cudaMemcpy(distances_host_vector.data(), distances_device, size * sizeof(float), cudaMemcpyDeviceToHost));

		// 计算有效索引数量
		thrust::device_vector<unsigned> counters_device(static_cast<std::size_t>(size / stride));
		countValidIndices << <divUp(counters_device.size(), 512), 512 >> > (
			indices_device, indices_device + size, stride, thrust::raw_pointer_cast(counters_device.data()));
		cudaSafeCall(cudaDeviceSynchronize());
		thrust::host_vector<unsigned> counters_host = counters_device;

		// 生成有效索引与距离
		indices_host.resize(counters_host.size());
		distances_host.resize(counters_host.size());

		using ZipIter = thrust::zip_iterator<thrust::tuple<thrust::counting_iterator<unsigned>, decltype(counters_host)::iterator>>;
		thrust::for_each(
			ZipIter(thrust::make_tuple(thrust::counting_iterator<unsigned>(0), counters_host.begin())),
			ZipIter(thrust::make_tuple(thrust::counting_iterator<unsigned>(counters_host.size()), counters_host.end())),
			[&](ZipIter::value_type i_size)
			{
				indices_host[thrust::get<0>(i_size)].resize(thrust::get<1>(i_size));
				distances_host[thrust::get<0>(i_size)].resize(thrust::get<1>(i_size));

				thrust::copy_n(thrust::host, indices_host_vector.data() + thrust::get<0>(i_size) * stride, thrust::get<1>(i_size), indices_host[thrust::get<0>(i_size)].data());
				thrust::copy_n(thrust::host, distances_host_vector.data() + thrust::get<0>(i_size) * stride, thrust::get<1>(i_size), distances_host[thrust::get<0>(i_size)].data());
			}
		);
	}


	// #######################################################################################################################################################################
	// #######################################################################################################################################################################
	// -------------------------------------------------------------------- OctreeCuda ---------------------------------------------------------------------------------------
	// #######################################################################################################################################################################
	// #######################################################################################################################################################################


	struct calcuDistances_
	{
		const pcl::PointXYZ* queries_ptr_;
		const pcl::PointXYZ* clouds_ptr_;
		const unsigned stride_;

		calcuDistances_(const pcl::PointXYZ* const queries_ptr, const pcl::PointXYZ* const clouds_ptr, const unsigned stride)
			: queries_ptr_(queries_ptr), clouds_ptr_(clouds_ptr), stride_(stride)
		{
		}

		__host__ __device__ float operator()(const unsigned cloud_index, const std::uint32_t index)
		{
			unsigned query_index = index / stride_;
			float distance_x = clouds_ptr_[cloud_index].x - queries_ptr_[query_index].x;
			float distance_y = clouds_ptr_[cloud_index].y - queries_ptr_[query_index].y;
			float distance_z = clouds_ptr_[cloud_index].z - queries_ptr_[query_index].z;

			return cuda::std::pow(distance_x, 2) + cuda::std::pow(distance_y, 2) + cuda::std::pow(distance_z, 2);
		}
	};

	__global__ void indicesOrder(int* indices, float* distances, int* sizes, const unsigned max_nn, const std::uint32_t boundary)
	{
		const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
		if (tid < boundary)
		{
			const unsigned int begin_ = tid * max_nn;
			thrust::sort_by_key(thrust::device, distances + begin_, distances + begin_ + sizes[tid], indices + begin_);
		}
	}


	void calcuDistances(pcl::PointXYZ* query_cloud_device, pcl::PointXYZ* target_cloud_device,
		pcl::index_t* indices_device, int* indices_sizes, float* distances_device,
		const unsigned query_cloud_size, const unsigned stride, const bool sorted)
	{
		// 计算近邻点平方距离
		thrust::transform_if(thrust::device, indices_device, indices_device + query_cloud_size * stride,
			thrust::counting_iterator<std::uint32_t>(0), indices_device, distances_device, calcuDistances_(
				query_cloud_device, target_cloud_device, stride), is_valid_index());

		// 对结果进行排序
		if (sorted)
		{
			indicesOrder << <divUp(query_cloud_size, 512), 512 >> > (indices_device, distances_device, indices_sizes, stride, query_cloud_size);

			cudaSafeCall(cudaGetLastError());
			cudaSafeCall(cudaDeviceSynchronize());
		}
	}
}