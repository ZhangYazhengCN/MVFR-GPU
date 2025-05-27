#include <type_traits>
#include <complex>

#include <boost/preprocessor/seq.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <vector_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <cuda/std/tuple>

#include "../utilities/cuda_utilities.h"
#include "registration_cuda.cuh"

namespace mvfr
{
	template<typename Scalar, int StorageOrder>
	void transformCloudDevice(pcl::PointXYZ* const cloud_device_ptr, const std::size_t cloud_device_size, const Eigen::Matrix<Scalar, 4, 4, StorageOrder>& trans)
	{
		Scalar* trans_device_ptr;
		cudaSafeCall(cudaMalloc(&trans_device_ptr, 16 * sizeof(Scalar)));
		cudaSafeCall(cudaMemcpy(trans_device_ptr, trans.data(), 16 * sizeof(Scalar), cudaMemcpyHostToDevice));

		thrust::transform(thrust::device, cloud_device_ptr, cloud_device_ptr + cloud_device_size, cloud_device_ptr,
			[=] __device__(const pcl::PointXYZ & point)->pcl::PointXYZ
		{
			float x, y, z;

			x = trans_device_ptr[0] * point.x + trans_device_ptr[4] * point.y + trans_device_ptr[8] * point.z + trans_device_ptr[12];
			y = trans_device_ptr[1] * point.x + trans_device_ptr[5] * point.y + trans_device_ptr[9] * point.z + trans_device_ptr[13];
			z = trans_device_ptr[2] * point.x + trans_device_ptr[6] * point.y + trans_device_ptr[10] * point.z + trans_device_ptr[14];

			return pcl::PointXYZ(x, y, z);
		});

		cudaSafeCall(cudaFree(trans_device_ptr));
	}



	unsigned correspondencesDeviceRejector(const IndicesDevice& indices_device, const DistancesDevice& distances_device, CorrespondencesDevice& corr_device, const float th)
	{
		if (indices_device.second != distances_device.second || distances_device.second != std::get<3>(corr_device) || indices_device.second == 0)
		{
			std::cout << "数据为空或不匹配\n";
			cudaSafeCall(cudaErrorInvalidValue);
			exit(-1);
		}

		// 提取裸指针
		pcl::index_t* indices_device_ptr = indices_device.first.get();
		float* distances_device_ptr = distances_device.first.get();
		pcl::index_t* corr_device_src_ptr = std::get<0>(corr_device).get();
		pcl::index_t* corr_device_tgt_ptr = std::get<1>(corr_device).get();
		float* corr_device_dis_ptr = std::get<2>(corr_device).get();

		// 初始化计数器
		unsigned* counter_ptr;
		cudaSafeCall(cudaMalloc(&counter_ptr, sizeof(unsigned)));
		cudaSafeCall(cudaMemset(counter_ptr, 0, sizeof(unsigned)));

		// 将corr_source_ind 初始化为 -1 (后续判断用)
		thrust::host_vector<pcl::index_t> init_use_temp(std::get<3>(corr_device), -1);
		cudaSafeCall(cudaMemcpy(corr_device_src_ptr, init_use_temp.data(), std::get<3>(corr_device) * sizeof(pcl::index_t), cudaMemcpyHostToDevice));

		// 执行对应关系 filter/拒绝
		thrust::for_each(thrust::device,
			thrust::counting_iterator<pcl::index_t>(0),
			thrust::counting_iterator<pcl::index_t>(indices_device.second),
			[=] __device__(pcl::index_t i)
		{
			if (distances_device_ptr[i] <= th)
			{
				unsigned index_temp = atomicAdd(counter_ptr, 1u);

				corr_device_src_ptr[index_temp] = i;
				corr_device_tgt_ptr[index_temp] = indices_device_ptr[i];
				corr_device_dis_ptr[index_temp] = distances_device_ptr[i];

			}
		});

		unsigned counter;
		cudaSafeCall(cudaMemcpy(&counter, counter_ptr, sizeof(unsigned), cudaMemcpyDeviceToHost));
		cudaSafeCall(cudaFree(counter_ptr));

		return counter;
	}

	// @todo 异步cudastream 优化内存释放与其它操作
	template<typename Scalar, int StorageOrder>
	void computeHmatrixDevice(const CloudDevice& source_device, const CloudDevice& target_device,
		const CorrespondencesDevice& corr_device, const unsigned valid_corr,
		Eigen::Matrix<Scalar, 4, 1>& centroid_src,
		Eigen::Matrix<Scalar, 4, 1>& centroid_tgt,
		Eigen::Matrix<Scalar, 3, 3, StorageOrder>& H)
	{
		// --------------------------------- 1. 提取裸指针并生成对应于Scalar的PointType4_ ---------------------------------
		pcl::PointXYZ* src_points_ptr = source_device.first.get(), * tgt_points_ptr = target_device.first.get();
		pcl::index_t* corr_src_indices_ptr = std::get<0>(corr_device).get(), * corr_tgt_indices_ptr = std::get<1>(corr_device).get();

		using PointType4_ =
			std::conditional_t<std::is_same_v<Scalar, double>, double4,
			std::conditional_t<std::is_same_v<Scalar, float>, float4,
			std::conditional_t<std::is_same_v<Scalar, int>, int4, void>>>;


		// --------------------------------- 2. 获取对应点对 --------------------------------
		PointType4_* corr_src_points_ptr, * corr_tgt_points_ptr;
		cudaSafeCall(cudaMalloc(&corr_src_points_ptr, valid_corr * sizeof(PointType4_)));
		cudaSafeCall(cudaMalloc(&corr_tgt_points_ptr, valid_corr * sizeof(PointType4_)));
		thrust::for_each(thrust::device, thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(valid_corr),
			[=] __device__(unsigned i)
		{
			corr_src_points_ptr[i].x = src_points_ptr[corr_src_indices_ptr[i]].x;
			corr_src_points_ptr[i].y = src_points_ptr[corr_src_indices_ptr[i]].y;
			corr_src_points_ptr[i].z = src_points_ptr[corr_src_indices_ptr[i]].z;

			corr_tgt_points_ptr[i].x = tgt_points_ptr[corr_tgt_indices_ptr[i]].x;
			corr_tgt_points_ptr[i].y = tgt_points_ptr[corr_tgt_indices_ptr[i]].y;
			corr_tgt_points_ptr[i].z = tgt_points_ptr[corr_tgt_indices_ptr[i]].z;
		});


		/// @todo 两个cudastream 并发执行 ↓
		// ------------------------------ 3. 计算中心坐标并去中心化 ------------------------------
		PointType4_ sum_src = thrust::reduce(thrust::device, corr_src_points_ptr, corr_src_points_ptr + valid_corr, PointType4_(),
			[=] __device__(PointType4_ & point1, PointType4_ & point2) -> PointType4_
		{
			PointType4_ temp;
			temp.x = point1.x + point2.x;
			temp.y = point1.y + point2.y;
			temp.z = point1.z + point2.z;
			return temp;
		});

		PointType4_ sum_tgt = thrust::reduce(thrust::device, corr_tgt_points_ptr, corr_tgt_points_ptr + valid_corr, PointType4_(),
			[=] __device__(PointType4_ & point1, PointType4_ & point2) -> PointType4_
		{
			PointType4_ temp;
			temp.x = point1.x + point2.x;
			temp.y = point1.y + point2.y;
			temp.z = point1.z + point2.z;
			return temp;
		});

		centroid_src[0] = sum_src.x / valid_corr; centroid_src[1] = sum_src.y / valid_corr; centroid_src[2] = sum_src.z / valid_corr;
		centroid_tgt[0] = sum_tgt.x / valid_corr; centroid_tgt[1] = sum_tgt.y / valid_corr; centroid_tgt[2] = sum_tgt.z / valid_corr;

		Scalar* centroid_src_device_ptr, * centroid_tgt_device_ptr;
		cudaSafeCall(cudaMalloc(&centroid_src_device_ptr, 3 * sizeof(Scalar)));
		cudaSafeCall(cudaMalloc(&centroid_tgt_device_ptr, 3 * sizeof(Scalar)));
		cudaSafeCall(cudaMemcpy(centroid_src_device_ptr, centroid_src.data(), 3 * sizeof(Scalar), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemcpy(centroid_tgt_device_ptr, centroid_tgt.data(), 3 * sizeof(Scalar), cudaMemcpyHostToDevice));

		thrust::for_each(thrust::device, thrust::counting_iterator<unsigned>(0), thrust::counting_iterator<unsigned>(valid_corr),
			[=] __device__(unsigned i)
		{
			corr_src_points_ptr[i].x -= centroid_src_device_ptr[0];
			corr_src_points_ptr[i].y -= centroid_src_device_ptr[1];
			corr_src_points_ptr[i].z -= centroid_src_device_ptr[2];

			corr_tgt_points_ptr[i].x -= centroid_tgt_device_ptr[0];
			corr_tgt_points_ptr[i].y -= centroid_tgt_device_ptr[1];
			corr_tgt_points_ptr[i].z -= centroid_tgt_device_ptr[2];
		});

		cudaSafeCall(cudaFree(centroid_src_device_ptr));
		cudaSafeCall(cudaFree(centroid_tgt_device_ptr));

		// --------------------------------- 4. 计算H矩阵 ---------------------------------
		// 分配H矩阵GPU内存
		Scalar* H_device_ptr;
		cudaSafeCall(cudaMalloc(&H_device_ptr, 9 * sizeof(Scalar)));

		// 分配计算过程中间矩阵GPU内存
		using Matrix4_4Type = cuda::std::tuple<PointType4_, PointType4_, PointType4_, PointType4_>;
		Matrix4_4Type* matrix_4_4_ptr;
		cudaSafeCall(cudaMalloc(&matrix_4_4_ptr, valid_corr * sizeof(Matrix4_4Type)));

		// 计算每对对应点对的外积
		thrust::transform(thrust::device, corr_src_points_ptr, corr_src_points_ptr + valid_corr,
			corr_tgt_points_ptr, matrix_4_4_ptr,
			[] __device__(PointType4_ & x, PointType4_ & y) -> Matrix4_4Type
		{
			Matrix4_4Type matrix_temp;
			cuda::std::get<0>(matrix_temp).x = x.x * y.x;	// row:0 col:0
			cuda::std::get<0>(matrix_temp).y = x.x * y.y;	// row:0 col:1
			cuda::std::get<0>(matrix_temp).z = x.x * y.z;	// row:0 col:2

			cuda::std::get<1>(matrix_temp).x = x.y * y.x;	// row:1 col:0
			cuda::std::get<1>(matrix_temp).y = x.y * y.y;	// row:1 col:1
			cuda::std::get<1>(matrix_temp).z = x.y * y.z;	// row:1 col:2

			cuda::std::get<2>(matrix_temp).x = x.z * y.x;	// row:2 col:0
			cuda::std::get<2>(matrix_temp).y = x.z * y.y;	// row:2 col:1
			cuda::std::get<2>(matrix_temp).z = x.z * y.z;	// row:2 col:2

			return matrix_temp;
		});


		//using PointIter = thrust::device_vector<PointType4_>::iterator;
		//using ZipIter = thrust::zip_iterator<thrust::tuple<PointIter, PointIter>>;
		//// 计算外积
		//struct calcu_outer
		//{
		//	__device__ Matrix4_4Type& operator()(const ZipIter::value_type& x_y)
		//	{
		//		Matrix4_4Type matrix_temp;
		//		cuda::std::get<0>(matrix_temp).x = thrust::get<0>(x_y).x * thrust::get<1>(x_y).x;	// row:0 col:0
		//		cuda::std::get<0>(matrix_temp).y = thrust::get<0>(x_y).x * thrust::get<1>(x_y).y;	// row:0 col:1
		//		cuda::std::get<0>(matrix_temp).z = thrust::get<0>(x_y).x * thrust::get<1>(x_y).z;	// row:0 col:1

		//		cuda::std::get<1>(matrix_temp).x = thrust::get<0>(x_y).y * thrust::get<1>(x_y).x;	// row:1 col:0
		//		cuda::std::get<1>(matrix_temp).y = thrust::get<0>(x_y).y * thrust::get<1>(x_y).y;	// row:1 col:1
		//		cuda::std::get<1>(matrix_temp).z = thrust::get<0>(x_y).y * thrust::get<1>(x_y).z;	// row:1 col:1

		//		cuda::std::get<2>(matrix_temp).x = thrust::get<0>(x_y).z * thrust::get<1>(x_y).x;	// row:2 col:0
		//		cuda::std::get<2>(matrix_temp).y = thrust::get<0>(x_y).z * thrust::get<1>(x_y).y;	// row:2 col:1
		//		cuda::std::get<2>(matrix_temp).z = thrust::get<0>(x_y).z * thrust::get<1>(x_y).z;	// row:2 col:1

		//		return matrix_temp;
		//	}
		//};
		//using TransformIter = thrust::transform_iterator<calcu_outer, ZipIter>;

		// 外积求和
		Matrix4_4Type res = thrust::reduce(thrust::device, matrix_4_4_ptr, matrix_4_4_ptr + valid_corr, Matrix4_4Type(),
			[] __device__(Matrix4_4Type & x, Matrix4_4Type & y) -> Matrix4_4Type
		{
			Matrix4_4Type matrix_temp;
			cuda::std::get<0>(matrix_temp).x = cuda::std::get<0>(x).x + cuda::std::get<0>(y).x;	// row:0 col:0
			cuda::std::get<0>(matrix_temp).y = cuda::std::get<0>(x).y + cuda::std::get<0>(y).y;	// row:0 col:1
			cuda::std::get<0>(matrix_temp).z = cuda::std::get<0>(x).z + cuda::std::get<0>(y).z;	// row:0 col:2

			cuda::std::get<1>(matrix_temp).x = cuda::std::get<1>(x).x + cuda::std::get<1>(y).x;	// row:1 col:0
			cuda::std::get<1>(matrix_temp).y = cuda::std::get<1>(x).y + cuda::std::get<1>(y).y;	// row:1 col:1
			cuda::std::get<1>(matrix_temp).z = cuda::std::get<1>(x).z + cuda::std::get<1>(y).z;	// row:1 col:2

			cuda::std::get<2>(matrix_temp).x = cuda::std::get<2>(x).x + cuda::std::get<2>(y).x;	// row:2 col:0
			cuda::std::get<2>(matrix_temp).y = cuda::std::get<2>(x).y + cuda::std::get<2>(y).y;	// row:2 col:1
			cuda::std::get<2>(matrix_temp).z = cuda::std::get<2>(x).z + cuda::std::get<2>(y).z;	// row:2 col:2

			return matrix_temp;
		});

		H(0, 0) = cuda::std::get<0>(res).x; H(0, 1) = cuda::std::get<0>(res).y; H(0, 2) = cuda::std::get<0>(res).z;
		H(1, 0) = cuda::std::get<1>(res).x; H(1, 1) = cuda::std::get<1>(res).y; H(1, 2) = cuda::std::get<1>(res).z;
		H(2, 0) = cuda::std::get<2>(res).x; H(2, 1) = cuda::std::get<2>(res).y; H(2, 2) = cuda::std::get<2>(res).z;

		// ------------------------------- 5. 释放申请的GPU内存 ------------------------------
		cudaSafeCall(cudaFree(matrix_4_4_ptr));
		cudaSafeCall(cudaFree(H_device_ptr));
		cudaSafeCall(cudaFree(corr_src_points_ptr));
		cudaSafeCall(cudaFree(corr_tgt_points_ptr));
	}


// transformCloudDevice 和 computeHmatrixDevice 显式实例化. 
// @note: 不支持 (Eigen::RowMajor)
// @note: 调整 Eigen_stdTypes 后，需修改 computeHmatrixDevice 函数
#define Eigen_storageOrder (Eigen::ColMajor)(Eigen::AutoAlign)(Eigen::DontAlign)
#define Eigen_stdTypes (int)(float)(double)
#define Instantiate_transformCloudDevice(r, product) template void transformCloudDevice\
<BOOST_PP_SEQ_ELEM(0, product),BOOST_PP_SEQ_ELEM(1, product)>\
(pcl::PointXYZ* const, const std::size_t, const Eigen::Matrix<BOOST_PP_SEQ_ELEM(0, product), 4, 4, BOOST_PP_SEQ_ELEM(1, product)>&);\
\
template void computeHmatrixDevice<BOOST_PP_SEQ_ELEM(0, product), BOOST_PP_SEQ_ELEM(1, product)>\
(const CloudDevice& source_device, const CloudDevice& target_device,\
const CorrespondencesDevice& corr_device, const unsigned valid_corr,\
Eigen::Matrix<BOOST_PP_SEQ_ELEM(0, product), 4, 1>& centroid_src,\
Eigen::Matrix<BOOST_PP_SEQ_ELEM(0, product), 4, 1>& centroid_tgt,\
Eigen::Matrix<BOOST_PP_SEQ_ELEM(0, product), 3, 3, BOOST_PP_SEQ_ELEM(1, product)>& H);

	BOOST_PP_SEQ_FOR_EACH_PRODUCT(Instantiate_transformCloudDevice, (Eigen_stdTypes)(Eigen_storageOrder))

#undef Instantiate_transformCloudDevice
#undef Eigen_stdTypes
#undef Eigen_storageOrder
}