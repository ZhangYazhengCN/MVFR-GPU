/**
 *
 *  @file      octree_cuda.h
 *  @brief     CUDA implementation of approximate nearest search abstract class CUDA implementation based on OcTree
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/common/io.h>
#include <pcl/common/distances.h>
#include <pcl/search/search.h>
#include <pcl/gpu/octree/octree.hpp>

#include "search_cuda.h"
#include "../utilities/types.h"

namespace mvfr
{
	template<typename PointT>
	class OcTreeCuda : public SearchCuda<PointT>
	{
	public:
		using Ptr = std::shared_ptr<OcTreeCuda<PointT>>;
		using ConstPtr = std::shared_ptr<const OcTreeCuda<PointT>>;

		using PointCloud = SearchCuda<PointT>::PointCloud;
		using PointCloudPtr = SearchCuda<PointT>::PointCloudPtr;
		using PointCloudConstPtr = SearchCuda<PointT>::PointCloudConstPtr;
		using IndicesPtr = SearchCuda<PointT>::IndicesPtr;
		using IndicesConstPtr = SearchCuda<PointT>::IndicesConstPtr;

		using SearchCuda<PointT>::getName;
		using SearchCuda<PointT>::nearestKSearch;
		using SearchCuda<PointT>::radiusSearch;
		using SearchCuda<PointT>::approxNearestSearch;


		OcTreeCuda(bool sorted = true) :
			SearchCuda<PointT>("OcTreeCuda", sorted)
		{
			tree_.reset(new pcl::gpu::Octree);
		}

		virtual ~OcTreeCuda() = default;


		/** @brief 判断当前 OcTreeCuda 对象是否可用. */
		inline bool isValid(void) const { return tree_ != nullptr; }

		/** @brief 获得内部的Octree. */
		inline const pcl::gpu::Octree::ConstPtr& getOcTree(void) const { return tree_; }

		/** @brief 设置cuda搜索的阈值（仅当查询点数量大于等于该阈值时，才会在cuda上执行并行搜索）. */
		inline void setCudaSearchThreshold(const std::uint32_t th) { search_in_cuda_th_ = th; }

		/** @brief 获取cuda搜索的阈值（仅当查询点数量大于等于该阈值时，才会在cuda上执行并行搜索）. */
		inline const std::uint32_t& getCudaSearchThreshold(void) const { return search_in_cuda_th_; }


		/**
		 * @brief 对处于GPU内存的查询点云 \c query_cloud_device 进行KNN搜索.
		 *
		 * @param[in] query_cloud_device 处于GPU内存的查询点云
		 * @param[in] k 期望查询的近邻点个数
		 * @param[in out] indices 近邻点索引.
		 * @param[in out] distances 近邻点距离.
		 */
		virtual void nearestKSearch(const CloudDevice& query_cloud_device, unsigned k,
			IndicesDevice& indices, DistancesDevice& distances) const override;

		/**
		 * @brief 对处于GPU内存的查询点云 \c query_cloud 进行RNN搜索.
		 *
		 * @param[in] query_cloud 处于GPU内存的查询点云
		 * @param[in] radius 查询半径
		 * @param[in out] indices 近邻点索引
		 * @param[in out] distances 近邻点距离
		 * @param[in] max_nn 返回的最大的查询点个数. 若为0，则返回查询半径内所有的近邻点.
		 *
		 * @note 若点云数量过大，建议不要将 \c max_nn 置为0，以免引起过大的显存开销
		 */
		virtual void radiusSearch(const CloudDevice& query_cloud_device, const double radius,
			IndicesDevice& indices, DistancesDevice& distances, unsigned max_nn = 50) const override;

		/**
		 *  @brief 近似近邻搜索，执行速度较快误差较大的近邻点搜索.
		 *
		 *  @param[in] query_cloud_device 处于GPU内存的查询点云
		 *  @param[out] indices 近邻点索引
		 *  @param[out] distances 近邻点距离
		 *
		 *  @note 若派生类无法提供近似近邻点搜索，则可用 k=1 的 nearestKSearch 覆盖该虚函数.
		 */
		virtual void approxNearestSearch(const CloudDevice& query_cloud_device, IndicesDevice& indices, DistancesDevice& distances) const override;


	protected:
		using SearchCuda<PointT>::initSearchCuda;
		using SearchCuda<PointT>::deinitSearchCuda;

		using SearchCuda<PointT>::initCompute;
		using SearchCuda<PointT>::deinitCompute;
		inline virtual bool initCompute(void) override;

	protected:
		using SearchCuda<PointT>::input_;
		using SearchCuda<PointT>::indices_;
		using SearchCuda<PointT>::sorted_results_;
		using SearchCuda<PointT>::name_;
		using SearchCuda<PointT>::cloud_device_;
		using SearchCuda<PointT>::cloud_update_;


		pcl::gpu::Octree::PointCloud cloud_device_octree_;		// pcl::gpu::Octree 内部存在一个指向cloud的地址，因此需要一个成员变量，避免悬空指针
		pcl::gpu::Octree::Ptr tree_;
		std::uint32_t search_in_cuda_th_ = 100'000;
	};
}

#include "octree_cuda.hpp"
