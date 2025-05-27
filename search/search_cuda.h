/**
 *
 *  @file      search_cuda.h
 *  @brief     CUDA implementation of approximate nearest search abstract class
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <memory>
#include <utility>

#include <cuda_runtime.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/pcl_base.h>
#include <pcl/search/search.h>

#include "../utilities/cuda_smart_ptr.h"
#include "../utilities/types.h"

namespace mvfr
{

	/**
	 * @brief 在GPU上执行最近邻点搜索(抽象类).
	 *
	 * @note 根据点云的(x, y, z)坐标进行近邻点搜索，因此模板实参 \c PointT 应具有成员变量 x y z.
	 * @note 仅支持PCL内置的PointType
	 */
	template<Point3D PointT>
	class SearchCuda : public pcl::search::Search<PointT>
	{
	public:
		using Ptr = std::shared_ptr<SearchCuda<PointT>>;
		using ConstPtr = std::shared_ptr<const SearchCuda<PointT>>;

		using PointCloud = typename pcl::search::Search<PointT>::PointCloud;
		using PointCloudPtr = typename pcl::search::Search<PointT>::PointCloudPtr;
		using PointCloudConstPtr = typename pcl::search::Search<PointT>::PointCloudConstPtr;

		using IndicesPtr = typename pcl::search::Search<PointT>::IndicesPtr;
		using IndicesConstPtr = typename pcl::search::Search<PointT>::IndicesConstPtr;

		// 引入模板基类 pcl::search::Search<PointT> 成员函数标识符
		using pcl::search::Search<PointT>::getName;
		using pcl::search::Search<PointT>::setInputCloud;
		using pcl::search::Search<PointT>::nearestKSearch;
		using pcl::search::Search<PointT>::radiusSearch;


		SearchCuda(const std::string& name = "SearchCuda", bool sorted = false) :
			pcl::search::Search<PointT>::Search(name, sorted)
		{
			//// 判断设备是否支持cuda  
			//int gpu_count_ = 0;
			//cudaError_t cudaStatus = cudaGetDeviceCount(&gpu_count_);
			//if (cudaStatus != cudaSuccess)
			//{
			//	PCL_ERROR("[% s::OcTreeCuda] 设备信息获取失败（错误码：%s）! %s(%d)\n", getName().c_str(), cudaGetErrorString(cudaStatus), __FILE__, __LINE__);
			//	exit(EXIT_FAILURE);
			//}
			//else if (gpu_count_ == 0)
			//{
			//	PCL_ERROR("[% s::OcTreeCuda] 设备中无可用GPU! %s(%d)\n", getName().c_str(), __FILE__, __LINE__);
			//	exit(EXIT_FAILURE);
			//}
		}

		virtual ~SearchCuda() = default;

		/**
		 * @brief 设置主机点云.
		 *
		 * @param[in] cloud 指向主机点云的共享指针
		 * @param[in] indices 主机点云有效索引
		 * @return 主机点云是否设置成功
		 */
		virtual bool setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = IndicesConstPtr());

		/**
		 * @brief 设置设备点云.
		 *
		 * @param[in] cloud_device 设备点云:std::pair<点云指针, 点云大小>
		 * @return 设备点云是否设置成功
		 */
		virtual bool setInputCloud(const CloudDevice& cloud_device);


		/**
		 * @brief 同时设置主机点云与设备点云.
		 *
		 * @param[in] cloud 指向主机点云的共享指针
		 * @param[in] cloud_device 设备点云:std::pair<点云指针, 点云大小>
		 * @param[in] indices 主机点云有效索引
		 * @return 点云是否设置成功
		 *
		 * @note 应保证设置的主机点云与设备点云相一致，即设备点云大小的与主机点云（或主机点云有效索引）的数量相等
		 */
		virtual bool setInputCloud(const PointCloudConstPtr& cloud, const CloudDevice& cloud_device, const IndicesConstPtr& indices = IndicesConstPtr());

		/** @brief 获取设备点云. */
		const CloudDevice& getInputDevice(void) const { return cloud_device_; }


		/*##########################################################################################################
		*********************************************  传入GPU查询点云 *******************************************
		##########################################################################################################*/

		/**
		 * @brief 对处于GPU内存的查询点云 \c query_cloud_device 进行KNN搜索.
		 *
		 * @param[in] query_cloud_device 处于GPU内存的查询点云
		 * @param[in] k 期望查询的近邻点个数
		 * @param[in out] indices 近邻点索引.
		 * @param[in out] distances 近邻点距离.
		 *
		 * @note 保存 \c indices 和 \c distances 的显存可以在函数外部预分配（传入非空指针参数），也可以在函数内部自动分配（传入空指针参数）.
		 * 若显存是预分配的，则其大小应为 \b query_cloud_device.second*k , 否则在函数内部通过 \b cudaruntime 提供的接口自动分配同样大小的显存.
		 *
		 * @note \c indices 中的无效索引应小于0, 需要根据 \c sorted_results_ 对结果进行排序
		 *
		 * @note 注意调用 initSearchCuda 与 deinitSearchCuda
		 */
		virtual void nearestKSearch(const CloudDevice& query_cloud_device, unsigned k,
			IndicesDevice& indices, DistancesDevice& distances) const = 0;

		/**
		 * @brief 对处于GPU内存的查询点云 \c query_cloud 进行RNN搜索.
		 *
		 * @param[in] query_cloud 处于GPU内存的查询点云
		 * @param[in] radius 查询半径
		 * @param[in out] indices 近邻点索引
		 * @param[in out] distances 近邻点距离
		 * @param[in] max_nn 返回的最大的查询点个数. 若为0，则返回查询半径内所有的近邻点.
		 *
		 * @note 保存 \c indices 和 \c distances 的显存可以在函数外部预分配（传入非空指针参数），也可以在函数内部自动分配（传入空指针参数）.
		 * 若显存是预分配的，则其大小应为 \b query_cloud_device.second*max_nn (当 \c max_nn 等于0时)或  \b query_cloud_device.second*cloud_deivce_.second (当 \c max_nn 不等于0时),
		 * 否则在函数内部通过 \b cudaruntime 提供的接口自动分配同样大小的显存.
		 *
		 * @note \c indices 中的无效索引应小于0, 需要根据 \c sorted_results_ 对结果进行排序
		 *
		 * @note 若点云数量过大，建议不要将 \c max_nn 置为0，以免引起过大的显存开销
		 *
		 * @note 注意调用 \c initSearchCuda 与 \c deinitSearchCuda
		 */
		virtual void radiusSearch(const CloudDevice& query_cloud_device, const double radius,
			IndicesDevice& indices, DistancesDevice& distances, unsigned max_nn = 50) const = 0;


		/**
		 *  @brief 近似近邻搜索，执行速度较快误差较大的近邻点搜索.
		 *
		 *  @param[in] query_cloud_device 处于GPU内存的查询点云
		 *  @param[out] indices 近邻点索引
		 *  @param[out] distances 近邻点距离
		 *
		 *  @note 保存 \c indices 和 \c distances 的显存可以在函数外部预分配（传入非空指针参数），也可以在函数内部自动分配（传入空指针参数）.
		 *  若显存是预分配的，则其大小应为 \b query_cloud_device.second, 否则在函数内部通过 \b cudaruntime 提供的接口自动分配同样大小的显存.
		 *
		 *  @note 若派生类无法提供近似近邻点搜索，则可用 k=1 的 nearestKSearch 覆盖该虚函数.
		 */
		virtual void approxNearestSearch(const CloudDevice& query_cloud_device, IndicesDevice& indices, DistancesDevice& distances) const = 0;


		/*##########################################################################################################
		*********************************************  传入CPU查询点云 *********************************************
		##########################################################################################################*/

		/**
		 * @brief 对处于CPU的查询点 \c point 进行KNN搜索.
		 *
		 * @param[in] point 查询点
		 * @param[in] k 期望查询的近邻点数量
		 * @param[out] k_indices 近邻点索引
		 * @param[out] k_sqr_distances 近邻点距离
		 * @return 查询的近邻点数量
		 */
		virtual int nearestKSearch(const PointT& point, int k, pcl::Indices& k_indices, std::vector<float>& k_sqr_distances) const;

		/**
		 * @brief 对处于CPU的查询点云 \c cloud 进行KNN搜索.
		 *
		 * @param[in] cloud 查询点云
		 * @param[in] indices 查询点云的有效索引
		 * @param[in] k 期望查询的近邻点数量
		 * @param[out] k_indices 近邻点索引
		 * @param[out] k_sqr_distances 近邻点距离
		 */
		virtual void nearestKSearch(const PointCloud& cloud, const pcl::Indices& indices, int k,
			std::vector<pcl::Indices>& k_indices, std::vector< std::vector<float> >& k_sqr_distances) const;

		/**
		 * @brief 对处于CPU的查询点 \c point 进行RNN搜索.
		 *
		 * @param[in] point 查询点
		 * @param[in] radius 查询半径
		 * @param[out] k_indices 近邻点索引
		 * @param[out] k_sqr_distances 近邻点距离
		 * @param[in] max_nn 查询半径内返回的最大近邻点数量，为0则返回所有的近邻点
		 * @return 查询的紧邻点数量
		 */
		virtual int radiusSearch(const PointT& point, double radius, pcl::Indices& k_indices,
			std::vector<float>& k_sqr_distances, unsigned int max_nn = 0) const;

		/**
		 * @brief 对处于CPU的查询点云 \c cloud 进行KNN搜索.
		 *
		 * @param[in] cloud 查询点云
		 * @param[in] indices 查询点云有效索引
		 * @param[in] radius 查询半径
		 * @param[out] k_indices 近邻点索引
		 * @param[out] k_sqr_distances 近邻点距离
		 * @param[in] max_nn 查询半径内返回的最大近邻点数量，为0则返回所有的近邻点
		 */
		virtual void radiusSearch(const PointCloud& cloud, const pcl::Indices& indices, double radius,
			std::vector<pcl::Indices>& k_indices, std::vector< std::vector<float> >& k_sqr_distances, unsigned int max_nn = 0) const;


	protected:
		// 开始（或结束）执行搜索算法时，需执行相应的初始化（解初始化）操作
		bool initSearchCuda(void);
		inline bool deinitSearchCuda(void) { return deinitCompute(); }

		// 若派生类需要额外的初始化（解初始化）操作，则 override 以下两个函数
		inline virtual bool initCompute(void) { return true; }
		inline virtual bool deinitCompute(void) { return true; }


		//void moveIndicesAndDistances(const IndicesDevice& indices_device, const DistancesDevice& distances_device,
		//	pcl::Indices& indices_host, std::vector<float>& distances_host);

		//void moveIndicesAndDistances(const IndicesDevice& indices_device, const DistancesDevice& distances_device,
		//	std::vector<pcl::Indices>& indices_host, std::vector<std::vector<float>>& distances_host, unsigned stride);

	protected:
		// 引入模板基类 pcl::search::Search<PointT> 成员变量标识符
		using pcl::search::Search<PointT>::input_;
		using pcl::search::Search<PointT>::indices_;
		using pcl::search::Search<PointT>::sorted_results_;
		using pcl::search::Search<PointT>::name_;

		/** @brief 设备点云:std::pair<点云指针, 点云大小>. */
		CloudDevice cloud_device_ = std::make_pair(getCudaSharedPtr<pcl::PointXYZ>(), static_cast<std::size_t>(0));
		/** @brief 点云是否已更新. */
		bool cloud_update_ = true;
	};

}

#include "search_cuda.hpp"


