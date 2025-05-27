/**
 *
 *  @file      types.h
 *  @brief     custom concepts and types
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <utility>
#include <type_traits>
#include <tuple>
#include <memory>

#include <pcl/types.h>
#include <pcl/point_types.h>

namespace mvfr
{

#ifndef __NVCC__
#	if __cpp_concepts
	/// floating concept
	template<typename T>
	concept FloatingType = std::is_floating_point_v<std::decay_t<T>>;
	
	/// arithmetic concept
	template<typename T>
	concept ArithMeticType = std::is_arithmetic_v<std::decay_t<T>>;

	/// same types concept
	template<typename T, typename... U>
	concept SameTypes = (std::same_as<T, U>&&...);

	/// Point3D concept, has x, y, z fields which are same arithmetic type. 
	template<typename T>
	concept Point3D = requires(T point) {
		{ point.x }->ArithMeticType;
		{ point.y }->ArithMeticType;
		{ point.z }->ArithMeticType;
		requires SameTypes<
			decltype(point.x),
			decltype(point.y),
			decltype(point.z)>;
	};
#	else
#		error "no support for concept feature in current compiler"
#	endif
#endif
    /// GPU点云类型，first为指向GPU内存块的共享指针，second为内存块的大小
    using CloudDevice = std::pair<std::shared_ptr<pcl::PointXYZ>, std::size_t>;

    /// GPU近邻点索引类型，first为指向GPU内存块的共享指针，second为内存块的大小
    using IndicesDevice = std::pair<std::shared_ptr<pcl::index_t>, std::size_t>;

    /// GPU近邻点距离类型，first为指向GPU内存块的共享指针，second为内存块的大小
    using DistancesDevice = std::pair<std::shared_ptr<float>, std::size_t>;

    /// GPU对应关系类型，源点云索引 ==> 目标点云索引 ==> 对应点距离索引 ==> 对应关系数量
    using CorrespondencesDevice = std::tuple<std::shared_ptr<pcl::index_t>,
        std::shared_ptr<pcl::index_t>, std::shared_ptr<float>, std::size_t>;

}