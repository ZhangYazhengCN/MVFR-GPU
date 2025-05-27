/**
 *
 *  @file      search_cuda.cuh
 *  @brief     some functions that need to be compiled by nvcc and used in approximate nearest search
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <vector>

#include <pcl/point_types.h>

namespace mvfr
{
    /**
        @brief 将GPU显存内的近邻点索引和距离数据移动至CPU内存中(单个查询点).
        @param[in] indices_device   指向保存近邻点索引的指针
        @param[in] distances_device 指向保存近邻点距离的指针
        @param[in] size             近邻点数量
        @param[out] indices_host     CPU近邻点索引vector
        @param[out] distances_host   CPU近邻点距离vector
    **/
    void moveIndicesAndDistances(pcl::index_t* indices_device, float* distances_device, const std::size_t size,
        pcl::Indices& indices_host, std::vector<float>& distances_host);


    /**
        @brief 将GPU显存内的近邻点索引和距离数据移动至CPU内存中(多个查询点).
        @param[in] indices_device   指向保存近邻点索引的指针
        @param[in] distances_device 指向保存近邻点距离的指针
        @param[in] size             总的近邻点数量
        @param[out] indices_host     CPU近邻点索引vector
        @param[out] distances_host   CPU近邻点距离vector
        @param[in] stride           每个查询点近邻点数量
    **/
    void moveIndicesAndDistances(pcl::index_t* indices_device, float* distances_device, const std::size_t size,
        std::vector<pcl::Indices>& indices_host, std::vector<std::vector<float>>& distances_host, const unsigned stride);


    /**
        @brief 根据近邻点索引值计算近邻点距离，并根据 \c sorted 的取值对结果进行排序
        @param[in] query_cloud_device  查询点云
        @param[in] target_cloud_device 目标点云
        @param[in out] indices_device      近邻点索引
        @param[in] indices_sizes       每个查询点的近邻点数量
        @param[out] distances_device    近邻点距离
        @param[in] query_cloud_size    查询点云大小
        @param[in] stride              每个查询点的最大近邻点数量
        @param[in] sorted              是否对结果进行排序
    **/
    void calcuDistances(pcl::PointXYZ* query_cloud_device, pcl::PointXYZ* target_cloud_device,
        pcl::index_t* indices_device, int* indices_sizes, float* distances_device,
        const unsigned query_cloud_size, const unsigned stride, const bool sorted = true);
}