/**
 *
 *  @file      cuda_utilities.h
 *  @brief     utilities for cuda programming
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <iostream>

#include <boost/assert/source_location.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mvfr
{
    /**
     * @brief 根据cudaruntime函数的错误码输出错误信息，并结束程序.
     *
     * @param[in] err 错误码
     * @param[in] location 文件定位信息
     */
    inline void cudaSafeCall(const cudaError_t err, const boost::source_location& location = BOOST_CURRENT_LOCATION)
    {
        if (err != cudaSuccess) {
            std::cout << "CUDAError: " << cudaGetErrorString(err) << "\t" << location.file_name() << ":" << location.line()
                << ":" << location.function_name() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    /**
     * @brief 计算 \c total 除以 \c grain 的商（上取整）.
     *
     * @param[in] total
     * @param[in] grain
     * @return
     */
    inline int divUp(int total, int grain)
    {
        return (total + grain - 1) / grain;
    }

}
