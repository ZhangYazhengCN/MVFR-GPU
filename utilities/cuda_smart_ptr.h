/**
 *
 *  @file      cuda_smart_ptr.h
 *  @brief     shared_ptr and unique_ptr wrapper for cuda plain pointer
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <memory>
#include <type_traits>
#include <source_location>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mvfr
{
	/**
	 * @brief 适用于cuda指针的deleter.
	 *
	 * @note 仅支持 <b>非数组类型</b> T的delete
	 * @note 调用者应保证删除的指针是 \b cudaruntime 分配的
	 */
	template<typename T, std::enable_if_t<!std::is_array_v<T>, bool> = true>
	struct cudaPtrDeleter
	{
		constexpr cudaPtrDeleter() noexcept = default;

		template<typename T2, std::enable_if_t<std::is_convertible_v<T2*, T*>, int> = 0>
		cudaPtrDeleter(const cudaPtrDeleter<T2>&) noexcept {}

		void operator()(T* ptr) const noexcept
		{
			static_assert(0 < sizeof(T), "T 当前为不完整类型(incomplete type)，无法执行delete操作");
			cudaError_t cudaStatus;

			cudaStatus = cudaFree(ptr);
			if (cudaStatus != cudaSuccess)
			{
				const auto location = std::source_location::current();
				std::cout << "Error: " << cudaGetErrorString(cudaStatus) << "\t" << location.file_name() << ":" << location.line()
					<< ":" << location.function_name() << '\n';
			}
		}
	};

	/**
	 * @brief 将指向GPU内存的裸指针包装为共享指针.
	 *
	 * @param[in] ptr 指向GPU内存的裸指针
	 *
	 * @note 若裸指针为空或未指向GPU内存，则返回空的共享指针
	 */
	template<typename T, std::enable_if_t<!std::is_array_v<T>, bool> = true>
	std::shared_ptr<T> getCudaSharedPtr(T* ptr = nullptr)
	{
		// 裸指针为空则返回空的共享指针
		if (ptr == nullptr)
			return std::shared_ptr<T>();

		// 裸指针没有指向GPU内存同样返回空的共享指针
		cudaError_t cudaStatus;
		cudaPointerAttributes ptr_attr;
		cudaStatus = cudaPointerGetAttributes(&ptr_attr, (void*)ptr);
		if (cudaStatus != cudaSuccess || ptr_attr.type != cudaMemoryTypeDevice)
			return std::shared_ptr<T>();

		return std::shared_ptr<T>(ptr, cudaPtrDeleter<T>());
	}

	/**
	 * @brief 将指向GPU内存的裸指针包装为独占指针.
	 *
	 * @param[in] ptr 指向GPU内存的裸指针
	 *
	 * @note 若裸指针为空或未指向GPU内存，则返回空的独占指针
	 */
	template<typename T, std::enable_if_t<!std::is_array_v<T>, bool> = true>
	std::unique_ptr<T, cudaPtrDeleter<T>> getCudaUniquePtr(T* ptr = nullptr)
	{
		// 裸指针为空则返回空的共享指针
		if (ptr == nullptr)
			return std::unique_ptr<T, cudaPtrDeleter<T>>();

		// 裸指针没有指向GPU内存同样返回空的共享指针
		cudaError_t cudaStatus;
		cudaPointerAttributes ptr_attr;
		cudaStatus = cudaPointerGetAttributes(&ptr_attr, (void*)ptr);
		if (cudaStatus != cudaSuccess || ptr_attr.type != cudaMemoryTypeDevice)
			return std::unique_ptr<T, cudaPtrDeleter<T>>();

		return std::unique_ptr<T, cudaPtrDeleter<T>>(ptr);
	}

}