/**
 *
 *  @file      accumulative_registration.h
 *  @brief     CUDA implementation of accumulative multi-view point cloud fine registration
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once

#include "multiview_registration.h"


///** @brief 空多视图点云配准算法基类，用于为特化模板提供同一的接口. */
//template<typename PointT, typename Scalar>
//struct NullMultiViewRegistrationBase
//{
//	using Ptr = std::shared_ptr<NullMultiViewRegistrationBase<PointT, Scalar>>;
//	using ConstPtr = std::shared_ptr<const NullMultiViewRegistrationBase<PointT, Scalar>>;
//
//	// 成员函数
//	inline void setClouds(void) const
//	{
//		PCL_ERROR("[% s::setClouds] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void getClouds(void) const
//	{
//		PCL_ERROR("[% s::getClouds] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void getIndicess(void) const
//	{
//		PCL_ERROR("[% s::getIndicess] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void setSearchs(void) const
//	{
//		PCL_ERROR("[% s::setSearchs] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void getSearchs(void) const
//	{
//		PCL_ERROR("[% s::getSearchs] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void setInitTransforms(void) const
//	{
//		PCL_ERROR("[% s::setInitTransforms] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void getInitTransforms(void) const
//	{
//		PCL_ERROR("[% s::getsInitTransforms] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void setRegistrationMethod(void) const
//	{
//		PCL_ERROR("[% s::setRegistrationMethod] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void getRegistrationMethod(void) const
//	{
//		PCL_ERROR("[% s::getRegistrationMethod] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void getFinalTransforms(void) const
//	{
//		PCL_ERROR("[% s::getFinalTransforms] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline void getFullCloud(void) const
//	{
//		PCL_ERROR("[% s::getFullCloud] 正在调用空函数!\n", getClassName().c_str());
//	}
//	inline const std::string& getClassName(void) const
//	{
//		return multiview_reg_name_;
//	}
//	void alignclouds(void) const
//	{
//		PCL_ERROR("[% s::alignclouds] 正在调用空函数!\n", getClassName().c_str());
//	}
//	void operator()(void) const
//	{
//		PCL_ERROR("[% s::operator()] 正在调用空函数!\n", getClassName().c_str());
//	}
//
//	// 成员变量 
//	std::string multiview_reg_name_ = "NullMultiViewRegistrationBase";
//	bool clouds_ = false;
//	bool indicess_ = false;
//	bool init_transforms_ = false;
//	bool searchs_ = false;
//	bool registration_ = false;
//	bool transforms_ = false;
//	bool full_cloud_ = false;
//};
//
//
//template<>
//class SequentialRegistration<void, void> : public NullMultiViewRegistrationBase<void, void> 
//{ 
//public:
//	using NullMultiViewRegistrationBase<void, void>::multiview_reg_name_;
//
//	SequentialRegistration(){
//		multiview_reg_name_ = "NullSequentialRegistration";
//	};
//};
//
//template<>
//class CumulativeRegistration<void, void> : public NullMultiViewRegistrationBase<void, void> 
//{
//public:
//	CumulativeRegistration(){
//		multiview_reg_name_ = "NullCumulativeRegistration";
//	};
//};