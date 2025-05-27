/**
 *
 *  @file      sequential_registration.h
 *  @brief     CUDA implementation of sequential multi-view point cloud fine registration
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once


#include "multiview_registration.h"


///** @brief 顺序成对多视图点云配准算法配准. */
//template<typename PointT = pcl::PointXYZ,typename Scalar = double>
//class SequentialRegistration : public MultiViewRegistrationBase<PointT, Scalar>
//{
//public:
//	using Ptr = std::shared_ptr<SequentialRegistration<PointT, Scalar>>;
//	using ConstPtr = std::shared_ptr<const SequentialRegistration<PointT, Scalar>>;
//
//	using PointCloudPtr = typename MultiViewRegistrationBase<PointT, Scalar>::PointCloudPtr;
//	using PointCloudConstPtr = typename MultiViewRegistrationBase<PointT, Scalar>::PointCloudConstPtr;
//	using PointCloudConstPtrVector = typename MultiViewRegistrationBase<PointT, Scalar>::PointCloudConstPtrVector;
//	using Matrix4 = typename MultiViewRegistrationBase<PointT, Scalar>::Matrix4;
//	using Matrix4Vector = typename MultiViewRegistrationBase<PointT, Scalar>::Matrix4Vector;
//
//	using MultiViewRegistrationBase<PointT, Scalar>::getClassName;
//
//	SequentialRegistration()
//	{
//		multiview_reg_name_ = "SequentialRegistration";
//	}
//
//	~SequentialRegistration() override = default;
//
//
//
//protected:
//	using MultiViewRegistrationBase<PointT, Scalar>::multiview_reg_name_;
//	using MultiViewRegistrationBase<PointT, Scalar>::clouds_;
//	using MultiViewRegistrationBase<PointT, Scalar>::indicess_;
//	using MultiViewRegistrationBase<PointT, Scalar>::init_transforms_;
//	using MultiViewRegistrationBase<PointT, Scalar>::registration_;
//	using MultiViewRegistrationBase<PointT, Scalar>::transforms_;
//	using MultiViewRegistrationBase<PointT, Scalar>::full_cloud_;
//	using MultiViewRegistrationBase<PointT, Scalar>::prebuild_searchs_;
//	using MultiViewRegistrationBase<PointT, Scalar>::searchs_;
//
//private:
//	virtual bool computeTransforms(void) override;
//};
//
///** @brief 顺序累积多视图点云配准算法. */
//template<typename PointT = pcl::PointXYZ,typename Scalar = double>
//class CumulativeRegistration : public MultiViewRegistrationBase<PointT, Scalar>
//{
//public:
//	using Ptr = std::shared_ptr<CumulativeRegistration<PointT, Scalar>>;
//	using ConstPtr = std::shared_ptr<const CumulativeRegistration<PointT, Scalar>>;
//
//	using PointCloudPtr = typename MultiViewRegistrationBase<PointT, Scalar>::PointCloudPtr;
//	using PointCloudConstPtr = typename MultiViewRegistrationBase<PointT, Scalar>::PointCloudConstPtr;
//	using PointCloudConstPtrVector = typename MultiViewRegistrationBase<PointT, Scalar>::PointCloudConstPtrVector;
//	using Matrix4 = typename MultiViewRegistrationBase<PointT, Scalar>::Matrix4;
//	using Matrix4Vector = typename MultiViewRegistrationBase<PointT, Scalar>::Matrix4Vector;
//
//	using MultiViewRegistrationBase<PointT, Scalar>::getClassName;
//
//	CumulativeRegistration()
//	{
//		multiview_reg_name_ = "CumulativeRegistration";
//	}
//
//	~CumulativeRegistration() override = default;
//
//protected:
//	using MultiViewRegistrationBase<PointT, Scalar>::multiview_reg_name_;
//	using MultiViewRegistrationBase<PointT, Scalar>::clouds_;
//	using MultiViewRegistrationBase<PointT, Scalar>::indicess_;
//	using MultiViewRegistrationBase<PointT, Scalar>::init_transforms_;
//	using MultiViewRegistrationBase<PointT, Scalar>::registration_;
//	using MultiViewRegistrationBase<PointT, Scalar>::transforms_;
//	using MultiViewRegistrationBase<PointT, Scalar>::full_cloud_;
//	using MultiViewRegistrationBase<PointT, Scalar>::prebuild_searchs_;
//	using MultiViewRegistrationBase<PointT, Scalar>::searchs_;
//
//private:
//	virtual bool computeTransforms(void) override;
//};