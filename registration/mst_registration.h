/**
 *
 *  @file      mst_registration.h
 *  @brief	   CUDA implementation of (Dynamic)MST multi-view point cloud fine registraion
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <vector>
#include <memory>
#include <type_traits>

#include <boost/graph/adjacency_list.hpp>

#include "multiview_registration.h"

namespace mvfr
{
	namespace details
	{
		/// Internal Use. 动态最小生成树多视图配准算法邻接关系图顶点的额外属性
		template<typename PointT>
		struct DynamicMSTVertexProperty
		{
			std::vector<std::size_t> joint_clouds_id;
			typename pcl::PointCloud<PointT>::Ptr joint_clouds;
			Eigen::Array<bool, 1, Eigen::Dynamic> overlap_indices_flag;
		};
	}

	/**
	 *
	 *  @class   MSTRegistration
	 *  @brief   基于（动态）最小生成树的多视图点云精配准
	 *  @details 当 \c dynamic 为false(默认)，基于最小生成树策略执行多视图点云精配准；否则，基于动态图的策略执行最小生成树
	 *  @tparam  PointT  点云类型
	 *  @tparam  Scalar  计算精度
	 *  @tparam  RegistrationT	两视图配准方法
	 *  @tparam  Dynamic 是否采用动态最小生成树
	 *
	 *  @note 当采用动态图执行多视图点云精配准时，必须借助模板点云构建邻接关系图，请确保正确设置了模板点云
	 */
	template<typename PointT = pcl::PointXYZ, typename Scalar = double,
		template<typename, typename, typename> typename RegistrationT = IterativeClosestPointCuda, bool Dynamic = false>
	class MSTRegistration
		: public MultiViewRegistrationBase<PointT, Scalar, RegistrationT,
		std::conditional_t<Dynamic,
		// 动态最小生成树邻接关系图
		boost::adjacency_list<boost::listS, boost::listS, boost::undirectedS,
		boost::property<boost::vertex_index_t, unsigned, details::DynamicMSTVertexProperty<PointT>>,
		boost::property<boost::edge_index_t, unsigned, boost::property<boost::edge_overlap_ratio_t, float>>>,
		// 普通最小生成树邻接关系图
		boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_index_t, unsigned>,
		boost::property<boost::edge_index_t, unsigned, boost::property<boost::edge_overlap_ratio_t, float>>>
		>>
	{
	public:
		using Ptr = std::shared_ptr<MSTRegistration>;
		using ConstPtr = std::shared_ptr<const MSTRegistration>;


		using MultiViewRegistrationBase = MultiViewRegistrationBase<PointT, Scalar, RegistrationT,
			std::conditional_t<Dynamic,
			boost::adjacency_list<boost::listS, boost::listS, boost::undirectedS,
			boost::property<boost::vertex_index_t, unsigned, details::DynamicMSTVertexProperty<PointT>>,
			boost::property<boost::edge_index_t, unsigned, boost::property<boost::edge_overlap_ratio_t, float>>>,
			boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_index_t, unsigned>,
			boost::property<boost::edge_index_t, unsigned, boost::property<boost::edge_overlap_ratio_t, float>>>
			>>;


		using PointCloud = typename MultiViewRegistrationBase::PointCloud;
		using PointCloudPtr = typename MultiViewRegistrationBase::PointCloudPtr;
		using PointCloudConstPtr = typename MultiViewRegistrationBase::PointCloudConstPtr;
		using PointCloudPtrVector = typename MultiViewRegistrationBase::PointCloudPtrVector;

		using Matrix4 = typename MultiViewRegistrationBase::Matrix4;
		using Matrix4Vector = typename MultiViewRegistrationBase::Matrix4Vector;

		using Registration = typename MultiViewRegistrationBase::Registration;
		using RegistrationPtr = typename MultiViewRegistrationBase::RegistrationPtr;
		using RegistrationConstPtr = typename MultiViewRegistrationBase::RegistrationConstPtr;

		using Graph = typename MultiViewRegistrationBase::Graph;
		using vertex_iterator = typename boost::graph_traits<Graph>::vertex_iterator;
		using edge_iterator = typename boost::graph_traits<Graph>::edge_iterator;
		using vertex_descriptor = typename boost::graph_traits<Graph>::vertex_descriptor;
		using edge_descriptor = typename boost::graph_traits<Graph>::edge_descriptor;

		MSTRegistration(bool use_template = false, float overlap_adj_point_dis = 0.7f, float overlap_th = 0.1f) :
			MultiViewRegistrationBase("MSTRegistration", true, use_template, overlap_adj_point_dis, overlap_th)
		{
			if (Dynamic)	// 当执行动态最小生成树的多视图点云配准时，必须借助模板点云构建邻接关系图
				use_template_blade_ = true;
		}

		~MSTRegistration() override = default;

		using MultiViewRegistrationBase::printGraph;

		void printGraph(void) const override;


	protected:
		using MultiViewRegistrationBase::getClassName;

	private:
		virtual bool computeTransforms(void) override;

	protected:
		using MultiViewRegistrationBase::multiview_reg_name_;
		using MultiViewRegistrationBase::clouds_;
		using MultiViewRegistrationBase::indicess_;
		//using MultiViewRegistrationBase::init_transforms_;
		using MultiViewRegistrationBase::registration_;
		using MultiViewRegistrationBase::transforms_;
		using MultiViewRegistrationBase::full_cloud_;
		using MultiViewRegistrationBase::prebuild_searchs_;
		using MultiViewRegistrationBase::searchs_;

		using MultiViewRegistrationBase::graph_;
		using MultiViewRegistrationBase::use_template_blade_;
		using MultiViewRegistrationBase::overlap_ratio_th_;
		using MultiViewRegistrationBase::overlap_indices_flag_;
		using MultiViewRegistrationBase::template_cloud_search_;

	};
}
#include "mst_registration.hpp"

