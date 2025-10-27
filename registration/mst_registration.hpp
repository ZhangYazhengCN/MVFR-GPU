#pragma once
#include <deque>

#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/graph_utility.hpp>

#include "mst_registration.h"
#include "../utilities/utilities.h"

namespace mvfr
{

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, bool Dynamic>
	inline void MSTRegistration<PointT, Scalar, RegistrationT, Dynamic>::printGraph(void) const
	{
		MultiViewRegistrationBase::printGraph();

		if constexpr (Dynamic)
		{
			std::cout << "AdjGraph各个顶点的属性为：\n";

			vertex_iterator v_begin, v_end;
			auto vertex_index_map = boost::get(boost::vertex_index, *graph_);

			std::cout <<
				"┌──────────┬────────────────────┬────────────────────┬──────────────────────────────┐\n"
				"│ edge id  │     cloud size     │    overlap area    │           cloud ids          │\n"
				"├──────────┼────────────────────┼────────────────────┼──────────────────────────────┤\n";



			for (std::tie(v_begin, v_end) = boost::vertices(*graph_); v_begin != v_end; ++v_begin)
			{
				const auto& ids_vector_temp = (*graph_)[*v_begin].joint_clouds_id;

				std::ostringstream out_string_temp;
				std::ostream_iterator<std::size_t> os_iter(out_string_temp, ",");
				std::copy(ids_vector_temp.begin(), ids_vector_temp.end() - 1, os_iter);
				out_string_temp << ids_vector_temp.back();


				std::cout << "│" << std::setw(10) << vertex_index_map[*v_begin] << "│" << std::setw(20) << (*graph_)[*v_begin].joint_clouds->size() <<
					"│" << std::setw(20) << (*graph_)[*v_begin].overlap_indices_flag.count() << "│" << std::setw(30) << out_string_temp.view() << "│\n";
			}

			std::cout <<
				"└──────────┴────────────────────┴────────────────────┴──────────────────────────────┘\n";
		}

	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, bool Dynamic>
	bool MSTRegistration<PointT, Scalar, RegistrationT, Dynamic>::computeTransforms(void)
	{
		// ****************************************************************************************
		// ----------------------------- 基于最小生成树执行多视图点云配准 ----------------------------
		// ****************************************************************************************
		if constexpr (!Dynamic)
		{
			// ---------------------------------- 1. 采用 prim 算法生成最小生成树 -------------------------------------
			// 对重叠率预处理，使得重叠率越大，权重越小（并且重叠率不为负值）
			edge_iterator e_begin, e_end;
			auto edge_overlap_ratio_map = boost::get(boost::edge_overlap_ratio, *graph_);
			for (std::tie(e_begin, e_end) = boost::edges(*graph_); e_begin != e_end; ++e_begin)
				edge_overlap_ratio_map[*e_begin] = 1.0f - edge_overlap_ratio_map[*e_begin];

			// 执行 prim 最小生成树算法，默认以第一个vertex为最小生成树的根节点
			auto vertex_index_map = boost::get(boost::vertex_index, *graph_);
			std::vector<vertex_descriptor> predecessor_map(boost::num_vertices(*graph_));
			//std::vector<boost::default_color_type> color_map(boost::num_vertices(*graph_));
			boost::prim_minimum_spanning_tree(*graph_,
				boost::make_iterator_property_map(predecessor_map.begin(), vertex_index_map),
				boost::weight_map(edge_overlap_ratio_map));
			//.color_map(boost::make_iterator_property_map(color_map.begin(), vertex_index_map)));


			// -------------------------------------- 2. 检查是否存在孤立点云 -----------------------------------------
			// @todo color_map 在 prim_minimum_spanning_tree 函数内部没有发生变化（need debug?）
			//for (const auto& v_color : color_map)
				//if (v_color == boost::color_traits<boost::default_color_type>::white())
				//{
				//	PCL_ERROR("[% s::computeTransforms] 最小生成树未能连同所有点云，请检查点云数据是否正确! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				//	return false;
				//}
			// 除了起始节点（第0个节点）外，如果当前节点索引与其父节点索引相同，则该节点为孤立节点，存在孤立点云
			for (int i = 1; i < predecessor_map.size(); ++i)
				if (vertex_index_map[predecessor_map[i]] == i)
				{
					PCL_ERROR("[% s::computeTransforms] 最小生成树未能连同所有点云，请检查点云数据是否正确! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
					return false;
				}

			// ----------------------------------------- 3. 执行多视图配准 -------------------------------------------
			// 流程参考（优先级队列 -> 先进先出队列） https://www.boost.org/doc/libs/1_86_0/libs/graph/doc/prim_minimum_spanning_tree.html 伪代码
			pcl::copyPointCloud(*(clouds_[0]), *full_cloud_);
			std::deque<unsigned> target_queue{ 0 };		// 目标点云队列
			while (!target_queue.empty())
			{
				unsigned cur_tgt_id = target_queue.front();		// 当前目标点云的索引
				target_queue.pop_front();

				// 设置精配准算法 registration_ 的目标点云
				if constexpr (std::is_same_v<Registration, IterativeClosestPointCuda<PointT, PointT, Scalar>>)
				{
					// 若采用 ICPCuda ，则检查搜索树的主机点云是否已加载至GPU（若未加载至GPU，则执行icp需要对搜索树进行重建）
					if (searchs_[cur_tgt_id]->getInputDevice().first == nullptr)
					{
						registration_->setSearchMethodTarget(searchs_[cur_tgt_id]);
						registration_->setInputTarget(clouds_[cur_tgt_id]);
					}
					else
					{
						registration_->setSearchMethodTarget(searchs_[cur_tgt_id], true);
						registration_->setInputTargetDevice(clouds_[cur_tgt_id], searchs_[cur_tgt_id]->getInputDevice());
					}
				}
				else
				{
					registration_->setSearchMethodTarget(searchs_[cur_tgt_id], true);
					registration_->setInputTarget(clouds_[cur_tgt_id]);
				}

				// 配准与当前目标点云邻接的源点云
				for (int i = 1; i < predecessor_map.size(); ++i)
					if (vertex_index_map[predecessor_map[i]] == cur_tgt_id)
					{
						PointCloudPtr cloud_temp(new pcl::PointCloud<PointT>);
						registration_->setInputSource(clouds_[i]);
						registration_->align(*cloud_temp);

						if (!registration_->hasConverged())
						{
							PCL_ERROR("[% s::computeTransforms] 点云 %d <- %d 配准失败! %s(%d)\n", getClassName().c_str(), cur_tgt_id, i, __FILE__, __LINE__);
							return false;
						}

						// 若当前目标点云 Pj 不是基准点云（第一片点云）j!=0 ，则需要对源点云 Pi 再次变换到基准点云所在的坐标系中  Pi' = Tij * Tj0 * Pi
						if (cur_tgt_id != 0)
						{
							pcl::transformPointCloud(*cloud_temp, *cloud_temp, transforms_[cur_tgt_id - 1]);
							transforms_[i - 1] =  transforms_[cur_tgt_id - 1] * registration_->getFinalTransformation();
						}
						else
							transforms_[i - 1] = registration_->getFinalTransformation();

						*full_cloud_ += *cloud_temp;
						target_queue.push_back(i);
					}
			}
		}
		// ****************************************************************************************
		// --------------------------- 基于动态最小生成树进行多视图点云配准 ---------------------------
		// ****************************************************************************************
		else
		{
			// ---------------------------- 1. 初始化动态最小生成树邻接关系图顶点属性 DynamicMSTVertexProperty -------------------------
			auto vertex_index_map = boost::get(boost::vertex_index, *graph_);
			vertex_iterator v_begin, v_end;
			for (std::tie(v_begin, v_end) = boost::vertices(*graph_); v_begin != v_end; ++v_begin)
			{
				(*graph_)[*v_begin].joint_clouds_id.reserve(clouds_.size() / 2);
				(*graph_)[*v_begin].joint_clouds_id.push_back(vertex_index_map[*v_begin]);

				(*graph_)[*v_begin].joint_clouds.reset(new PointCloud);
				pcl::copyPointCloud(*clouds_[vertex_index_map[*v_begin]], *(*graph_)[*v_begin].joint_clouds);

				(*graph_)[*v_begin].overlap_indices_flag = overlap_indices_flag_.col(vertex_index_map[*v_begin]);
			}

			// ---------------------------- 2. 执行DynamicMST直至*graph_中边的数量为0 -------------------------

			unsigned counter = 0;
			while (boost::num_edges(*graph_) != 0)
			{
				//std::cout << "############################### 第" << ++counter << "次配准#####################################\n";

				//std::cout << "\n\n";
				//printGraph();
				//std::cout << "\n\n";


				// 2.1 寻找目前重叠率最大的边
				edge_descriptor max_overlap_edge;
				float cur_overlap = 0.0f;

				auto edge_overlap_ratio_map = boost::get(boost::edge_overlap_ratio, *graph_);
				edge_iterator e_begin, e_end;
				for (std::tie(e_begin, e_end) = boost::edges(*graph_); e_begin != e_end; ++e_begin)
					if (edge_overlap_ratio_map[*e_begin] > cur_overlap)
					{
						max_overlap_edge = *e_begin;
						cur_overlap = edge_overlap_ratio_map[*e_begin];
					}

				std::cout << "重叠区域最大的边为：" << vertex_index_map[boost::source(max_overlap_edge, *graph_)] << "<===>" << vertex_index_map[boost::target(max_overlap_edge, *graph_)] << "\n\n";

				// 2.2 对重叠率最大的边所对应的点云进行配准(vertex_index_map 序号较小的成为目标点云)
				vertex_descriptor vertex_src =
					vertex_index_map[boost::source(max_overlap_edge, *graph_)] > vertex_index_map[boost::target(max_overlap_edge, *graph_)] ?
					boost::source(max_overlap_edge, *graph_) : boost::target(max_overlap_edge, *graph_);
				vertex_descriptor vertex_tgt =
					vertex_index_map[boost::source(max_overlap_edge, *graph_)] > vertex_index_map[boost::target(max_overlap_edge, *graph_)] ?
					boost::target(max_overlap_edge, *graph_) : boost::source(max_overlap_edge, *graph_);


				PointCloudPtr cloud_temp(new pcl::PointCloud<PointT>);
				registration_->setInputSource((*graph_)[vertex_src].joint_clouds);
				registration_->setInputTarget((*graph_)[vertex_tgt].joint_clouds);
				registration_->align(*cloud_temp);
				if (!registration_->hasConverged())
				{
					PCL_ERROR("[% s::computeTransforms] 点云 %d <- %d 配准失败! %s(%d)\n", getClassName().c_str(), vertex_index_map[vertex_tgt], vertex_index_map[vertex_src], __FILE__, __LINE__);
					return false;
				}


				// 2.3 更新邻接关系图
				// 首先删除 vertex_src 和 vertex_tgt 的出边和入边
				boost::clear_vertex(vertex_src, *graph_);
				boost::clear_vertex(vertex_tgt, *graph_);

				// 然后将 vertex_src 融合到 vertex_tgt 内
				for (auto&& id_temp : (*graph_)[vertex_src].joint_clouds_id)			// 更新源点云变换位姿, 并将其编号加入目标已连接点云中
				{
                    transforms_[id_temp - 1] =  registration_->getFinalTransformation() * transforms_[id_temp - 1];

					(*graph_)[vertex_tgt].joint_clouds_id.push_back(id_temp);
				}
				*(*graph_)[vertex_tgt].joint_clouds += *cloud_temp;	// 将源点云加入目标点云中
				(*graph_)[vertex_tgt].overlap_indices_flag = (*graph_)[vertex_tgt].overlap_indices_flag || (*graph_)[vertex_src].overlap_indices_flag;	// 更新目标点云重叠区域标志


				// 最后删除 vertex_src， 并更新 vertex_tgt 与图中其它顶点的邻接关系
				boost::remove_vertex(vertex_src, *graph_);
				for (std::tie(v_begin, v_end) = boost::vertices(*graph_); v_begin != v_end; ++v_begin)
					if (*v_begin != vertex_tgt)
					{
						float overlap_ratio_temp = ((*graph_)[vertex_tgt].overlap_indices_flag && (*graph_)[*v_begin].overlap_indices_flag).count() / (float)(template_cloud_search_->getInputCloud()->size());
						if (overlap_ratio_temp >= overlap_ratio_th_)
						{
							edge_descriptor new_edge = boost::add_edge(vertex_tgt, *v_begin, *graph_).first;
							edge_overlap_ratio_map[new_edge] = overlap_ratio_temp;
						}
					}
			}

			// ---------------------------- 3. 判断是否配准成功并保存结果 -------------------------
			if (boost::num_vertices(*graph_) != 1)
			{
				PCL_ERROR("[% s::computeTransforms] 存在孤立点云，多视图点云配准失败! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				return false;
			}
			full_cloud_ = (*graph_)[*boost::vertices(*graph_).first].joint_clouds;
		}

		return true;
	}
}
