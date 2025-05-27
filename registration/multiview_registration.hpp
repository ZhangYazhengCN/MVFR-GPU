#pragma once
#include <utility>
#include <chrono>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <boost/graph/graphviz.hpp>

#include "multiview_registration.h"

namespace mvfr
{
	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	inline bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::alignclouds(void)
	{
		//if (!initCompute() || !computeTransforms())
		//	return false;

		if (!initCompute())
			return false;

		const auto begin = std::chrono::system_clock::now();
		if (!computeTransforms())
			return false;
		const auto end = std::chrono::system_clock::now();

		std::cout << "配准用时:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin) << '\n';


		//if (!full_cloud_)
		//{
		//	full_cloud_.reset(new pcl::PointCloud<PoinT>);
		//	*full_cloud += *(clouds_[0]);
		//	for (int i = 1; i < clouds_.size(); ++i)
		//	{
		//		pcl::PointCloud<PointT> cloud_temp_;
		//		pcl::transformPointCloud(*(clouds_[i]), cloud_temp_, transforms_[i - 1]);
		//		*full_cloud_ += cloud_temp_;
		//	}
		//}

		return deinitCompute();
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	inline bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::alignclouds(const PointCloudPtrVector& clouds, PointCloudPtr& full_cloud, const RegistrationPtr& registration, const IndicesPtrVector& indicess)
	{
		// 初始化
		setClouds(clouds, indicess);
		//if (!init_transforms.empty()) setInitTransforms(init_transforms);
		if (registration) setRegistrationMethod(registration);

		// 执行多视图精配准
		if (alignclouds())
		{
			full_cloud = full_cloud_;
			return true;
		}
		else
			return false;
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	inline bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::alignclouds(const SearchPtrVector& searchs, PointCloudPtr& full_cloud, const RegistrationPtr& registration)
	{
		// 初始化
		setSearchs(searchs);
		//if (!init_transforms.empty()) setInitTransforms(init_transforms);
		if (registration) setRegistrationMethod(registration);


		if (alignclouds())
		{
			full_cloud = full_cloud_;
			return true;
		}
		else
			return false;
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::buildGraph(void)
	{
		if (!build_graph_)
		{
			PCL_ERROR("[% s::buildGraph] 当前实例不要求构建邻接关系图! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return false;
		}

		if (update_clouds_ || update_searchs_)
			if (!updateCloudsAndSearchs())
				return false;

		// 构建邻接关系矩阵
		if (update_graph_)
		{
			Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> overlap_ratio_array;
			if (!(use_template_blade_ ? calcuOverlapRatioUseTemplate(overlap_ratio_array) : calcuOverlapRatio(overlap_ratio_array)))
			{
				PCL_ERROR("[% s::buildGraph] 点云邻接关系图构建失败! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				return false;
			}

			// 检查点云的邻接关系是否充足
			// @note 存在孤立点云的充分条件(满足该条件时，多视图点云中仍存在孤立点的可能)
			if ((overlap_ratio_array > 0).count() < clouds_.size() - 1)
			{
				PCL_ERROR("[% s::buildAdjGraph] 存在孤立点云，请检查输入点云数据是否正确! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				return false;
			}

			// 计算邻接边并构建邻接关系图
			std::vector<std::pair<unsigned, unsigned>> overlap_edges;
			overlap_edges.reserve(clouds_.size() * (clouds_.size() - 1) / 2);
			for (int i = 0; i < overlap_ratio_array.rows(); ++i)
				for (int j = 0; j < overlap_ratio_array.cols(); ++j)
					if (overlap_ratio_array(i, j) > 0)
						overlap_edges.push_back(std::make_pair(i, j));
			graph_.reset(new Graph(overlap_edges.begin(), overlap_edges.end(), clouds_.size()));

			// 为vertex_index赋值
			// 当 VertexListS 不为 boost::vecS 时，为 vertex_index 赋值。若 VertexListS 为 boost::vecS，则 boost::property_traits<vertex_index_map_type>::category == boost::readable_property_map_tag
			using vertex_index_map_type = typename boost::property_map<Graph, boost::vertex_index_t>::type;
			vertex_index_map_type vertex_index_map = boost::get(boost::vertex_index, *graph_);
			if constexpr (std::is_base_of_v<boost::writable_property_map_tag, boost::property_traits<vertex_index_map_type>::category>)
			{
				typename boost::graph_traits<Graph>::vertex_iterator vertex_begin, vertex_end;

				std::tie(vertex_begin, vertex_end) = boost::vertices(*graph_);
				for (int i = 0, i_end = std::distance(vertex_begin, vertex_end); i != i_end; ++i)
					vertex_index_map[*vertex_begin++] = i;
			}

			// 为 edge_index, edge_overlap_ratio 赋值 
			typename boost::graph_traits<Graph>::edge_iterator edge_begin, edge_end;
			typename boost::property_map<Graph, boost::edge_index_t>::type edge_index_map = boost::get(boost::edge_index, *graph_);
			typename boost::property_map<Graph, boost::edge_overlap_ratio_t>::type edge_overlap_ratio_map = boost::get(boost::edge_overlap_ratio, *graph_);
			std::tie(edge_begin, edge_end) = boost::edges(*graph_);
			for (int i = 0, i_end = std::distance(edge_begin, edge_end); i != i_end; ++i)
			{
				edge_index_map[*edge_begin] = i;
				edge_overlap_ratio_map[*edge_begin] = overlap_ratio_array(
					vertex_index_map(boost::source(*edge_begin, *graph_)),
					vertex_index_map(boost::target(*edge_begin, *graph_)));
				++edge_begin;
			}

			// 重置点云更新标志
			update_graph_ = false;
		}

		return true;
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	void MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::printGraph(void) const
	{
		if (graph_ == nullptr)
		{
			PCL_ERROR("[% s::printAdjGraph] 不存在邻接关系图，请检查邻接关系图是否构建成功! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return;
		}

		std::cout << "AdjGraph的顶点数量为：" << boost::num_vertices(*graph_) << ", 邻接边的数量为：" << boost::num_edges(*graph_) << '\n';
		std::cout << "AdjGraph的结构如下：\n";
		typename boost::graph_traits<Graph>::vertex_iterator vertex_begin, vertex_end;
		typename boost::graph_traits<Graph>::out_edge_iterator edge_begin, edge_end;

		auto vertex_index_map = boost::get(boost::vertex_index, *graph_);
		auto edge_overlap_ratio_map = boost::get(boost::edge_overlap_ratio, *graph_);

		for (std::tie(vertex_begin, vertex_end) = boost::vertices(*graph_); vertex_begin != vertex_end; ++vertex_begin)
		{
			std::cout << vertex_index_map[*vertex_begin] << ": ";
			for (std::tie(edge_begin, edge_end) = boost::out_edges(*vertex_begin, *graph_); edge_begin != edge_end; ++edge_begin)
				std::cout << "(" << vertex_index_map[boost::source(*edge_begin, *graph_)] << ", "
				<< vertex_index_map[boost::target(*edge_begin, *graph_)] << ", "
				<< edge_overlap_ratio_map[*edge_begin] << ") ";
			//std::cout << *out_e_begin << ' ';
			std::cout << '\n';
		}
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::writeGraph(std::ostream& os)
	{
		if (graph_ == nullptr && !buildGraph())
		{
			PCL_WARN("[% s::writeGraph] 没有可以保存的邻接关系图! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return false;
		}

		auto vertex_index_map = boost::get(boost::vertex_index, *graph_);
		auto edge_index_map = boost::get(boost::edge_index, *graph_);
		auto edge_overlap_ratio_map = boost::get(boost::edge_overlap_ratio, *graph_);

		boost::dynamic_properties dp;
		dp.property("node_id", vertex_index_map);
		dp.property("edge_id", edge_index_map);
		dp.property("edge_overlap_ratio", edge_overlap_ratio_map);
		boost::write_graphviz_dp(os, *graph_, dp);
		return true;
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	void MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::readGraph(std::istream& is)
	{
		graph_.reset(new Graph());

		auto vertex_index_map = boost::get(boost::vertex_index, *graph_);
		auto edge_index_map = boost::get(boost::edge_index, *graph_);
		auto edge_overlap_ratio_map = boost::get(boost::edge_overlap_ratio, *graph_);

		boost::dynamic_properties dp;
		dp.property("node_id", vertex_index_map);
		dp.property("edge_id", edge_index_map);
		dp.property("edge_overlap_ratio", edge_overlap_ratio_map);

		boost::read_graphviz(is, *graph_, dp);

		// 当从文件中读取邻接关系图后，class 内部就不再构建邻接关系图了
		build_graph_ = false;
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::initCompute(void)
	{
		// 更新点云及搜索树
		if (!updateCloudsAndSearchs())
			return false;


		const auto begin = std::chrono::system_clock::now();
		// 构建邻接关系图
		if (build_graph_ && !buildGraph())
			return false;
		const auto end = std::chrono::system_clock::now();

		std::cout << "建图用时:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin) << '\n';


		// 检查配准方法是否正确设置
		if (!registration_)
		{
			PCL_WARN("[% s::initCompute] 未指定配准方法，已生成指定类型默认参数的配准方法! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			registration_.reset(new Registration);
		}

		//// 检查初始位姿是否正确设置
		//if (init_transforms_.size() != clouds_.size() - 1)
		//{
		//	PCL_WARN("[% s::initCompute] 初始位姿与待配准的点云数量不匹配，已将全部初始位姿设置为单位矩阵! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
		//	init_transforms_.resize(clouds_.size() - 1, Matrix4::Identity());
		//}

		// 重置配准结果
		transforms_.resize(clouds_.size() - 1, Matrix4::Identity());
		full_cloud_.reset(new PointCloud);

		return true;
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::updateCloudsAndSearchs(void)
	{
		// 检查是否正确输入输入点云（及有效索引）和搜索树
		if (update_searchs_ && prebuild_searchs_)
		{
			if (searchs_.empty() || searchs_.size() < 2)
			{
				PCL_ERROR("[% s::updateCloudsAndSearchs] 待配准的搜索树数量不足! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				return false;
			}

			clouds_.resize(searchs_.size());
			indicess_.resize(searchs_.size());
			for (int i = 0; i < searchs_.size(); ++i)
			{
				clouds_[i] = std::const_pointer_cast<PointCloud>(searchs_[i]->getInputCloud());
				indicess_[i] = std::const_pointer_cast<pcl::Indices>(searchs_[i]->getIndices());
			}

			update_searchs_ = false;
			update_clouds_ = false;
		}
		else if (update_clouds_)
		{
			if (clouds_.empty() || clouds_.size() < 2)
			{
				PCL_ERROR("[% s::updateCloudsAndSearchs] 待配准的点云数量不足! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				return false;
			}

			if (clouds_.size() != indicess_.size())
			{
				PCL_ERROR("[% s::updateCloudsAndSearchs] 有效索引数量与待配准的点云数量不匹配! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
				return false;
			}

			searchs_.resize(clouds_.size());
			for (int i = 0; i < clouds_.size(); ++i)
			{
				searchs_[i].reset(new Search);
				searchs_[i]->setInputCloud(clouds_[i], indicess_[i]);
			}

			update_searchs_ = false;
			update_clouds_ = false;
		}
		return true;
	}

	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::calcuOverlapRatio(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>& overlap_ratio_array)
	{
		overlap_ratio_array.resize(clouds_.size(), clouds_.size());
		overlap_ratio_array.setZero();

		// 计算重叠区域大小及比率
		std::vector<pcl::Indices> indices;
		std::vector<std::vector<float>> distances;
		for (int i = 0; i < clouds_.size(); ++i)
			for (int j = i + 1; j < clouds_.size(); ++j)
			{
				searchs_[i]->nearestKSearch(*(clouds_[j]), pcl::Indices(), 1, indices, distances);
				unsigned counter_temp = thrust::count_if(thrust::host, distances.begin(), distances.end(),
					[this](const std::vector<float>& dis) -> bool
					{
						return dis[0] <= this->overlap_adj_point_dis_;
					});

				// 仅当重叠区域大小在源点云 clouds_[j] 或目标点云 clouds_[i] 中的占比大于等于 overlap_ratio_th_ 时，才视为 clouds_[i] 与 clouds_[j] 邻接
				if ((float)counter_temp / clouds_[i]->size() >= overlap_ratio_th_ || (float)counter_temp / clouds_[j]->size() >= overlap_ratio_th_)
					overlap_ratio_array(i, j) = counter_temp;
			}
		overlap_ratio_array /= overlap_ratio_array.maxCoeff();
		return true;
	}


	template<typename PointT, typename Scalar, template<typename, typename, typename> typename RegistrationT, typename GraphT>
	bool MultiViewRegistrationBase<PointT, Scalar, RegistrationT, GraphT>::calcuOverlapRatioUseTemplate(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>& overlap_ratio_array)
	{
		if (!template_cloud_search_ || !template_cloud_search_->getInputCloud())
		{
			PCL_ERROR("[% s::buildAdjGraphUseTemplate] 模板点云搜索树为空，或者搜索树内没有点云! %s(%d)\n", getClassName().c_str(), __FILE__, __LINE__);
			return false;
		}

		// 初始化重叠区域标签数组
		overlap_indices_flag_.resize(template_cloud_search_->getInputCloud()->size(), clouds_.size());
		overlap_indices_flag_.setConstant(false);

		// 计算 clouds_ 中每片点云与模板点云的重叠区域
		std::vector<pcl::Indices> indices;
		std::vector<std::vector<float>> distances;
		using ZipIter = thrust::zip_iterator<thrust::tuple<std::vector<pcl::Indices>::iterator, std::vector<std::vector<float>>::iterator>>;
		for (int i = 0; i < clouds_.size(); ++i)
		{
			// @todo 在GPU上采用不同的执行策略？
			//if constexpr (std::is_same_v<Registration, IterativeClosestPointCuda<PointT, PointT, Scalar>>){}

			template_cloud_search_->nearestKSearch(*(clouds_[i]), pcl::Indices(), 1, indices, distances);
			thrust::for_each(
				ZipIter(thrust::make_tuple(indices.begin(), distances.begin())),
				ZipIter(thrust::make_tuple(indices.end(), distances.end())),
				[i, this](const ZipIter::value_type& i_d) {
					if (thrust::get<1>(i_d)[0] <= this->overlap_adj_point_dis_)
						this->overlap_indices_flag_(thrust::get<0>(i_d)[0], i) = true;
				});
		}

		// 计算 clouds_ 间重叠区域的比率
		overlap_ratio_array.resize(clouds_.size(), clouds_.size());
		overlap_ratio_array.setZero();
		for (int i = 0; i < clouds_.size(); ++i)
			for (int j = i + 1; j < clouds_.size(); ++j)
			{
				// 计算 clouds_[i] 与 clouds_[j] 间的重叠区域相对于模板点云的占比
				float ratio_temp = (overlap_indices_flag_.col(i) && overlap_indices_flag_.col(j)).count() / (float)(template_cloud_search_->getInputCloud()->size());

				// 仅当重叠区域比率大于等于 overlap_ratio_th_ 时，才视为 clouds_[i] 与 clouds_[j] 邻接
				if (ratio_temp >= overlap_ratio_th_)
					overlap_ratio_array(i, j) = ratio_temp;
			}
		return true;
	}
}