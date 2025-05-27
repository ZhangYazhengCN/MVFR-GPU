/**
 *
 *  @file      multiview_registration.h
 *  @brief     CUDA implementation of multi-view point cloud fine registration abstract class
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
#include <vector>
#include <string>
#include <memory>
#include <type_traits>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/graph/adjacency_list.hpp>
#include <pcl/point_cloud.h>

#include <pcl/registration/registration.h>

#include "icp_cuda.h"


#define BOOST_INSTALL_PROPERTY_temp(KIND, NAME)                \
    template <> struct property_kind< KIND##_##NAME##_t > \
    {                                                     \
        typedef KIND##_property_tag type;                 \
    }

#define BOOST_DEF_PROPERTY_temp(KIND, NAME)        \
    enum KIND##_##NAME##_t { KIND##_##NAME }; \
    BOOST_INSTALL_PROPERTY_temp(KIND, NAME)

namespace boost
{
	///// 定义vertex_point_cloud 标签（用作boost graph vertex 属性）
	//BOOST_DEF_PROPERTY_temp(vertex, point_cloud);
	
	/// 定义 edge_overlap_ratio 标签（用作 boost graph edge 属性）
	BOOST_DEF_PROPERTY_temp(edge, overlap_ratio);
}

#undef BOOST_DEF_PROPERTY_temp
#undef BOOST_INSTALL_PROPERTY_temp

namespace mvfr
{
	/**
	 *  @class   MultiViewRegistrationBase
	 *  @brief   多视图点云精配准基类
	 *  @details 将多个视图的局部点云变换至统一的坐标系下（默认以第一个视图为基准），并生成最终对齐的完整点云
	 *  @tparam  PointT 点云类型
	 *  @tparam  Scalar 计算精度
	 *  @tparam  RegistrationT 两视图配准方法
	 *  @tparam  GraphT 邻接关系图
	 *
	 *	@note 执行多视图精配准前，需预先对多视图点云预对齐，将其变换至同一个坐标系下
	 *
	 *	@note 借助模板点云构建邻接关系图时，需调用者自行确保模板点云的密度小于待配准多视图点云的平均密度（以确保重叠区域的计算正确）
	 *
	 *	@note 模板参数 \c GraphT 可由派生类按需修改顶点及边的数据结构并“扩展”相应的属性。
	 *	特别注意当 boost::adjacency_list 的 VertexListS 不为 boost::vecS 时，需要在 VertexProperty 处添加属性 boost::vertex_index_t
	 */
	template<typename PointT = pcl::PointXYZ, typename Scalar = double,
		template<typename, typename, typename> typename RegistrationT = IterativeClosestPointCuda,
		typename GraphT = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property,
		boost::property<boost::edge_index_t, unsigned, boost::property<boost::edge_overlap_ratio_t, float>>>>
		class MultiViewRegistrationBase
	{
	public:
		using Ptr = std::shared_ptr<MultiViewRegistrationBase>;
		using ConstPtr = std::shared_ptr<const MultiViewRegistrationBase>;

		using PointCloud = pcl::PointCloud<PointT>;
		using PointCloudPtr = typename PointCloud::Ptr;
		using PointCloudConstPtr = typename PointCloud::ConstPtr;
		using PointCloudPtrVector = std::vector<PointCloudPtr>;

		using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
		using Matrix4Vector = std::vector<Matrix4>;

		using Registration = RegistrationT<PointT, PointT, Scalar>;
		using RegistrationPtr = Registration::Ptr;
		using RegistrationConstPtr = Registration::ConstPtr;

		using Graph = GraphT;

		using IndicesPtr = pcl::IndicesPtr;
		using IndicesConstPtr = pcl::IndicesConstPtr;
		using IndicesPtrVector = std::vector<IndicesPtr>;

		using Search = typename Registration::KdTree;
		using SearchPtr = typename Search::Ptr;
		using SearchConstPtr = typename Search::ConstPtr;
		using SearchPtrVector = std::vector<SearchPtr>;

		MultiViewRegistrationBase(const std::string& class_name = "MultiViewRegistrationBase",
			bool build_graph = false, bool use_template = false,
			float overlap_adj_point_dis = 0.7f, float overlap_th = 0.1f) :
			multiview_reg_name_(class_name),
			registration_(new Registration),
			build_graph_(build_graph),
			use_template_blade_(use_template),
			overlap_adj_point_dis_(overlap_adj_point_dis),
			overlap_ratio_th_(overlap_th)
		{
		}

		virtual ~MultiViewRegistrationBase() = default;


		/*##########################################################################################################
		*******************************  基本接口（输入点云、搜索树、配准方法等） **************************************
		##########################################################################################################*/

		/**
		 * @brief 设置待配准的多视图点云及其有效索引.
		 *
		 * @param clouds 待配准的多视图点云
		 * @param indicess 多视图点云的有效索引
		 */
		inline virtual void setClouds(const PointCloudPtrVector& clouds, const IndicesPtrVector& indicess = IndicesPtrVector())
		{
			// 仅设置非空点云
			clouds_.clear();
			clouds_.reserve(clouds.size());
			for (const auto& cloud_ptr : clouds)
				if (cloud_ptr != nullptr && !cloud_ptr->empty())
					clouds_.push_back(cloud_ptr);

			// 设置有效索引
			if (indicess.empty())
				indicess_.resize(clouds_.size());
			else
				indicess_ = indicess;

			searchs_.clear();
			prebuild_searchs_ = false;
			update_clouds_ = true;
			update_graph_ = true;
		}

		/**
		 * @brief 获取待配准的多视图点云.
		 *
		 * @return 多视图点云
		 */
		inline PointCloudPtrVector& getClouds(void)
		{
			return clouds_;
		}

		/**
		 * @brief 获取多视图点云的有效索引.
		 *
		 * @return 多视图点云的有效索引
		 */
		inline IndicesPtrVector& getIndicess(void)
		{
			return indicess_;
		}

		/**
		 * @brief 设置预构建的搜索树.
		 *
		 * @param searchs 预构建的搜索树
		 */
		inline virtual void setSearchs(const SearchPtrVector& searchs)
		{
			// 仅设置非空搜索树
			searchs_.clear();
			searchs_.reserve(searchs.size());
			for (const auto& search_ptr : searchs)
				if (search_ptr != nullptr && search_ptr->getInputCloud() != nullptr && !search_ptr->getInputCloud()->empty())
					searchs_.push_back(search_ptr);

			clouds_.clear();
			indicess_.clear();
			prebuild_searchs_ = true;
			update_searchs_ = true;
			update_graph_ = true;
		}

		/**
		 * @brief 获取预构建的搜索树.
		 *
		 * @return 预构建的搜索树
		 */
		inline SearchPtrVector& getSearchs(void)
		{
			return searchs_;
		}

		///**
		// * @brief 设置相邻点云间的初始位姿.
		// */
		//inline void setInitTransforms(const Matrix4Vector& init_transforms)
		//{
		//	init_transforms_ = init_transforms;
		//}

		///**
		// * @brief 获取多视图点云的初始位姿.
		// */
		//inline Matrix4Vector& getInitTransforms(void)
		//{
		//	return init_transforms_;
		//}

		/**
		 * @brief 设置两视图点云配准算法.
		 */
		inline void setRegistrationMethod(const RegistrationPtr& registration)
		{
			registration_ = registration;
		}

		/**
		 * @brief @brief 获取两视图点云配准算法..
		 */
		inline RegistrationPtr& getRegistrationMethod(void)
		{
			return registration_;
		}

		/** @brief 获取计算的多视图点云间的绝对变换位姿（以第一幅点云为参考）. */
		inline Matrix4Vector& getFinalTransforms(void)
		{
			return transforms_;
		}

		/** @brief 获取配准后的完整点云. */
		inline PointCloudPtr& getFullCloud(void)
		{
			return full_cloud_;
		}

		/** @brief 多视图点云配准. */
		bool alignclouds(void);

		inline bool operator()(void)
		{
			alignclouds();
		}

		/**
		 * @brief 多视图点云配准（重载版）.
		 *
		 * @param clouds[in] 待配准的多视图点云
		 * @param full_cloud[out] 配准后的完整点云
		 * @param registration[in] 两视图点云配准方法
		 * @param init_transforms[in] 初始位姿
		 * @param indicess[in] 多视图点云的有效索引
		 * @return 是否完成配准
		 *
		 * @note registration 默认参数将会覆盖 setRegistrationMethod 指定的配准方法，从而导致配准方法为空
		 */
		bool alignclouds(const PointCloudPtrVector& clouds, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr(),
			const IndicesPtrVector& indicess = IndicesPtrVector());

		inline bool operator()(const PointCloudPtrVector& clouds, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr(),
			const IndicesPtrVector& indicess = IndicesPtrVector())
		{
			alignclouds(clouds, full_cloud, registration, indicess);
		}
		//bool alignclouds(const PointCloudPtrVector& clouds, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr(),
		//	const Matrix4Vector& init_transforms = Matrix4Vector(), const IndicesPtrVector& indicess = IndicesPtrVector());
		//
		//inline bool operator()(const PointCloudPtrVector& clouds, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr(),
		//	const Matrix4Vector& init_transforms = Matrix4Vector(), const IndicesPtrVector& indicess = IndicesPtrVector())
		//{
		//	alignclouds(clouds, full_cloud, registration, init_transforms, indicess);
		//}

		/**
		 * @brief 多视图点云配准（重载版）.
		 *
		 * @param clouds[in] 预构建的搜索树
		 * @param full_cloud[out] 配准后的完整点云
		 * @param registration[in] 两视图点云配准方法
		 * @param init_transforms[in] 初始位姿
		 * @return 是否完成配准
		 *
		 * @note registration 默认参数将会覆盖 setRegistrationMethod 指定的配准方法，从而导致配准方法为空
		 */
		bool alignclouds(const SearchPtrVector& searchs, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr());

		inline bool operator()(const SearchPtrVector& searchs, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr())
		{
			alignclouds(searchs, full_cloud, registration);
		}
		//bool alignclouds(const SearchPtrVector& searchs, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr(),
		//	const Matrix4Vector& init_transforms = Matrix4Vector());

		//inline bool operator()(const SearchPtrVector& searchs, PointCloudPtr& full_cloud, const RegistrationPtr& registration = RegistrationPtr(),
		//	const Matrix4Vector& init_transforms = Matrix4Vector())
		//{
		//	alignclouds(searchs, full_cloud, registration, init_transforms);
		//}


		/*##########################################################################################################
		*********************************************  邻接关系图相关接口 ********************************************
		##########################################################################################################*/
		/// 设置是否构建邻接关系图
		void setIsBuildGraph(bool is_build = true)
		{
			build_graph_ = is_build;
		}
		/// 获取是否构建邻接关系图
		bool getIsBuildGraph(void) const
		{
			return build_graph_;
		}


		// 设置是否借助模板点云构建邻接关系图
		void setIsUseTemplate(bool use_templae = true)
		{
			use_template_blade_ = use_templae;
		}
		// 获取是否借助模板点云构建邻接关系图
		bool getIsUseTemplate(void) const
		{
			return use_template_blade_;
		}


		/// 设置模板点云
		void setTemplateCloudSearch(const SearchConstPtr& search)
		{
			template_cloud_search_ = search;
		}
		/// 获取模板点云
		SearchConstPtr& getTemplateCloudSearch(void) const
		{
			return template_cloud_search_;
		}


		/// 设置重叠区域近邻点距离阈值（距离小于等于该阈值的点视为处于重叠区域）
		void setOverlapAdjPointsDis(float dis)
		{
			overlap_adj_point_dis_ = dis;
		}
		/// 获取重叠区域近邻点距离阈值（距离小于等于该阈值的点视为处于重叠区域）
		float getOverlapAdjPointsDis(void) const
		{
			return overlap_adj_point_dis_;
		}


		/// 设置重叠区域比例阈值（当重叠区域面积在源点云目标点云或模板点云中的占比大于等于该阈值时，两片点云才视为邻接）
		void setOverlapAreaRatioThreshold(float th)
		{
			overlap_ratio_th_ = th;
		}
		/// 获取重叠区域比例阈值（当重叠区域面积在源点云目标点云或模板点云中的占比大于等于该阈值时，两片点云才视为邻接）
		float getOverlapAreaRatioThreshold(void) const
		{
			return overlap_ratio_th_;
		}


		/// 构建邻接关系图（需要确保存在输入点云或者搜索树，且成员变量 \c build_graph_ 已置为true）
		bool buildGraph(void);
		/// 获得邻接关系图
		const Graph* getGraph(void) const { return graph_.get(); }
		/// 控制台输出邻接关系图结构
		virtual void printGraph(void) const;



		/**
		 * @brief 将构建的邻接关系图写入指定的标准输出流内.
		 * @param os 指定的标准输出流
		 * @return 是否写入成功
		 *
		 * @note 调用该方法前需要成员变量 \c build_graph_ 置为true
		 */
		virtual bool writeGraph(std::ostream& os = stdout);
		/**
		 * @brief 从指定的标准输入流内读取邻接关系图
		 * @param is 指定的标准输入流
		 *
		 * @note 调用该方法时将自动置成员变量 \c build_graph_ 为false
		 */
		virtual void readGraph(std::istream& is = stdin);


	protected:
		/** @brief 获取多视图点云配准算法名称. */
		inline const std::string& getClassName(void) const
		{
			return multiview_reg_name_;
		}

		bool initCompute(void);

		bool deinitCompute(void) { return true; }

	private:
		/**
		 * @brief 计算多视图点云的绝对变换位姿 {transforms_}。纯虚函数，所有派生类都应对其进行覆盖.
		 *
		 * @details 派生函数应完成以下任务：
		 * @li 计算多视图点云的绝对位姿
		 * @li 更新变换后多视图点云对应的搜索树(remove)
		 * @li 生成配准后的完整点云
		 *
		 * @return  绝对位姿是否计算成功
		 */
		virtual bool computeTransforms(void) = 0;

		/// 更新输入点云及相应的搜索树
		bool updateCloudsAndSearchs(void);

		/**
		 * @brief 计算点云簇 clouds_ 间的重叠区域比率 overlap_ratio_array
		 * @param overlap_ratio_array 重叠区域比率. 上三角阵，两片点云间的重叠比率仅保留一份
		 * @return 是否计算成功
		 *
		 * @note 对于 overlap_ratio_array(i, j) ，仅当其代表的重叠区域面积在源点云 clouds_[j] 或者目标点云 clouds_[i] 中的占比 >= overlap_ratio_th_，
		 * 才视为两片点云有效重叠.
		 *
		 * @note 最终 overlap_ratio_array 中保存的重叠区域比率是相对于最大重叠区域面积来计算的.
		 */
		bool calcuOverlapRatio(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>& overlap_ratio_array);

		/**
		 * @brief 借助模板点云计算重叠区域比率
		 * @param overlap_ratio_array 重叠区域比率. 上三角阵，两片点云间的重叠比率仅保留一份
		 * @return 是否计算成功
		 *
		 * @note 最终 overlap_ratio_array 中保存的重叠区域比率是相对于模板点云的面积来计算的.
		 *
		 * @note 仅当 overlap_ratio_array(i, j) >= overlap_ratio_th_，才视为两片点云有效重叠.
		 */
		bool calcuOverlapRatioUseTemplate(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>& overlap_ratio_array);

	protected:
		/** @brief 多视图配准算法名称. */
		std::string multiview_reg_name_;

		/** @brief 待配准的多视图点云. */
		PointCloudPtrVector clouds_;

		/// 是否需要更新点云
		bool update_clouds_ = true;

		/** @brief 待配准的多视图点云有效索引. */
		IndicesPtrVector indicess_;

		///** @brief 多视图中相邻两片点云初始位姿. */
		//Matrix4Vector init_transforms_;

		/** @brief 采用的两视图点云配准算法. */
		RegistrationPtr registration_;

		/** @brief 多视图点云相对于参考系（第一幅点云）的绝对位姿. */
		Matrix4Vector transforms_;

		/** @brief 配准后的整体点云. */
		PointCloudPtr full_cloud_;

		/** @brief 是否采用预构建的搜索树（kdTree、OcTree等），以避免重复构建.（Internel use） */
		bool prebuild_searchs_ = false;

		/** @brief 多视图点云 {clouds_} 对应的搜索树. */
		SearchPtrVector searchs_;

		/// 是否需要更新搜索树
		bool update_searchs_ = true;


		// ------------------------------- 构建邻接关系矩阵相关参数 -------------------------------
		/// 点云邻接关系图
		std::unique_ptr<Graph> graph_;
		/// 是否构建点云的邻接关系图
		bool build_graph_;
		/// 是否需要更新/重建邻接图（Internel use）
		bool update_graph_ = true;


		/// 是否基于模板叶片构建邻接关系图
		bool use_template_blade_;
		/// 模板点云搜索树
		SearchConstPtr template_cloud_search_;
		/// 以模板点云为参考的重叠区域有效区域（overlap_indices_flag_(i,j) == true 表示clouds[j]在模板点云 template_cloud_[i] 处与模板点云重叠）（Internel use）
		Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> overlap_indices_flag_;


		/// 重叠区域近邻点距离阈值/点云组的平均点间隔(最近邻点距离小于等于该阈值的点云视为重叠区域)
		float overlap_adj_point_dis_;
		/// 重叠区域比率阈值（重叠区域在源点云或目标点云中的占比大于等于该阈值的重叠区域视为有效重叠区域） 
		/// 若基于模板点云构建邻接关系矩阵，则通过重叠区域面积相对模板点云计算重叠区域比率
		float overlap_ratio_th_;
	};
}

#include "multiview_registration.hpp"
