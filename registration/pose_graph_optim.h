/**
 *
 *  @file      pose_graph_optim.h
 *  @brief     CUDA implementation of pose graph optimization
 *  @details   ~
 *  @author    ZhangYazheng
 *  @date      12.05.2025
 *  @copyright MIT License.
 *
 */
#pragma once
//#include <iostream>
//#include <vector>
//#include <tuple>
//#include <memory>
//#include <utility>
//#include <type_traits>
//#include <filesystem>
//#include <chrono>
//
//namespace fs = std::filesystem;
//using namespace std::chrono_literals;
//
//#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/graph_traits.hpp>
//#include <boost/property_map/property_map.hpp>
//#include <Eigen/Dense>
//#include <Eigen/Geometry>
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
//#include <sophus/se3.hpp>
//#include <g2o/core/base_vertex.h>
//#include <g2o/core/base_binary_edge.h>
//#include <g2o/core/block_solver.h>
//#include <g2o/core/optimization_algorithm_levenberg.h>
//#include <g2o/core/optimization_algorithm_gauss_newton.h>
//#include <g2o/core/optimization_algorithm_dogleg.h>
//#include <g2o/solvers/eigen/linear_solver_eigen.h>
////#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
////#include <g2o/solvers/csparse/linear_solver_csparse.h>
//
//
//#include "pcl_support/registration/graph_handler.h"
//
//typedef Eigen::Matrix<double, 6, 6> Matrix6d;
//
//// 给定误差求J_R^{-1}的近似
///**
// * .
// * 
// * \param e
// * \return 
// */
//Matrix6d JRInv(const Sophus::SE3d& e) {
//    Matrix6d J;
//    J.block(0, 0, 3, 3) = Sophus::SO3d::hat(e.so3().log());
//    J.block(0, 3, 3, 3) = Sophus::SO3d::hat(e.translation());
//    J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero(3, 3);
//    J.block(3, 3, 3, 3) = Sophus::SO3d::hat(e.so3().log());
//    // J = J * 0.5 + Matrix6d::Identity();
//    J = Matrix6d::Identity();    // try Identity if you want
//    return J;
//}
//
//// 李代数顶点
//typedef Eigen::Matrix<double, 6, 1> Vector6d;
//
//
//class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3d> {
//public:
//    virtual bool read(istream& is) override {
//        double data[7];
//        for (int i = 0; i < 7; i++)
//            is >> data[i];
//        setEstimate(Sophus::SE3d(
//            Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
//            Eigen::Vector3d(data[0], data[1], data[2])
//        ));
//        return true;
//    }
//
//    virtual bool write(ostream& os) const override {
//        os << id() << " ";
//        Eigen::Quaterniond q = _estimate.unit_quaternion();
//        os << _estimate.translation().transpose() << " ";
//        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
//        return true;
//    }
//
//    virtual void setToOriginImpl() override {
//        _estimate = Sophus::SE3d();
//    }
//
//    // 左乘更新
//    virtual void oplusImpl(const double* update) override {
//        Vector6d upd;
//        upd << update[0], update[1], update[2], update[3], update[4], update[5];
//        _estimate = Sophus::SE3d::exp(upd) * _estimate;
//    }
//};
//
//// 两个李代数节点之边
//class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
//public:
//    virtual bool read(istream& is) override {
//        double data[7];
//        for (int i = 0; i < 7; i++)
//            is >> data[i];
//        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
//        q.normalize();
//        setMeasurement(Sophus::SE3d(q, Eigen::Vector3d(data[0], data[1], data[2])));
//        for (int i = 0; i < information().rows() && is.good(); i++)
//            for (int j = i; j < information().cols() && is.good(); j++) {
//                is >> information()(i, j);
//                if (i != j)
//                    information()(j, i) = information()(i, j);
//            }
//        return true;
//    }
//
//    virtual bool write(ostream& os) const override {
//        VertexSE3LieAlgebra* v1 = static_cast<VertexSE3LieAlgebra*> (_vertices[0]);
//        VertexSE3LieAlgebra* v2 = static_cast<VertexSE3LieAlgebra*> (_vertices[1]);
//        os << v1->id() << " " << v2->id() << " ";
//        Sophus::SE3d m = _measurement;
//        Eigen::Quaterniond q = m.unit_quaternion();
//        os << m.translation().transpose() << " ";
//        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";
//
//        // information matrix 
//        for (int i = 0; i < information().rows(); i++)
//            for (int j = i; j < information().cols(); j++) {
//                os << information()(i, j) << " ";
//            }
//        os << endl;
//        return true;
//    }
//    
//    // 误差计算与书中推导一致
//    virtual void computeError() override {
//        Sophus::SE3d v1 = (static_cast<VertexSE3LieAlgebra*> (_vertices[0]))->estimate();
//        Sophus::SE3d v2 = (static_cast<VertexSE3LieAlgebra*> (_vertices[1]))->estimate();
//        _error = (_measurement.inverse() * v1.inverse() * v2).log();
//    }
//
//    // 雅可比计算
//    virtual void linearizeOplus() override {
//        Sophus::SE3d v1 = (static_cast<VertexSE3LieAlgebra*> (_vertices[0]))->estimate();
//        Sophus::SE3d v2 = (static_cast<VertexSE3LieAlgebra*> (_vertices[1]))->estimate();
//        Matrix6d J = JRInv(Sophus::SE3d::exp(_error));
//
//        _jacobianOplusXi = -J * v2.inverse().Adj();
//        _jacobianOplusXj = J * v2.inverse().Adj();
//    }
//};
//
///** \brief BGL adjacency_list 类型别名 */
//template<typename PointT = pcl::PointXYZ, typename VertexT = std::size_t,
//    typename InformationT = Eigen::Matrix<double, EdgeSE3LieAlgebra::Dimension, EdgeSE3LieAlgebra::Dimension>>
//using Graph = boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS,
//    pcl::registration::PoseEstimate<PointT>, pcl::registration::PoseMeasurement<VertexT, InformationT>>;
//
//template<typename GraphT = Graph<>>
//class PoseGraphOptim
//{
//public:
//    // pcl::registartion::GraphHandler 相关类型名
//    using GraphHandler = pcl::registration::GraphHandler<GraphT>;
//    using GraphHandlerPtr = typename GraphHandler::Ptr;
//    using GraphHandlerConstPtr = typename GraphHandler::ConstPtr;
//
//    // GraphT 相关属性
//    using PointT = typename boost::vertex_bundle_type<GraphT>::type::PointType;
//    using VertexT = typename boost::edge_bundle_type<GraphT>::type::VertexType;
//	using InformationT = typename boost::edge_bundle_type<GraphT>::type::InformationType;
//    using VertexD = typename boost::graph_traits<GraphT>::vertex_descriptor;
//	using EdgeD = typename boost::graph_traits<GraphT>::edge_descriptor;
//    using VertexI = typename boost::graph_traits<GraphT>::vertex_iterator;
//	using EdgeI = typename boost::graph_traits<GraphT>::edge_iterator;
//    using VertexST = typename boost::graph_traits<GraphT>::vertices_size_type;
//    using EdgeST = typename boost::graph_traits<GraphT>::edges_size_type;
//
//    // g2o 不同优化算法(Optimization Algorithm)及线性求解器(linear Solver)标签
//    enum struct OptimAlgm_G2O { LevenBerg, GaussNewton, DogLeg };
//    enum struct LinearSolver_G2O { Eigen, Cholmod, Csparse };
//
//    
//
//    /**
//     * .
//     * 
//     */
//	PoseGraphOptim() = default;
//
//    /**
//     * .
//     */
//	PoseGraphOptim(VertexST n) :graph_handler_(new GraphHandler(n)), graph_(graph_handler_->getGraph().get()) {};
//
//    /**
//     * .
//     * 
//     * \param clouds
//     * \param init_poses
//     */
//    PoseGraphOptim(std::vector<std::shared_ptr<const pcl::PointCloud<PointT>>>& clouds, std::vector<Eigen::Matrix4f>& init_poses)
//        :graph_handler_(new GraphHandler(clouds.size() == init_poses.size() ? clouds.size() : 0)), graph_(graph_handler_->getGraph().get())
//    {
//        LOG_IF(ERROR, clouds.size() != init_poses.size()) << "点云数量与初始位姿数量不匹配";
//        for (int i = 0; i < clouds.size(); ++i)
//        {
//            addPointCloud(clouds[i], init_poses[i]);
//            registrate();
//        }
//    };
//    //PoseGraphOptim(std::vector<pcl::PointCloud<PointT>::Ptr>&& clouds, std::vector<Eigen::Matrix4f>&& init_poses)     ??????
//
//
//    /**
//     * 添加节点（点云及对应的初始位姿）.
//     * 
//     * \param cloud
//     * \param pose
//     */ 
//    inline void addPointCloud(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const Eigen::Matrix4f& pose)
//    {
//        last_vertices_.push_back(graph_handler_->addPointCloud(cloud, pose));
//    }
//
// 
//    /**
//     * 设置graph_.
//     * 
//     * \param gh
//     */
//    inline void setGraphHandler(GraphHandlerPtr& gh)
//    {
//        graph_handler_ = gh;
//    }
//
//    /** \brief 获得指向 graph_ 的指针 */
//    inline GraphHandlerPtr getGraphHandler()
//    {
//        return graph_handler_;
//    }
//
//    /** \brief 获得指向 graph_ 的常量指针 */
//    inline GraphHandlerConstPtr getGraphHandler() const
//    {
//        return graph_handler_;
//    }
//
//    inline void setOptimAlgm(OptimAlgm_G2O algorithm = OptimAlgm_G2O::LevenBerg)
//    {
//        optim_algm_tag_ = algorithm;
//    }
//
//    inline void setLinearSolver(LinearSolver_G2O solver = LinearSolver_G2O::Eigen)
//    {
//        linear_solver_tag_ = solver;
//    }
//
//    /**
//     * 计算新添加的顶点与已有顶点的约束关系，即进行精配准.
//     * 
//     * \return 
//     */
//    bool registrate();
//
//    /**
//     * 进行图优化.
//     * 
//     * \return 
//     */
//    bool optimize(const unsigned iterations = 30, const bool verbose = true);
//
//    /**
//     * 配准并进行图优化.
//     * 
//     * \return 
//     */
//    bool registAndOptim();
//
//private:
//    /** \brief 图权柄 */
//    GraphHandlerPtr graph_handler_;
//    /** \brief 指向 graph_handler_ 封装的图 */
//    GraphT* graph_ = garph_handler_.get();
//    /** \brief 上一个已配准节点的索引 */
//    VertexD last_aligned_vertex_;
//    /** \brief 图中已有节点的全部索引（顺序push_back） */
//    std::vector<VertexD> last_vertices_;
//    /** \brief 选择的g2o优化算法 */
//    OptimAlgm_G2O optim_algm_tag_ = OptimAlgm_G2O::LevenBerg;
//    /** \brief 选择的g2o线性求解算法 */
//    LinearSolver_G2O linear_solver_tag_ = LinearSolver_G2O::Eigen;
//};
//
//
//template<typename GraphT>
//bool PoseGraphOptim<GraphT>::registrate()
//{
//    if (last_vertices_.empty())
//        return;
//
//    //
//    // TO DO: computeRegistration();
//    //
//
//    last_aligned_vertex_ = last_vertices_.back();
//    last_vertices_.clear();
//}
//
//template<typename GraphT>
//bool PoseGraphOptim<GraphT>::optimize(const unsigned iterations, const bool verbose)
//{
//    if (!graph_handler_ || (boost::num_vertices(*(graph_handler_->getGraph())) < 2) || (boost::num_edges(*(graph_handler_->getGraph())) < 1))
//    {
//        std::cout << "graph_ 为空, 或顶点、边的数量不足 (" << __FILE__ << ", " << __LINE__ << ")\n";
//        return false;
//    }
//
//    // 设置g2o求解器
//    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<VertexSE3LieAlgebra::Dimension, EdgeSE3LieAlgebra::Dimension>>;
//
//    //switch (linear_solver_tag_)
//    //{
//    //case LinearSolver_G2O::Eigen: using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>; break;
//    //case LinearSolver_G2O::Cholmod: using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>; break;
//    //case LinearSolver_G2O::Csparse: using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>; break;
//    //default:using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>; break;
//    //}
//    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
//
//    //switch (optim_algm_tag_)
//    //{
//    //case OptimAlgm_G2O::LevenBerg: auto solver =
//    //    new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>(())); break;
//    //default:
//    //    break;
//    //}
//    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//
//    g2o::SparseOptimizer optimizer;
//    optimizer.setAlgorithm(solver);
//    optimizer.setVerbose(verbose);
//
//    // 引入一个 vector 记录G2O图中添加的顶点，以便后续利用G2O中的顶点更新 graph_ 中的信息
//    int counter_ = 0;
//    std::vector<VertexSE3LieAlgebra*> vertices_(boost::num_vertices(*graph_));
//
//    // 向G2O图中添加顶点(vertices)
//    VertexI iter1_v, iter2_v;
//    for (std::tie(iter1_v, iter2_v) = boost::vertices(*graph_); iter1_v != iter2_v; ++iter1_v)
//    {
//        VertexSE3LieAlgebra* v = new VertexSE3LieAlgebra();
//        v->setEstimate(Sophus::SE3d((*graph_)[*iter1_v].pose.matrix()));
//        v->setId((*graph_)[*iter1_v].id);
//        if ((*graph_)[*iter1_v].id == 0)
//            v->setFixed(true);
//        optimizer.addVertex(v);
//        vertices_[(*graph_)[*iter1_v].id] = v;
//    }
//
//    // 向G2O图中添加边(edges)
//    EdgeI iter1_e, iter2_e;
//    for (std::tie(iter1_e, iter2_e) = boost::edges(*graph_); iter1_e != iter2_e; ++iter1_e)
//    {
//        EdgeSE3LieAlgebra* e = new EdgeSE3LieAlgebra();
//        e->setMeasurement(Sophus::SE3d((*graph_)[*iter1_e].relative_transformation.matrix()));
//        e->setId((*graph_)[*iter1_e].id);
//        e->setInformation((*graph_)[*iter1_e].information_matrix);
//        e->setVertex(0, optimizer.vertices()[(*graph_)[boost::source(*iter1_e, *graph_)].id]);
//        e->setVertex(1, optimizer.vertices()[(*graph_)[boost::target(*iter1_e, *graph_)].id]);
//        optimizer.addEdge(e);
//    }
//
//    // 优化
//    optimizer.initializeOptimization();
//    optimizer.optimize(iterations);
//
//    // 将优化后的位姿更新至 graph_ 顶点内
//    for (std::tie(iter1_v, iter2_v) = boost::vertices(*graph_); iter1_v != iter2_v; ++iter1_v)
//        (*graph_)[*iter1_v].pose = vertices_[(*graph_)[*iter1_v].id]->estimate().matrix();
//
//    return true;
//}
//
//template<typename GraphT>
//bool PoseGraphOptim<GraphT>::registAndOptim()
//{
//
//}
//
