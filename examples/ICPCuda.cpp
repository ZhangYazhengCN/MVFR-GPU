/**
 *
 *  @file      ICPCuda.cpp
 *  @brief     ICPCuda test program
 *  @author    ZhangYazheng
 *  @date      26.05.2025
 *  @copyright MIT License.
 *
 */
#include <iostream>
#include <numbers>
#include <random>
#include <type_traits>
#include <concepts>
#include <string>
#include <map>
#include <iterator>
#include <algorithm>
#include <ranges>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include "registration/mst_registration.h"
#include "utilities/utilities.h"


void printHelpMessages(const char* progName)
{
    std::cout << "\n\n"
        << "Usage: "<<progName<<" [options]...\n"
        << "\n"
        << "ICP CUDA algorithm test program based on 3DMatch sub-dataset:\n"
        << "1. read and align source and target PointCloud one by one through gt.log file\n"
        << "2. generate a random rotation matrix and transform the source PointCloud\n"
        << "3. align the transformed source PointCloud to target PointCloud by ICP algorithm\n"
        << "4. record align time and error\n"
        << "\n"
        << "Options:\n"
        << "----------------------------------------------------------------------------------------------------------\n"
        << "-d, --dir               sub-dataset directory or \"./7-scenes-redkitchen\" of current directory by default\n"
        << "-h, --help              display this help and exit\n"
        << "-g, --gpu               enable GPU acceleration\n"
        << "-v, --visual            visualize registration results\n"
        << "--rot_angle             rotation angle(degree) of rotation matrix generated randomly, 3.0d by default\n"
        << "--max_cor_dis           maximum correspondence distance of ICP, 0.06 by default\n"
        << "--tol-err               tolerant rotation error(degree) of ICP, 2.0 by default\n"
        << "-r, --record[=PATH]     record registration results in PATH(must be a csv file)\n"
        << "                            or the current directory whih name \"ICP-{CPU,GPU}.csv\" by default\n"
        << "\n\n";
}


int main(int argc,char** argv)
{
    // --------------------------- show help messages --------------------------
    if(pcl::console::find_switch(argc,argv,"-h") || pcl::console::find_switch(argc,argv,"--help"))
    {
        printHelpMessages(argv[0]);
        return EXIT_SUCCESS;
    }

    // --------------------------- check other flags ---------------------------
    // dataset dir
    std::string dataset_path_;
    (pcl::console::parse_argument(argc,argv,"-d",dataset_path_) != -1)
        || (pcl::console::parse_argument(argc,argv,"--dir",dataset_path_) != -1);
    fs::path dataset_dir(dataset_path_);
    if(!fs::is_directory(dataset_dir))
        dataset_dir = fs::path(R"(7-scenes-redkitchen)");

    // enable gpu
    bool enable_gpu = pcl::console::find_switch(argc,argv,"-g") || pcl::console::find_switch(argc,argv,"--gpu");

    // enable visualization
    bool enable_visual = pcl::console::find_switch(argc,argv,"-v") || pcl::console::find_switch(argc,argv,"--visual");

    // enable record
    std::string record_path_;
    bool enable_record = (pcl::console::parse_argument(argc,argv,"-r",record_path_) != -1)
        || (pcl::console::parse_argument(argc,argv,"--record",record_path_) != -1);
    fs::path record_file_path(record_path_);
    if(enable_record && (!fs::is_regular_file(record_file_path) || record_file_path.extension()!=".csv"))
    {
        record_file_path = enable_gpu ? fs::path(R"(ICP-GPU.csv)") : fs::path(R"(ICP-CPU.csv)");
    }
    
    // random ratation angle
    double rot_angle;
    if(pcl::console::parse_argument(argc,argv,"--rot_angle", rot_angle) == -1)
        rot_angle = 3.0;

    // maximum correspondence distance
    double max_cor_dis;
    if(pcl::console::parse_argument(argc,argv,"--max_cor_dis", max_cor_dis) == -1)
        max_cor_dis = 0.06;

    // tolerant error
    double tol_err;
    if(pcl::console::parse_argument(argc,argv,"--tol_err", tol_err) == -1)
        tol_err = 2.0;

    // output program settings information
    std::cout<< "Program Settings:\n" << std::boolalpha
        << "dataset directory:      " << fs::absolute(dataset_dir) << '\n'
        << "enable GPU:             " << enable_gpu << '\n'
        << "enable visual:          " << enable_visual << '\n'
        << "random rotation angle:  " << rot_angle << '\n'
        << "max correspondence dis: " << max_cor_dis << '\n'
        << "tolerant rotation error:" << tol_err << '\n'
        << "enable record:          " << enable_record << '\n';
    if(enable_record)
        std::cout << "record path:       " << fs::absolute(record_file_path) << '\n';
    std::cout<<'\n';

    // ---------------------- read transform matrix truth ----------------------
    std::map<std::pair<unsigned, unsigned>, Eigen::Matrix4f> id_to_trans;
    std::ifstream gt_file(dataset_dir / "gt.log");

    while (!gt_file.eof())
    {
    	unsigned id1, id2;
    	gt_file >> id1;
    	gt_file >> id2;
    	gt_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    	decltype(id_to_trans)::mapped_type trans;
    	for (Eigen::Index i = 0; i < trans.rows(); ++i)
    		for (Eigen::Index j = 0; j < trans.cols(); ++j)
    			gt_file >> trans(i, j);

    	id_to_trans[std::pair(id1, id2)] = trans;
    	gt_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    gt_file.close();
    
    // -------------------- Perform point cloud registration on specified dataset -------------------
    std::ofstream log_file;
    if(enable_record)
    {
        log_file.open(record_file_path,std::ios_base::out);
        log_file<<"source cloud id, target cloud id, registration time, rotation error, translation error\n";
        log_file<<", , ms, rad, mm\n";
    }
    auto cloud_src = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto cloud_ref = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto cloud_trans = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    std::default_random_engine rng(1234);
    unsigned correct_num = 0u;
    auto total_duration = std::chrono::milliseconds::zero();
    std::cout << "================================ Start Registration ================================\n";
    for (const auto& value : id_to_trans)
    {
        std::cout << "cloud_" << std::setw(2) << value.first.second << " -> " << "cloud_" << std::setw(2) << value.first.first << ": ";
    	pcl::io::loadPLYFile((dataset_dir / ("data/cloud_bin_" + std::to_string(value.first.first) + ".ply")).string(), *cloud_ref);
    	pcl::io::loadPLYFile((dataset_dir / ("data/cloud_bin_" + std::to_string(value.first.second) + ".ply")).string(), *cloud_src);

    	// generate transform matrix randomly (only rotation)
    	Eigen::Vector4f centroid;
    	pcl::compute3DCentroid(*cloud_ref, centroid);
    	pcl::demeanPointCloud(*cloud_ref, centroid, *cloud_ref);

    	pcl::transformPointCloud(*cloud_src, *cloud_src, value.second);
    	pcl::demeanPointCloud(*cloud_src, centroid, *cloud_src);
        auto rotation = mvfr::createSO3Transformation(static_cast<float>(rot_angle),rng,true);
    	auto [random_angle, _] = mvfr::calcuIsometry3DError(Eigen::Matrix4f(Eigen::Matrix4f::Identity()), rotation);

    	pcl::transformPointCloud(*cloud_src, *cloud_src, rotation);

    	// Register the current source and target point clouds, and record registration time and accuracy if needed
        auto cur_duration = std::chrono::milliseconds::zero();
        double R_err = 0.0, t_err = 0.0;
        if(enable_gpu)
        {
            mvfr::IterativeClosestPointCuda<pcl::PointXYZ,pcl::PointXYZ,float> icp_cuda;
            icp_cuda.setInputSource(cloud_src);
            icp_cuda.setInputTarget(cloud_ref);
            icp_cuda.setMaxCorrespondenceDistance(max_cor_dis);
            icp_cuda.setEuclideanFitnessEpsilon(0.006);
            icp_cuda.setMaximumIterations(50);
            const auto begin = std::chrono::system_clock::now();
            icp_cuda.align(*cloud_trans);
            const auto end = std::chrono::system_clock::now();
            cur_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
            std::tie(R_err,t_err) = icp_cuda.getFitnessScore(rotation.inverse());
        }
        else
        {
            pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ,float> icp;
            icp.setInputSource(cloud_src);
            icp.setInputTarget(cloud_ref);
            icp.setMaxCorrespondenceDistance(max_cor_dis);
            icp.setEuclideanFitnessEpsilon(0.006);
            icp.setMaximumIterations(50);
            const auto begin = std::chrono::system_clock::now();
            icp.align(*cloud_trans);
            const auto end = std::chrono::system_clock::now();
            cur_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
            std::tie(R_err,t_err) = mvfr::calcuIsometry3DError(Eigen::Matrix4f(rotation.inverse()), icp.getFinalTransformation());
        }

        std::cout<< "Registration time: " << std::setw(10) << cur_duration << ", "
            << "Rotation error: " << std::setw(10) << pcl::rad2deg(R_err) << "deg, "
            <<"Translation error: " << std::setw(10) << t_err << "mm\n\n";

    	if (pcl::rad2deg(R_err) < tol_err)
    		correct_num += 1;
    	total_duration += cur_duration;

        if(enable_record)
            log_file<<value.first.second<<", "<<value.first.first<<", "
                <<cur_duration.count()<<", "<<R_err<<","<<t_err<<"\n";

        if(enable_visual)
    	{
    		pcl::visualization::PCLVisualizer view;
    		view.addPointCloud<pcl::PointXYZ>(cloud_ref, "cloud_ref");
    		view.addPointCloud<pcl::PointXYZ>(cloud_src, "cloud_src");
    		view.addPointCloud<pcl::PointXYZ>(cloud_trans, "cloud_trans");

    		view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud_ref");
    		view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "cloud_src");
    		view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "cloud_trans");

    		view.addCoordinateSystem();
    		view.spin();
    	}
    }

    std::cout << "====================================================================================\n";
    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::seconds>(total_duration)
    	<< ", total num: " << id_to_trans.size() << ", correct num: " << correct_num
    	<< ", correctness: " << (double)correct_num / id_to_trans.size() * 100.0 << "%\n";

    log_file.close();

    return EXIT_SUCCESS;
}
