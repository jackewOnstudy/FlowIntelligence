#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace VideoMatcher {

struct Parameters {
    // 视频路径和配置
    std::string video_name1;
    std::string video_name2;
    cv::Size video_size1;
    cv::Size video_size2;
    
    // 运动检测参数
    std::vector<int> motion_threshold1;  // 各层级运动阈值
    std::vector<int> motion_threshold2;
    cv::Size GaussianBlurKernel;
    int Binary_threshold;
    int max_frames;  // 最大处理帧数（可根据网格大小自动调整）
    bool use_otsu_t1;  // 是否在帧差分时使用Otsu阈值
    bool use_otsu_t2;  // 是否在运动状态判断时使用Otsu阈值
    bool is_global_otsu;  // 是否使用全局Otsu（true）还是网格内Otsu（false）
    float otsu_min_threshold;  // Otsu阈值的最小值
    
    // 网格参数
    cv::Size grid_size;   // 初始网格大小
    cv::Size grid_size2;
    cv::Size stride;      // 步长
    cv::Size stride2;
    
    // 匹配参数
    int segment_length;
    int max_mismatches;
    std::string distance_metric;
    int select_grid_factor;
    int mismatch_distance_factor;
    int stage_length;
    int propagate_step;
    
    // 路径配置
    std::string dataset_path;
    std::string base_output_folder;
    std::string motion_status_path;
    std::string motion_counts_path;
    std::string match_result_path;
    std::string match_result_view_path;
    std::string csv_log_file_path;  // CSV日志文件路径
    
    // 时间对齐参数
    bool enable_time_alignment;         // 是否启用时间对齐
    int max_time_offset;               // 最大搜索偏移范围（帧数）
    int time_alignment_region_size;    // 区域划分网格大小
    float time_alignment_similarity_threshold;  // 相似性阈值
    int time_alignment_min_regions;    // 最少可靠区域对数
    
    // RANSAC筛选参数
    float ransac_threshold;            // RANSAC像素误差阈值
    int ransac_max_iterations;         // RANSAC最大迭代次数
    float ransac_min_inlier_ratio;     // RANSAC最小内点比例
    
    // 默认构造函数
    Parameters() {
        // 初始化默认参数 - 对应Python中的param_T0_B201
        video_name1 = "T10L.mp4";
        video_name2 = "T10R.mp4";
        // video_size1 = cv::Size(1920, 1080);
        // video_size2 = cv::Size(1920, 1080);
        // video_size1 = cv::Size(960, 540);
        // video_size2 = cv::Size(960, 540);
        
        motion_threshold1 = {24, 80, 160, 200};
        motion_threshold2 = {24, 80, 160, 200};
        
        GaussianBlurKernel = cv::Size(11, 11);
        Binary_threshold = 6;
        max_frames = 3000;  // 减少处理帧数以避免内存问题
        use_otsu_t1 = false;  // 默认不使用T1的Otsu阈值
        use_otsu_t2 = false;  // 默认不使用T2的Otsu阈值
        is_global_otsu = false;  // 默认使用网格内Otsu
        otsu_min_threshold = 1.0f;  // Otsu最小阈值
        
        grid_size = cv::Size(8, 8);
        grid_size2 = cv::Size(8, 8);
        stride = cv::Size(8, 8);
        stride2 = cv::Size(8, 8);
        
        segment_length = 800;
        max_mismatches = 1;
        distance_metric = "logic_and";
        select_grid_factor = 20;
        mismatch_distance_factor = 9;
        stage_length = 9000;
        propagate_step = 1;
        
        dataset_path = "/home/jackew/Project/FlowIntelligence/Datasets";
        base_output_folder = "./Output";
        motion_status_path = base_output_folder + "/MotionStatus";
        motion_counts_path = base_output_folder + "/MotionCounts";
        match_result_path = base_output_folder + "/MatchResult/List";
        match_result_view_path = base_output_folder + "/MatchResult/Pictures";
        
        // 时间对齐参数默认值
        enable_time_alignment = false;    // 默认关闭时间对齐
        max_time_offset = 30;            // 最大搜索30帧偏移
        time_alignment_region_size = 4;  // 4x4=16个区域
        time_alignment_similarity_threshold = 0.6f;  // 相似性阈值
        time_alignment_min_regions = 3;  // 至少3个可靠区域
        
        // RANSAC参数默认值
        ransac_threshold = 5.0f;         // 像素误差阈值
        ransac_max_iterations = 1000;    // 最大迭代次数
        ransac_min_inlier_ratio = 0.5f;  // 最小内点比例
    }
};

} // namespace VideoMatcher 