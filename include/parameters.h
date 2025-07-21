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
    
    // 默认构造函数
    Parameters() {
        // 初始化默认参数 - 对应Python中的param_T0_B201
        video_name1 = "T10L.mp4";
        video_name2 = "T10R.mp4";
        video_size1 = cv::Size(1920, 1080);
        video_size2 = cv::Size(1920, 1080);
        
        motion_threshold1 = {24, 80, 160, 400};
        motion_threshold2 = {24, 80, 160, 400};
        
        GaussianBlurKernel = cv::Size(11, 11);
        Binary_threshold = 6;
        max_frames = 3000;  // 减少处理帧数以避免内存问题
        
        grid_size = cv::Size(8, 8);
        grid_size2 = cv::Size(8, 8);
        stride = cv::Size(8, 8);
        stride2 = cv::Size(8, 8);
        
        segment_length = 100;
        max_mismatches = 1;
        distance_metric = "logic_and";
        select_grid_factor = 10;
        mismatch_distance_factor = 4;
        stage_length = 9000;
        propagate_step = 1;
        
        dataset_path = "/home/jackew/Project/FlowIntelligence/";
        base_output_folder = "./Output";
        motion_status_path = base_output_folder + "/MotionStatus";
        motion_counts_path = base_output_folder + "/MotionCounts";
        match_result_path = base_output_folder + "/MatchResult/List";
        match_result_view_path = base_output_folder + "/MatchResult/Pictures";
    }
};

} // namespace VideoMatcher 