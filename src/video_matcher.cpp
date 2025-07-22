#include "video_matcher.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

namespace VideoMatcher {

VideoMatcherEngine::VideoMatcherEngine(const Parameters& params) 
    : parameters_(params), cache_(std::make_unique<MemoryCache>()) {
    // 构造函数，对应Python的参数初始化 - 新增内存优化
    
    // 预分配缓冲区以减少运行时内存分配
    preallocateBuffers(1000);  // 预分配合理大小的缓冲区
}

void VideoMatcherEngine::preallocateBuffers(size_t expected_size) {
    // 预分配内存缓冲区 - 新增优化
    motion_matrix_buffer_.reserve(ITERATE_TIMES * 2);  // 为每个迭代级别预分配
    result_buffer_.reserve(ITERATE_TIMES);
    
    for (int i = 0; i < ITERATE_TIMES; ++i) {
        result_buffer_.emplace_back();
        result_buffer_[i].reserve(expected_size);
    }
}

std::vector<std::vector<MatchTriplet>> VideoMatcherEngine::calOverlapGrid() {
    // 对应Python的cal_overlap_grid函数
    std::cout << "开始计算overlap grid..." << std::endl;
    
    // 构建视频路径
    std::string video_path1 = parameters_.dataset_path + "/" + parameters_.video_name1;
    std::string video_path2 = parameters_.dataset_path + "/" + parameters_.video_name2;
    
    cv::Size grid_size = parameters_.grid_size;
    cv::Size grid_size2 = parameters_.grid_size2;
    cv::Size i_stride1 = parameters_.stride;
    cv::Size i_stride2 = parameters_.stride2;
    
    // 判断是否使用滑动网格
    bool shifting_flag = (grid_size.width != i_stride1.width);
    
    int num_cols1, num_rows1, num_cols2, num_rows2;
    
    // ================================== 计算8x8基网格的count序列以及status序列 ==================================
    std::cout << "计算基础网格运动状态序列..." << std::endl;
    
    // 计算网格状态序列(值为运动像素个数)
    cv::Mat motion_count_per_grid1, motion_count_per_grid2;
    
    if (parameters_.use_otsu_t1) {
        // 使用Otsu自适应T1参数
        std::cout << "使用Otsu自适应T1阈值进行运动检测..." << std::endl;
        motion_count_per_grid1 = VideoMatcherUtils::getMotionCountWithOtsu(
            video_path1, i_stride1, grid_size, parameters_, num_cols1, num_rows1);
        motion_count_per_grid2 = VideoMatcherUtils::getMotionCountWithOtsu(
            video_path2, i_stride2, grid_size2, parameters_, num_cols2, num_rows2);
    } else {
        // 使用固定T1阈值
        motion_count_per_grid1 = VideoMatcherUtils::getMotionCountWithShiftingGrid(
            video_path1, i_stride1, grid_size, parameters_, num_cols1, num_rows1);
        motion_count_per_grid2 = VideoMatcherUtils::getMotionCountWithShiftingGrid(
            video_path2, i_stride2, grid_size2, parameters_, num_cols2, num_rows2);
    }
    
    // 保存motion_count_per_grid
    VideoMatcherUtils::saveNpyFiles(video_path1, motion_count_per_grid1, grid_size, parameters_.motion_counts_path);
    VideoMatcherUtils::saveNpyFiles(video_path2, motion_count_per_grid2, grid_size2, parameters_.motion_counts_path);
    
    // 计算网格状态序列
    cv::Mat motion_status_per_grid1, motion_status_per_grid2;
    
    if (parameters_.use_otsu_t2) {
        // 使用Otsu自适应T2参数
        std::cout << "使用Otsu自适应T2阈值进行运动状态判断..." << std::endl;
        if (parameters_.is_global_otsu) {
            std::cout << "采用全局Otsu阈值模式..." << std::endl;
            motion_status_per_grid1 = VideoMatcherUtils::getMotionStatusGlobalOtsu(motion_count_per_grid1, parameters_.otsu_min_threshold);
            motion_status_per_grid2 = VideoMatcherUtils::getMotionStatusGlobalOtsu(motion_count_per_grid2, parameters_.otsu_min_threshold);
        } else {
            std::cout << "采用网格内Otsu阈值模式..." << std::endl;
            motion_status_per_grid1 = VideoMatcherUtils::getMotionStatusWithOtsu(motion_count_per_grid1, parameters_.otsu_min_threshold);
            motion_status_per_grid2 = VideoMatcherUtils::getMotionStatusWithOtsu(motion_count_per_grid2, parameters_.otsu_min_threshold);
        }
    } else {
        // 使用固定T2阈值
        motion_status_per_grid1 = VideoMatcherUtils::getMotionStatus(motion_count_per_grid1, parameters_.motion_threshold1[0]);
        motion_status_per_grid2 = VideoMatcherUtils::getMotionStatus(motion_count_per_grid2, parameters_.motion_threshold2[0]);
    }
    
    VideoMatcherUtils::saveNpyFiles(video_path1, motion_status_per_grid1, grid_size, parameters_.motion_status_path);
    VideoMatcherUtils::saveNpyFiles(video_path2, motion_status_per_grid2, grid_size2, parameters_.motion_status_path);
    
    // ======================================== 构造4N倍大网格 ========================================
    std::cout << "构造分层网格结构..." << std::endl;
    
    // 初始网格大小都是8x8，以此构造4xN倍网格状态序列
    cv::Size temp_grid_size = grid_size;
    cv::Size temp_grid_size2 = grid_size2;
    
    for (int i = 1; i < ITERATE_TIMES; ++i) {
        temp_grid_size.width *= 2;
        temp_grid_size.height *= 2;
        temp_grid_size2.width *= 2;
        temp_grid_size2.height *= 2;
        
        std::cout << "Constructing " << i << "th iteration: Grid size: " 
                  << temp_grid_size.width << "x" << temp_grid_size.height << "::"
                  << temp_grid_size2.width << "x" << temp_grid_size2.height << std::endl;
        
        motion_count_per_grid1 = VideoMatcherUtils::get4nGridMotionCount(
            motion_count_per_grid1, num_cols1, num_rows1, shifting_flag);
        motion_count_per_grid2 = VideoMatcherUtils::get4nGridMotionCount(
            motion_count_per_grid2, num_cols2, num_rows2, shifting_flag);
        
        if (parameters_.use_otsu_t2) {
            // 使用Otsu自适应T2参数
            if (parameters_.is_global_otsu) {
                motion_status_per_grid1 = VideoMatcherUtils::getMotionStatusGlobalOtsu(motion_count_per_grid1, parameters_.otsu_min_threshold);
                motion_status_per_grid2 = VideoMatcherUtils::getMotionStatusGlobalOtsu(motion_count_per_grid2, parameters_.otsu_min_threshold);
            } else {
                motion_status_per_grid1 = VideoMatcherUtils::getMotionStatusWithOtsu(motion_count_per_grid1, parameters_.otsu_min_threshold);
                motion_status_per_grid2 = VideoMatcherUtils::getMotionStatusWithOtsu(motion_count_per_grid2, parameters_.otsu_min_threshold);
            }
        } else {
            // 使用固定T2阈值
            motion_status_per_grid1 = VideoMatcherUtils::getMotionStatus(
                motion_count_per_grid1, parameters_.motion_threshold1[i]);
            motion_status_per_grid2 = VideoMatcherUtils::getMotionStatus(
                motion_count_per_grid2, parameters_.motion_threshold2[i]);
        }
        
        VideoMatcherUtils::saveNpyFiles(video_path1, motion_status_per_grid1, temp_grid_size, parameters_.motion_status_path);
        VideoMatcherUtils::saveNpyFiles(video_path2, motion_status_per_grid2, temp_grid_size2, parameters_.motion_status_path);
    }
    
    // ======================================== 分层匹配 ========================================
    std::cout << "开始分层匹配..." << std::endl;
    
    std::vector<std::vector<MatchTriplet>> all_match_results;
    std::vector<MatchTriplet> match_result;
    std::map<int, std::set<int>> sorted_large_grid_corre_small_dict;
    
    for (int iterate_time = 0; iterate_time < ITERATE_TIMES; ++iterate_time) {
        std::cout << "Matching " << iterate_time << "th iteration: Grid size: " 
                  << temp_grid_size.width << "x" << temp_grid_size.height << "::"
                  << temp_grid_size2.width << "x" << temp_grid_size2.height << std::endl;
        
        motion_status_per_grid1 = VideoMatcherUtils::loadNpyFiles(video_path1, temp_grid_size, parameters_.motion_status_path);
        motion_status_per_grid2 = VideoMatcherUtils::loadNpyFiles(video_path2, temp_grid_size2, parameters_.motion_status_path);
        
        // 将长序列分为多个阶段进行匹配
        auto motion_status_per_grid1_segments = SegmentMatcher::segmentMatrix(
            motion_status_per_grid1, motion_status_per_grid1.cols);
        auto motion_status_per_grid2_segments = SegmentMatcher::segmentMatrix(
            motion_status_per_grid2, motion_status_per_grid2.cols);
        
        std::vector<std::vector<MatchTriplet>> match_result_all; // 保存一个网格大小下所有匹配结果
        
        for (size_t idx = 0; idx < motion_status_per_grid1_segments.size(); ++idx) {
            if (idx == 0) {
                match_result = SegmentMatcher::findMatchingGridWithSegment(
                    motion_status_per_grid1_segments[idx], motion_status_per_grid2_segments[idx],
                    parameters_, sorted_large_grid_corre_small_dict, num_cols1, num_cols1 / 2, shifting_flag);
                
                // 最大网格匹配，只保留前N个匹配结果
                if (iterate_time == 0) {
                    size_t max_results = std::min(static_cast<size_t>(20), match_result.size());
                    match_result.resize(max_results);
                }
                
                match_result = VideoMatcherUtils::processTriplets(match_result);
                match_result_all.push_back(match_result);
                
                if (motion_status_per_grid1_segments.size() != 1) {
                    std::cout << "Not Only one stage, continue" << std::endl;
                    continue;
                }
            }
            
            // 传播匹配结果
            for (int i = 0; i < 6; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();
                
                match_result = SegmentMatcher::propagateMatchingResult(
                    match_result, motion_status_per_grid1_segments[idx], motion_status_per_grid2_segments[idx],
                    num_rows1, num_cols1, num_rows2, num_cols2, parameters_, shifting_flag);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                std::ofstream log_file("./Output/citydata.txt", std::ios::app);
                if (log_file.is_open()) {
                    log_file << "Matching " << iterate_time << "th iteration: Grid size: " 
                             << temp_grid_size.width << "x" << temp_grid_size.height << "::"
                             << temp_grid_size2.width << "x" << temp_grid_size2.height << std::endl;
                    log_file << "    Propagate " << i << " time: " << duration.count() << " ms" << std::endl;
                    log_file.close();
                }
                
                match_result = VideoMatcherUtils::processTriplets(match_result);
                match_result_all.push_back(match_result);
            }
        }
        
        // 删除重复的匹配结果
        match_result = VideoMatcherUtils::processTriplets(match_result);
        
        // 打印前16个匹配结果
        std::cout << "前16个匹配结果:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(16), match_result.size()); ++i) {
            std::cout << "(" << match_result[i].grid1 << ", " << match_result[i].grid2 
                      << ", " << match_result[i].distance << ")" << std::endl;
        }
        
        // 可视化结果
        cv::Size stride1 = temp_grid_size;
        cv::Size stride2 = temp_grid_size2;
        if (shifting_flag) {
            stride1.width /= 2;
            stride1.height /= 2;
            stride2.width /= 2;
            stride2.height /= 2;
        }
        
        VideoMatcherUtils::matchResultView(video_path1, match_result, temp_grid_size, stride1, 
                                          parameters_.match_result_view_path, 1);
        VideoMatcherUtils::matchResultView(video_path2, match_result, temp_grid_size2, stride2, 
                                          parameters_.match_result_view_path, 2);
        
        // 保存匹配结果到列表
        // 这里可以添加保存match_result_all的逻辑
        all_match_results.push_back(match_result_all[0]); // 简化版本，只保存第一个结果
        
        // 为下一次迭代准备
        cv::Size small_grid_size(temp_grid_size.width / 2, temp_grid_size.height / 2);
        cv::Size small_grid_size2(temp_grid_size2.width / 2, temp_grid_size2.height / 2);
        
        num_cols1 *= 2;
        num_rows1 *= 2;
        num_cols2 *= 2;
        num_rows2 *= 2;
        
        if (shifting_flag) {
            num_cols1 += 1;
            num_rows1 += 1;
            num_cols2 += 1;
            num_rows2 += 1;
        }
        
        sorted_large_grid_corre_small_dict = VideoMatcherUtils::getSmallGridIndex(
            match_result, parameters_.video_size2, small_grid_size2, temp_grid_size2, shifting_flag, 2);
        
        temp_grid_size = small_grid_size;
        temp_grid_size2 = small_grid_size2;
    }
    
    std::cout << "分层匹配完成!" << std::endl;
    return all_match_results;
}

} // namespace VideoMatcher
