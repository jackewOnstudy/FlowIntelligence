#include "video_matcher.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace VideoMatcher {

// 获取当前时间戳的辅助函数
static std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

VideoMatcherEngine::VideoMatcherEngine(const Parameters& params) 
    : parameters_(params), cache_(std::make_unique<MemoryCache>()) {
    // 构造函数，对应Python的参数初始化 - 新增内存优化
    
    // 初始化时间对齐引擎
    TimeAlignmentParameters time_params;
    time_params.enable_time_alignment = params.enable_time_alignment;
    time_params.max_time_offset = params.max_time_offset;
    time_params.region_grid_size = params.time_alignment_region_size;
    time_params.similarity_threshold = params.time_alignment_similarity_threshold;
    time_params.min_reliable_regions = params.time_alignment_min_regions;
    time_alignment_engine_ = std::make_unique<TimeAlignmentEngine>(time_params);
    
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
    
    // 初始化CSV日志文件头部
    if (!parameters_.csv_log_file_path.empty()) {
        std::ifstream test_file(parameters_.csv_log_file_path);
        if (!test_file.good()) {
            std::ofstream csv_file(parameters_.csv_log_file_path);
            if (csv_file.is_open()) {
                csv_file << "Timestamp,Video1,Video2,Operation,Level,Step,GridSize1,GridSize2,Duration_ms,Details" << std::endl;
                csv_file.close();
            }
        }
        test_file.close();
    }
    
    // 构建视频路径
    std::string video_path1 = parameters_.dataset_path + "/" + parameters_.video_name1;
    std::string video_path2 = parameters_.dataset_path + "/" + parameters_.video_name2;
    
    // 自动获取视频尺寸，而不是使用parameters传递
    cv::Size video_size1, video_size2;
    {
        cv::VideoCapture cap1(video_path1);
        cv::VideoCapture cap2(video_path2);
        
        if (!cap1.isOpened()) {
            throw std::runtime_error("Error: Could not open video: " + video_path1);
        }
        if (!cap2.isOpened()) {
            throw std::runtime_error("Error: Could not open video: " + video_path2);
        }
        
        video_size1 = cv::Size(
            static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        video_size2 = cv::Size(
            static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        
        cap1.release();
        cap2.release();
        
        std::cout << "自动获取视频尺寸 - Video1: " << video_size1.width << "x" << video_size1.height 
                  << ", Video2: " << video_size2.width << "x" << video_size2.height << std::endl;
    }
    
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
        std::cout << "使用固定T2阈值进行运动状态判断..." << std::endl;
        motion_status_per_grid1 = VideoMatcherUtils::getMotionStatus(motion_count_per_grid1, parameters_.motion_threshold1[0]);
        motion_status_per_grid2 = VideoMatcherUtils::getMotionStatus(motion_count_per_grid2, parameters_.motion_threshold2[0]);
    }
    
    VideoMatcherUtils::saveNpyFiles(video_path1, motion_status_per_grid1, grid_size, parameters_.motion_status_path);
    VideoMatcherUtils::saveNpyFiles(video_path2, motion_status_per_grid2, grid_size2, parameters_.motion_status_path);
    
    // ======================================== 时间对齐检测与补偿 ========================================
    if (parameters_.enable_time_alignment && time_alignment_engine_) {
        std::cout << "开始时间对齐检测..." << std::endl;
        
        auto alignment_result = time_alignment_engine_->detectTimeOffset(motion_status_per_grid1, motion_status_per_grid2);
        
        if (alignment_result.is_valid && std::abs(alignment_result.detected_offset) > 0) {
            std::cout << "检测到时间偏移 " << alignment_result.detected_offset << " 帧，应用补偿..." << std::endl;
            
            // 应用时间偏移补偿到第二个视频的运动状态
            motion_status_per_grid2 = TimeAlignmentEngine::applyTimeOffset(motion_status_per_grid2, alignment_result.detected_offset);
            
            // 保存调整后的运动状态
            VideoMatcherUtils::saveNpyFiles(video_path2, motion_status_per_grid2, grid_size2, 
                                          parameters_.motion_status_path + "_aligned");
            
            std::cout << "时间对齐补偿完成，置信度: " << alignment_result.confidence << std::endl;
        } else {
            std::cout << "未检测到明显的时间偏移或检测失败" << std::endl;
        }
    }
    
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
        
        // 保存到对齐后的路径（如果启用了时间对齐）
        std::string save_path2 = parameters_.enable_time_alignment ? 
                                 parameters_.motion_status_path + "_aligned" : 
                                 parameters_.motion_status_path;
        VideoMatcherUtils::saveNpyFiles(video_path2, motion_status_per_grid2, temp_grid_size2, save_path2);
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
        
        // 使用对齐后的路径（如果启用了时间对齐）
        std::string load_path2 = parameters_.enable_time_alignment ? 
                                parameters_.motion_status_path + "_aligned" : 
                                parameters_.motion_status_path;
        motion_status_per_grid2 = VideoMatcherUtils::loadNpyFiles(video_path2, temp_grid_size2, load_path2);
        
        // 将长序列分为多个阶段进行匹配
        auto motion_status_per_grid1_segments = SegmentMatcher::segmentMatrix(
            motion_status_per_grid1, motion_status_per_grid1.cols);
        auto motion_status_per_grid2_segments = SegmentMatcher::segmentMatrix(
            motion_status_per_grid2, motion_status_per_grid2.cols);

        std::cout << "###############[DEBUG]###############" << "segmentMatrix Finished!" <<std::endl;
        
        std::vector<std::vector<MatchTriplet>> match_result_all; // 保存一个网格大小下所有匹配结果
        
        for (size_t idx = 0; idx < motion_status_per_grid1_segments.size(); ++idx) {
            if (idx == 0) {
                match_result = SegmentMatcher::findMatchingGridWithSegment(
                    motion_status_per_grid1_segments[idx], motion_status_per_grid2_segments[idx],
                    parameters_, sorted_large_grid_corre_small_dict, num_cols1, num_cols1 / 2, shifting_flag);
                std::cout << "###############[DEBUG]###############" << "findMatchingGridWithSegment Finished!" <<std::endl;
                // 最大网格匹配，只保留前N个匹配结果
                if (iterate_time == 0) {
                    size_t max_results = std::min(static_cast<size_t>(20), match_result.size());
                    match_result.resize(max_results);
                }
                std::cout << "###############[DEBUG]###############" << "filter Finished!" <<std::endl;
               
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
                
                if (!parameters_.csv_log_file_path.empty()) {
                    std::ofstream csv_file(parameters_.csv_log_file_path, std::ios::app);
                    if (csv_file.is_open()) {
                        csv_file << getCurrentTimestamp() << ","
                                 << parameters_.video_name1 << "," << parameters_.video_name2 << ","
                                 << "propagate," << iterate_time << "," << i << ","
                                 << temp_grid_size.width << "x" << temp_grid_size.height << ","
                                 << temp_grid_size2.width << "x" << temp_grid_size2.height << ","
                                 << duration.count() << "," << std::endl;
                        csv_file.close();
                    }
                }
                
                match_result = VideoMatcherUtils::processTriplets(match_result);
                match_result_all.push_back(match_result);
                std::cout << "###############[DEBUG]###############" << i << "th propagateMatchingResult Finished!" << std::endl;
            }
        }
        
        // 删除重复的匹配结果
        match_result = VideoMatcherUtils::processTriplets(match_result);
        
        // 使用RANSAC算法筛选匹配结果
        match_result = VideoMatcherUtils::ransacFilterMatchResults(match_result, 
                                                                  temp_grid_size, temp_grid_size2,
                                                                  num_cols1, num_rows1, num_cols2, num_rows2,
                                                                  shifting_flag,
                                                                  parameters_.ransac_threshold,
                                                                  parameters_.ransac_max_iterations,
                                                                  parameters_.ransac_min_inlier_ratio);
        
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
        
        // 使用新的拼接函数替代分别保存两个图片
        VideoMatcherUtils::combinedMatchResultView(video_path1, video_path2, match_result, 
                                                  temp_grid_size, temp_grid_size2, stride1, stride2,
                                                  parameters_.match_result_view_path);
        
        // 保存每次迭代的匹配结果 
        VideoMatcherUtils::saveMatchResultList(video_path1, match_result, temp_grid_size, 
                                              parameters_.match_result_path);
        
        // 保存匹配结果到返回列表
        all_match_results.push_back(match_result); // 保存当前迭代的match_result
        
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
        
        std::cout << "Calculate small grid index..." << std::endl;
        sorted_large_grid_corre_small_dict = VideoMatcherUtils::getSmallGridIndex(
            match_result, video_size2, small_grid_size2, temp_grid_size2, shifting_flag, 2);
        
        temp_grid_size = small_grid_size;
        temp_grid_size2 = small_grid_size2;
    }
    
    std::cout << "分层匹配完成!" << std::endl;
    return all_match_results;
}

} // namespace VideoMatcher
