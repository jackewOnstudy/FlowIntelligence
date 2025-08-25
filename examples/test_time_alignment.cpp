#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/time_alignment.h"

using namespace VideoMatcher;

// 创建模拟的运动状态矩阵，用于测试
cv::Mat createSimulatedMotionStatus(int num_grids, int seq_length, int offset = 0) {
    cv::Mat motion_status = cv::Mat::zeros(num_grids, seq_length, CV_8U);
    
    // 创建一些运动模式
    for (int t = 0; t < seq_length; ++t) {
        int adjusted_t = t + offset;
        if (adjusted_t >= 0 && adjusted_t < seq_length) {
            // 周期性运动模式
            if ((adjusted_t / 10) % 3 == 0) {
                for (int g = 0; g < num_grids / 2; g += 2) {
                    motion_status.at<uchar>(g, t) = 1;
                }
            }
            
            // 脉冲运动模式
            if (adjusted_t % 50 == 0) {
                for (int g = num_grids / 2; g < num_grids; g += 3) {
                    motion_status.at<uchar>(g, t) = 1;
                }
            }
        }
    }
    
    return motion_status;
}

int main() {
    std::cout << "时间对齐功能测试\n" << std::endl;
    
    // 创建测试数据
    int num_grids = 256;  // 8x8网格
    int seq_length = 200;
    int true_offset = 15;  // 真实的时间偏移
    
    cv::Mat motion_status1 = createSimulatedMotionStatus(num_grids, seq_length, 0);
    cv::Mat motion_status2 = createSimulatedMotionStatus(num_grids, seq_length, true_offset);
    
    std::cout << "创建测试数据：" 
              << num_grids << "个网格，" 
              << seq_length << "帧序列，"
              << "真实偏移=" << true_offset << "帧\n" << std::endl;
    
    // 配置时间对齐参数
    TimeAlignmentParameters params;
    params.enable_time_alignment = true;
    params.max_time_offset = 30;
    params.region_grid_size = 4;
    params.similarity_threshold = 0.1f;
    params.min_reliable_regions = 2;
    params.use_coarse_detection = true;
    params.coarse_downsample_factor = 4;
    
    // 创建时间对齐引擎
    TimeAlignmentEngine engine(params);
    
    // 执行时间对齐检测
    auto result = engine.detectTimeOffset(motion_status1, motion_status2);
    
    // 输出结果
    std::cout << "检测结果：\n";
    std::cout << "  是否有效: " << (result.is_valid ? "是" : "否") << "\n";
    
    if (result.is_valid) {
        std::cout << "  检测偏移: " << result.detected_offset << " 帧\n";
        std::cout << "  真实偏移: " << true_offset << " 帧\n";
        std::cout << "  检测误差: " << std::abs(result.detected_offset - true_offset) << " 帧\n";
        std::cout << "  置信度: " << result.confidence << "\n";
        std::cout << "  可靠区域数: " << result.num_reliable_regions << "\n";
        std::cout << "  检测精度: " << (std::abs(result.detected_offset - true_offset) <= 2 ? "高精度" : "低精度") << "\n";
        
        // 应用时间偏移补偿
        cv::Mat aligned_status2 = TimeAlignmentEngine::applyTimeOffset(motion_status2, result.detected_offset);
        
        // 验证对齐效果（简单的相似性检查）
        int matching_frames = 0;
        for (int t = 0; t < seq_length; ++t) {
            bool frame_match = true;
            for (int g = 0; g < num_grids; ++g) {
                if (motion_status1.at<uchar>(g, t) != aligned_status2.at<uchar>(g, t)) {
                    frame_match = false;
                    break;
                }
            }
            if (frame_match) matching_frames++;
        }
        
        float alignment_accuracy = static_cast<float>(matching_frames) / seq_length;
        std::cout << "  对齐效果: " << alignment_accuracy * 100 << "% 帧完全匹配\n";
    } else {
        std::cout << "  检测失败\n";
    }
    
    std::cout << "\n测试完成!" << std::endl;
    return 0;
} 