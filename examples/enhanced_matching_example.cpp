#include "enhanced_video_matcher.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace VideoMatcher;

// 演示基本使用
void demonstrateBasicUsage() {
    std::cout << "\n=== 基本使用示例 ===" << std::endl;
    
    try {
        // 创建增强参数配置
        EnhancedParameters params;
        params.video_name1 = "../Datasets/T10L.mp4";
        params.video_name2 = "../Datasets/T10R.mp4";
        params.dataset_path = "../Datasets";
        params.base_output_folder = "./TEST_OUTPUT";
        
        // 启用所有增强功能
        params.feature_config.enable_motion_features = true;
        params.feature_config.enable_texture_features = true;
        params.feature_config.enable_temporal_features = true;
        params.feature_config.enable_geometric_features = true;
        params.feature_config.use_gpu_acceleration = true;
        
        params.quality_config.enable_motion_analysis = true;
        params.quality_config.enable_temporal_analysis = true;
        params.quality_config.enable_spatial_analysis = true;
        
        params.bidirectional_config.enable_parallel_matching = true;
        params.bidirectional_config.enable_quality_assessment = true;
        params.bidirectional_config.enable_conflict_resolution = true;
        
        params.temporal_config.enable_temporal_smoothing = true;
        params.temporal_config.enable_trajectory_prediction = true;
        params.temporal_config.enable_missing_interpolation = true;
        
        // 创建增强的视频匹配器
        auto matcher = std::make_unique<EnhancedVideoMatcherEngine>(params);
        
        // 设置进度回调
        matcher->setProgressCallback([](const ProcessingMonitor::StageInfo& stage) {
            std::cout << "进度: " << stage.description 
                      << " - " << std::fixed << std::setprecision(1) 
                      << stage.progress_percentage << "%" << std::endl;
        });
        
        // 执行匹配
        std::cout << "开始视频匹配处理..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto results = matcher->processVideoMatching();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 输出结果统计
        std::cout << "\n匹配完成!" << std::endl;
        std::cout << "总处理时间: " << duration.count() << " 毫秒" << std::endl;
        std::cout << "处理层级数: " << results.size() << std::endl;
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            std::cout << "\n层级 " << i << " 结果:" << std::endl;
            std::cout << "  网格大小: " << result.grid_size.width << "x" << result.grid_size.height << std::endl;
            std::cout << "  总匹配数: " << result.matches.size() << std::endl;
            std::cout << "  可靠匹配数: " << result.reliable_matches << std::endl;
            std::cout << "  平均质量: " << std::fixed << std::setprecision(3) << result.average_quality << std::endl;
            std::cout << "  时间一致性: " << result.temporal_consistency_score << std::endl;
            std::cout << "  轨迹数量: " << result.trajectories.size() << std::endl;
            std::cout << "  覆盖率: " << result.coverage_ratio << std::endl;
        }
        
        // 获取性能统计
        auto stats = matcher->getProcessingStats();
        std::cout << "\n性能统计:" << std::endl;
        std::cout << "  总找到匹配: " << stats.total_matches_found << std::endl;
        std::cout << "  可靠匹配: " << stats.total_reliable_matches << std::endl;
        std::cout << "  平均匹配质量: " << stats.average_match_quality << std::endl;
        std::cout << "  平均时间一致性: " << stats.average_temporal_consistency << std::endl;
        std::cout << "  内存峰值使用: " << stats.memory_stats.peak_usage / (1024*1024) << " MB" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

// 演示高精度配置
void demonstrateHighPrecisionMatching() {
    std::cout << "\n=== 高精度匹配示例 ===" << std::endl;
    
    try {
        // 使用工厂创建高精度匹配器
        auto matcher = EnhancedVideoMatcherFactory::createHighPrecisionMatcher();
        
        // 设置视频路径
        EnhancedParameters params;
        params.video_name1 = "../Datasets/T10L.mp4";
        params.video_name2 = "../Datasets/T10R.mp4";
        params.dataset_path = "../Datasets";
        params.base_output_folder = "./TEST_OUTPUT";
        
        // 高精度特定配置
        params.quality_config.consistency_threshold = 0.8f;  // 更高的一致性要求
        params.temporal_config.trajectory_confidence_threshold = 0.7f;  // 更高的轨迹置信度
        params.bidirectional_config.consistency_threshold = 0.8f;  // 更严格的双向一致性
        
        matcher->setParameters(params);
        
        // 执行高精度匹配
        auto results = matcher->processVideoMatching();
        
        // 分析精度结果
        for (const auto& result : results) {
            auto high_quality_matches = result.getHighQualityMatches(0.8f);
            auto consistent_trajectories = result.getConsistentTrajectories(0.7f);
            
            std::cout << "高质量匹配 (>0.8): " << high_quality_matches.size() << std::endl;
            std::cout << "一致轨迹 (>0.7): " << consistent_trajectories.size() << std::endl;
            
            // 输出详细质量分析
            if (!result.quality_metrics.empty()) {
                float min_quality = std::numeric_limits<float>::max();
                float max_quality = std::numeric_limits<float>::min();
                float avg_quality = 0.0f;
                
                for (const auto& quality : result.quality_metrics) {
                    min_quality = std::min(min_quality, quality.overall_confidence);
                    max_quality = std::max(max_quality, quality.overall_confidence);
                    avg_quality += quality.overall_confidence;
                }
                avg_quality /= result.quality_metrics.size();
                
                std::cout << "质量分布: 最小=" << min_quality 
                          << ", 最大=" << max_quality 
                          << ", 平均=" << avg_quality << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "高精度匹配错误: " << e.what() << std::endl;
    }
}

// 演示实时处理
void demonstrateRealTimeProcessing() {
    std::cout << "\n=== 实时处理示例 ===" << std::endl;
    
    try {
        // 创建实时匹配器
        auto matcher = EnhancedVideoMatcherFactory::createRealTimeMatcher();
        
        // 设置结果回调
        auto result_callback = [](const EnhancedMatchResult& result) {
            std::cout << "实时结果: 匹配数=" << result.matches.size() 
                      << ", 质量=" << result.average_quality 
                      << ", 处理时间=" << result.processing_time.count() << "ms" << std::endl;
            
            // 检查是否有高质量匹配
            auto high_quality = result.getHighQualityMatches(0.7f);
            if (!high_quality.empty()) {
                std::cout << "  检测到 " << high_quality.size() << " 个高质量匹配!" << std::endl;
            }
        };
        
        // 启动实时处理 (使用摄像头或视频流)
        std::string stream1 = "rtmp://stream1_url";  // 实际应用中替换为真实流地址
        std::string stream2 = "rtmp://stream2_url";
        
        std::cout << "启动实时处理 (模拟5秒)..." << std::endl;
        matcher->startRealTimeProcessing(stream1, stream2, result_callback);
        
        // 模拟运行5秒
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        // 停止实时处理
        matcher->stopRealTimeProcessing();
        std::cout << "实时处理已停止" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "实时处理错误: " << e.what() << std::endl;
    }
}

// 演示批量处理
void demonstrateBatchProcessing() {
    std::cout << "\n=== 批量处理示例 ===" << std::endl;
    
    try {
        // 创建高性能匹配器用于批量处理
        auto matcher = EnhancedVideoMatcherFactory::createHighPerformanceMatcher();
        
        // 准备视频对列表
        std::vector<std::pair<std::string, std::string>> video_pairs = {
            {"batch1_video1.mp4", "batch1_video2.mp4"},
            {"batch2_video1.mp4", "batch2_video2.mp4"},
            {"batch3_video1.mp4", "batch3_video2.mp4"}
        };
        
        std::cout << "开始批量处理 " << video_pairs.size() << " 个视频对..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto batch_results = matcher->processBatchVideos(video_pairs);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "批量处理完成! 总耗时: " << total_time.count() << " 秒" << std::endl;
        
        // 统计批量结果
        size_t total_matches = 0;
        size_t total_reliable = 0;
        float avg_quality = 0.0f;
        
        for (size_t i = 0; i < batch_results.size(); ++i) {
            const auto& results = batch_results[i];
            
            std::cout << "\n视频对 " << (i+1) << " 结果:" << std::endl;
            
            for (size_t j = 0; j < results.size(); ++j) {
                const auto& result = results[j];
                total_matches += result.matches.size();
                total_reliable += result.reliable_matches;
                avg_quality += result.average_quality;
                
                std::cout << "  层级 " << j << ": 匹配=" << result.matches.size() 
                          << ", 可靠=" << result.reliable_matches 
                          << ", 质量=" << result.average_quality << std::endl;
            }
        }
        
        size_t total_levels = 0;
        for (const auto& results : batch_results) {
            total_levels += results.size();
        }
        
        if (total_levels > 0) {
            avg_quality /= total_levels;
        }
        
        std::cout << "\n批量统计:" << std::endl;
        std::cout << "  总匹配数: " << total_matches << std::endl;
        std::cout << "  总可靠匹配: " << total_reliable << std::endl;
        std::cout << "  平均质量: " << avg_quality << std::endl;
        std::cout << "  平均处理时间: " << total_time.count() / video_pairs.size() << " 秒/对" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "批量处理错误: " << e.what() << std::endl;
    }
}

// 演示自适应优化
void demonstrateAdaptiveOptimization() {
    std::cout << "\n=== 自适应优化示例 ===" << std::endl;
    
    try {
        // 创建标准匹配器并启用自适应优化
        auto matcher = EnhancedVideoMatcherFactory::createStandardMatcher();
        matcher->enableAdaptiveOptimization(true);
        
        // 模拟多个视频的处理来触发自适应调整
        std::vector<std::string> test_videos = {
            "T10L.mp4", "T10R.mp4"
        };
        
        std::vector<EnhancedMatchResult> historical_results;
        
        for (size_t i = 0; i < test_videos.size(); ++i) {
            EnhancedParameters params;
            params.video_name1 = test_videos[i];
            params.video_name2 = test_videos[(i+1) % test_videos.size()];
            params.dataset_path = "../Datasets";
            params.base_output_folder = "../OUTPUT";
            
            matcher->setParameters(params);
            
            std::cout << "处理视频对 " << (i+1) << "..." << std::endl;
            auto results = matcher->processVideoMatching();
            
            if (!results.empty()) {
                historical_results.insert(historical_results.end(), results.begin(), results.end());
                
                // 触发自适应调整
                matcher->updateOptimizationStrategy(historical_results);
                
                std::cout << "  匹配质量: " << results[0].average_quality << std::endl;
                std::cout << "  处理时间: " << results[0].processing_time.count() << "ms" << std::endl;
            }
        }
        
        // 输出自适应优化效果
        if (historical_results.size() >= 2) {
            float first_quality = historical_results[0].average_quality;
            float last_quality = historical_results.back().average_quality;
            float improvement = ((last_quality - first_quality) / first_quality) * 100.0f;
            
            std::cout << "\n自适应优化效果:" << std::endl;
            std::cout << "  初始质量: " << first_quality << std::endl;
            std::cout << "  最终质量: " << last_quality << std::endl;
            std::cout << "  质量提升: " << std::fixed << std::setprecision(1) << improvement << "%" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "自适应优化错误: " << e.what() << std::endl;
    }
}

// 演示GPU加速
void demonstrateGPUAcceleration() {
    std::cout << "\n=== GPU加速示例 ===" << std::endl;
    
    try {
        // 创建标准匹配器
        auto cpu_matcher = EnhancedVideoMatcherFactory::createStandardMatcher();
        auto gpu_matcher = EnhancedVideoMatcherFactory::createStandardMatcher();
        
        // 配置参数
        EnhancedParameters params;
        params.video_name1 = "T10L.mp4";
        params.video_name2 = "T10R.mp4";
        params.dataset_path = "./Datasets";
        params.base_output_folder = "./Output";
        
        // CPU配置
        params.feature_config.use_gpu_acceleration = false;
        params.bidirectional_config.use_gpu_acceleration = false;
        params.performance_config.enable_gpu_acceleration = false;
        cpu_matcher->setParameters(params);
        
        // GPU配置
        params.feature_config.use_gpu_acceleration = true;
        params.bidirectional_config.use_gpu_acceleration = true;
        params.performance_config.enable_gpu_acceleration = true;
        gpu_matcher->setParameters(params);
        
        // 检查GPU可用性
        if (!gpu_matcher->isGPUAvailable()) {
            std::cout << "GPU不可用，跳过GPU加速测试" << std::endl;
            return;
        }
        
        std::cout << "GPU可用，开始性能对比测试..." << std::endl;
        
        // CPU处理
        std::cout << "CPU处理中..." << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_results = cpu_matcher->processVideoMatching();
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        
        // GPU处理
        std::cout << "GPU处理中..." << std::endl;
        auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_results = gpu_matcher->processVideoMatching();
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        
        // 性能对比
        std::cout << "\n性能对比结果:" << std::endl;
        std::cout << "  CPU处理时间: " << cpu_time.count() << " ms" << std::endl;
        std::cout << "  GPU处理时间: " << gpu_time.count() << " ms" << std::endl;
        
        if (gpu_time.count() > 0) {
            float speedup = static_cast<float>(cpu_time.count()) / gpu_time.count();
            std::cout << "  加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
        
        // 质量对比
        if (!cpu_results.empty() && !gpu_results.empty()) {
            float cpu_quality = cpu_results[0].average_quality;
            float gpu_quality = gpu_results[0].average_quality;
            std::cout << "  CPU匹配质量: " << cpu_quality << std::endl;
            std::cout << "  GPU匹配质量: " << gpu_quality << std::endl;
            std::cout << "  质量差异: " << std::abs(cpu_quality - gpu_quality) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "GPU加速测试错误: " << e.what() << std::endl;
    }
}

// 主函数
int main(int argc, char* argv[]) {
    std::cout << "FlowIntelligence 增强视频匹配系统示例" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // 检查命令行参数
        if (argc > 1) {
            std::string demo_type = argv[1];
            
            if (demo_type == "basic") {
                demonstrateBasicUsage();
            } else if (demo_type == "precision") {
                demonstrateHighPrecisionMatching();
            } else if (demo_type == "realtime") {
                demonstrateRealTimeProcessing();
            } else if (demo_type == "batch") {
                demonstrateBatchProcessing();
            } else if (demo_type == "adaptive") {
                demonstrateAdaptiveOptimization();
            } else if (demo_type == "gpu") {
                demonstrateGPUAcceleration();
            } else {
                std::cout << "未知的演示类型: " << demo_type << std::endl;
                std::cout << "可用类型: basic, precision, realtime, batch, adaptive, gpu" << std::endl;
                return 1;
            }
        } else {
            // 运行所有演示
            demonstrateBasicUsage();
            demonstrateHighPrecisionMatching();
            demonstrateRealTimeProcessing();
            demonstrateBatchProcessing();
            demonstrateAdaptiveOptimization();
            demonstrateGPUAcceleration();
        }
        
        std::cout << "\n所有演示完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "程序错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}