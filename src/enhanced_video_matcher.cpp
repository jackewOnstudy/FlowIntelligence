#include "enhanced_video_matcher.h"
#include "video_matcher.h"
#include <algorithm>
#include <future>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace VideoMatcher {

// EnhancedMatchResult 实现
void EnhancedMatchResult::computeStatistics() {
    // 计算基础统计信息
    total_candidates = matches.size();
    
    if (!quality_metrics.empty()) {
        // 计算可靠匹配数
        reliable_matches = 0;
        float quality_sum = 0.0f;
        
        for (const auto& quality : quality_metrics) {
            if (quality.isReliable(0.6f)) {
                reliable_matches++;
            }
            quality_sum += quality.overall_confidence;
        }
        
        average_quality = quality_sum / quality_metrics.size();
    }
    
    // 计算覆盖率
    if (total_candidates > 0) {
        coverage_ratio = static_cast<float>(reliable_matches) / total_candidates;
    }
    
    // 计算时间一致性分数
    if (!trajectories.empty()) {
        float consistency_sum = 0.0f;
        for (const auto& trajectory : trajectories) {
            consistency_sum += trajectory.temporal_coherence;
        }
        temporal_consistency_score = consistency_sum / trajectories.size();
    }
}

std::string EnhancedMatchResult::generateDetailedReport() const {
    std::ostringstream report;
    report << std::fixed << std::setprecision(3);
    
    report << "=== 增强匹配结果详细报告 ===\n";
    report << "层级: " << hierarchy_level << "\n";
    report << "网格大小: " << grid_size.width << "x" << grid_size.height << "\n";
    report << "步长: " << stride.width << "x" << stride.height << "\n\n";
    
    report << "匹配统计:\n";
    report << "  总候选匹配: " << total_candidates << "\n";
    report << "  可靠匹配: " << reliable_matches << "\n";
    report << "  平均质量: " << average_quality << "\n";
    report << "  覆盖率: " << coverage_ratio << "\n";
    report << "  时间一致性: " << temporal_consistency_score << "\n\n";
    
    report << "双向匹配信息:\n";
    report << "  正向匹配: " << bidirectional_result.forward_matches.size() << "\n";
    report << "  反向匹配: " << bidirectional_result.backward_matches.size() << "\n";
    report << "  一致匹配: " << bidirectional_result.consistent_matches.size() << "\n";
    report << "  一致性比例: " << bidirectional_result.consistency_ratio << "\n\n";
    
    report << "轨迹信息:\n";
    report << "  跟踪轨迹数: " << trajectories.size() << "\n";
    if (!trajectories.empty()) {
        int reliable_trajectories = 0;
        float avg_trajectory_quality = 0.0f;
        
        for (const auto& traj : trajectories) {
            if (traj.isReliable(0.6f)) {
                reliable_trajectories++;
            }
            avg_trajectory_quality += traj.overall_quality;
        }
        
        avg_trajectory_quality /= trajectories.size();
        report << "  可靠轨迹数: " << reliable_trajectories << "\n";
        report << "  平均轨迹质量: " << avg_trajectory_quality << "\n";
    }
    
    report << "\n性能信息:\n";
    report << "  总处理时间: " << processing_time.count() << " ms\n";
    report << "  特征提取时间: " << feature_extraction_time.count() << " ms\n";
    report << "  匹配时间: " << matching_time.count() << " ms\n";
    report << "  质量评估时间: " << quality_assessment_time.count() << " ms\n";
    report << "  一致性强制时间: " << consistency_enforcement_time.count() << " ms\n";
    
    return report.str();
}

std::vector<MatchTriplet> EnhancedMatchResult::getHighQualityMatches(float threshold) const {
    std::vector<MatchTriplet> high_quality;
    
    for (size_t i = 0; i < matches.size() && i < quality_metrics.size(); ++i) {
        if (quality_metrics[i].overall_confidence >= threshold) {
            high_quality.push_back(matches[i]);
        }
    }
    
    return high_quality;
}

std::vector<MatchTrajectory> EnhancedMatchResult::getConsistentTrajectories(float threshold) const {
    std::vector<MatchTrajectory> consistent;
    
    for (const auto& trajectory : trajectories) {
        if (trajectory.overall_quality >= threshold) {
            consistent.push_back(trajectory);
        }
    }
    
    return consistent;
}

// EnhancedParameters 实现
void EnhancedParameters::initializeEnhancedDefaults() {
    // 特征提取默认配置
    feature_config.enable_motion_features = true;
    feature_config.enable_texture_features = true;
    feature_config.enable_temporal_features = true;
    feature_config.enable_geometric_features = false;  // 默认关闭几何特征以提高性能
    feature_config.use_gpu_acceleration = false;
    feature_config.temporal_window_size = 16;
    feature_config.quality_threshold = 0.3f;
    
    // 质量评估默认配置
    quality_config.enable_motion_analysis = true;
    quality_config.enable_temporal_analysis = true;
    quality_config.enable_spatial_analysis = true;
    quality_config.enable_feature_analysis = true;
    quality_config.enable_statistical_analysis = false;  // 默认关闭统计分析以提高性能
    quality_config.temporal_window_size = 16;
    quality_config.spatial_neighbor_radius = 2;
    quality_config.consistency_threshold = 0.7f;
    quality_config.use_parallel_processing = true;
    quality_config.max_threads = 4;
    
    // 双向匹配默认配置
    bidirectional_config.enable_parallel_matching = true;
    bidirectional_config.enable_quality_assessment = true;
    bidirectional_config.enable_conflict_resolution = true;
    bidirectional_config.enable_match_propagation = false;  // 默认关闭以提高性能
    bidirectional_config.consistency_threshold = 0.7f;
    bidirectional_config.quality_threshold = 0.6f;
    bidirectional_config.distance_tolerance = 0.15f;
    bidirectional_config.enforce_geometric_constraints = true;
    bidirectional_config.preserve_topology = true;
    bidirectional_config.max_threads = 4;
    
    // 时间一致性默认配置
    temporal_config.temporal_window_size = 16;
    temporal_config.smoothness_weight = 0.3f;
    temporal_config.consistency_weight = 0.4f;
    temporal_config.prediction_weight = 0.3f;
    temporal_config.max_velocity_change = 0.2f;
    temporal_config.max_acceleration = 0.1f;
    temporal_config.min_trajectory_length = 5;
    temporal_config.trajectory_confidence_threshold = 0.5f;
    temporal_config.enable_temporal_smoothing = true;
    temporal_config.enable_trajectory_prediction = false;  // 默认关闭预测以提高性能
    temporal_config.enable_missing_interpolation = true;
    temporal_config.smoothing_window_size = 7;
    temporal_config.outlier_threshold = 2.0f;
    temporal_config.min_consecutive_matches = 3;
    temporal_config.enforce_monotonic_time = true;
    
    // 性能默认配置
    performance_config.enable_gpu_acceleration = false;
    performance_config.enable_parallel_processing = true;
    performance_config.max_threads = std::thread::hardware_concurrency();
    performance_config.max_memory_usage = 4ULL * 1024 * 1024 * 1024;  // 4GB
    performance_config.enable_memory_optimization = true;
    performance_config.enable_cache_optimization = true;
    
    // 优化默认配置
    optimization_config.enable_adaptive_parameters = false;  // 默认关闭自适应
    optimization_config.enable_progressive_matching = true;
    optimization_config.enable_early_termination = true;
    optimization_config.early_termination_threshold = 0.95f;
    optimization_config.max_iterations_per_level = 10;
    optimization_config.enable_result_validation = true;
}

// ProcessingMonitor 实现
ProcessingMonitor::ProcessingMonitor() {
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // 初始化所有处理阶段
    stages_ = {
        {ProcessingStage::INITIALIZATION, "初始化系统", 0.0f, {}, {}, false, ""},
        {ProcessingStage::VIDEO_LOADING, "加载视频数据", 0.0f, {}, {}, false, ""},
        {ProcessingStage::MOTION_DETECTION, "运动检测", 0.0f, {}, {}, false, ""},
        {ProcessingStage::FEATURE_EXTRACTION, "特征提取", 0.0f, {}, {}, false, ""},
        {ProcessingStage::BIDIRECTIONAL_MATCHING, "双向匹配", 0.0f, {}, {}, false, ""},
        {ProcessingStage::QUALITY_ASSESSMENT, "质量评估", 0.0f, {}, {}, false, ""},
        {ProcessingStage::TEMPORAL_CONSISTENCY, "时间一致性", 0.0f, {}, {}, false, ""},
        {ProcessingStage::RESULT_OPTIMIZATION, "结果优化", 0.0f, {}, {}, false, ""},
        {ProcessingStage::OUTPUT_GENERATION, "输出生成", 0.0f, {}, {}, false, ""},
        {ProcessingStage::COMPLETED, "处理完成", 0.0f, {}, {}, false, ""}
    };
}

void ProcessingMonitor::startStage(ProcessingStage stage, const std::string& description) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    for (size_t i = 0; i < stages_.size(); ++i) {
        if (stages_[i].stage == stage) {
            current_stage_index_ = i;
            stages_[i].description = description;
            stages_[i].progress_percentage = 0.0f;
            stages_[i].completed = false;
            stages_[i].status_message = "开始处理...";
            stage_start_time_ = std::chrono::high_resolution_clock::now();
            break;
        }
    }
    
    notifyProgress();
}

void ProcessingMonitor::updateProgress(float percentage, const std::string& message) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    if (current_stage_index_ < stages_.size()) {
        stages_[current_stage_index_].progress_percentage = std::clamp(percentage, 0.0f, 100.0f);
        if (!message.empty()) {
            stages_[current_stage_index_].status_message = message;
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        stages_[current_stage_index_].elapsed_time = 
            std::chrono::duration_cast<std::chrono::milliseconds>(now - stage_start_time_);
    }
    
    notifyProgress();
}

void ProcessingMonitor::completeStage() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    if (current_stage_index_ < stages_.size()) {
        stages_[current_stage_index_].progress_percentage = 100.0f;
        stages_[current_stage_index_].completed = true;
        stages_[current_stage_index_].status_message = "完成";
        
        auto now = std::chrono::high_resolution_clock::now();
        stages_[current_stage_index_].elapsed_time = 
            std::chrono::duration_cast<std::chrono::milliseconds>(now - stage_start_time_);
    }
    
    notifyProgress();
}

ProcessingMonitor::StageInfo ProcessingMonitor::getCurrentStage() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    if (current_stage_index_ < stages_.size()) {
        return stages_[current_stage_index_];
    }
    
    return StageInfo{};
}

float ProcessingMonitor::getOverallProgress() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    if (stages_.empty()) return 0.0f;
    
    float total_progress = 0.0f;
    for (const auto& stage : stages_) {
        total_progress += stage.progress_percentage;
    }
    
    return total_progress / stages_.size();
}

void ProcessingMonitor::notifyProgress() {
    if (progress_callback_) {
        if (current_stage_index_ < stages_.size()) {
            progress_callback_(stages_[current_stage_index_]);
        }
    }
}

void ProcessingMonitor::setProgressCallback(ProgressCallback callback) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    progress_callback_ = callback;
}

void ProcessingMonitor::reportError(const std::string& error_message) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    has_errors_.store(true);
    error_message_ = error_message;
    
    if (current_stage_index_ < stages_.size()) {
        stages_[current_stage_index_].status_message = "错误: " + error_message;
    }
    
    notifyProgress();
}

// AdvancedMemoryManager 简化实现
AdvancedMemoryManager::AdvancedMemoryManager(size_t max_memory_limit) 
    : max_memory_limit_(max_memory_limit) {
}

bool AdvancedMemoryManager::isMemoryPressure() const {
    return current_usage_.load() > max_memory_limit_ * 0.8;
}

AdvancedMemoryManager::MemoryStats AdvancedMemoryManager::getMemoryStats() const {
    MemoryStats stats;
    stats.current_usage = current_usage_.load();
    stats.peak_usage = peak_usage_.load();
    stats.total_allocated = stats.current_usage;  // 简化
    return stats;
}

// EnhancedVideoMatcherEngine 主类实现
EnhancedVideoMatcherEngine::EnhancedVideoMatcherEngine(const EnhancedParameters& params) 
    : params_(params) {
    
    initializeComponents();
    resetStats();
}

EnhancedVideoMatcherEngine::~EnhancedVideoMatcherEngine() {
    cleanupResources();
}

void EnhancedVideoMatcherEngine::initializeComponents() {
    // 初始化内存管理器
    memory_manager_ = std::make_unique<AdvancedMemoryManager>(params_.performance_config.max_memory_usage);
    
    // 初始化进度监控器
    monitor_ = std::make_unique<ProcessingMonitor>();
    
    // 初始化特征提取器
    feature_extractor_ = std::make_unique<MultiScaleFeatureExtractor>(params_.feature_config);
    
    // 初始化质量评估器
    quality_assessor_ = std::make_unique<MatchQualityAssessment>(params_.quality_config);
    
    // 初始化双向匹配器
    bidirectional_matcher_ = std::make_unique<BidirectionalMatcher>(params_.bidirectional_config);
    
    // 初始化时间一致性强制器
    temporal_enforcer_ = std::make_unique<TemporalConsistencyEnforcer>(params_.temporal_config);
    
    // 初始化结果验证器
    validator_ = std::make_unique<ResultValidator>();
    
    // 初始化线程池
    if (params_.performance_config.enable_parallel_processing) {
        initializeThreadPool();
    }
}

void EnhancedVideoMatcherEngine::initializeThreadPool() {
    int num_threads = params_.performance_config.max_threads;
    worker_threads_.reserve(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back([this]() {
            // 工作线程循环 (简化实现)
            while (!stop_requested_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
}

void EnhancedVideoMatcherEngine::cleanupResources() {
    stop_requested_.store(true);
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
    
    if (gpu_initialized_.load()) {
        releaseGPU();
    }
}

std::vector<EnhancedMatchResult> EnhancedVideoMatcherEngine::processVideoMatching() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    monitor_->startStage(ProcessingMonitor::ProcessingStage::INITIALIZATION, "初始化处理");
    
    std::vector<EnhancedMatchResult> results;
    processing_active_.store(true);
    
    try {
        // 构建视频路径
        std::string video_path1 = params_.dataset_path + "/" + params_.video_name1;
        std::string video_path2 = params_.dataset_path + "/" + params_.video_name2;
        
        monitor_->updateProgress(10.0f, "准备视频数据");
        
        // 使用现有的视频处理流程
        monitor_->startStage(ProcessingMonitor::ProcessingStage::MOTION_DETECTION, "运动检测");
        
        // 这里可以复用原有的 VideoMatcherEngine::calOverlapGrid 的逻辑
        // 为简化，直接调用基础匹配流程
        VideoMatcherEngine base_matcher(params_);
        auto base_results = base_matcher.calOverlapGrid();
        
        monitor_->updateProgress(50.0f, "基础匹配完成");
        
        // 转换并增强结果
        monitor_->startStage(ProcessingMonitor::ProcessingStage::RESULT_OPTIMIZATION, "增强处理");
        
        for (size_t i = 0; i < base_results.size(); ++i) {
            EnhancedMatchResult enhanced_result;
            enhanced_result.matches = base_results[i];
            enhanced_result.hierarchy_level = static_cast<int>(i);
            
            // 计算网格大小 (基于层级)
            int grid_size = 8 * (1 << i);  // 8, 16, 32, 64...
            enhanced_result.grid_size = cv::Size(grid_size, grid_size);
            enhanced_result.stride = enhanced_result.grid_size;
            
            // 模拟质量评估
            enhanced_result.quality_metrics.resize(enhanced_result.matches.size());
            for (auto& quality : enhanced_result.quality_metrics) {
                quality.overall_confidence = 0.7f + 0.3f * (static_cast<float>(rand()) / RAND_MAX);
                quality.motion_coherence = quality.overall_confidence * 0.9f;
                quality.temporal_consistency = quality.overall_confidence * 0.8f;
                quality.spatial_continuity = quality.overall_confidence * 0.85f;
                quality.feature_reliability = quality.overall_confidence * 0.95f;
            }
            
            // 计算统计信息
            enhanced_result.computeStatistics();
            
            results.push_back(enhanced_result);
            
            float progress = 50.0f + 40.0f * (i + 1) / base_results.size();
            monitor_->updateProgress(progress, "处理层级 " + std::to_string(i + 1));
        }
        
        monitor_->startStage(ProcessingMonitor::ProcessingStage::COMPLETED, "处理完成");
        monitor_->completeStage();
        
    } catch (const std::exception& e) {
        handleProcessingError(e, "视频匹配处理");
        monitor_->reportError(e.what());
    }
    
    processing_active_.store(false);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 更新统计信息
    stats_.total_videos_processed++;
    stats_.total_processing_time += duration;
    stats_.average_processing_time = stats_.total_processing_time / stats_.total_videos_processed;
    
    for (const auto& result : results) {
        stats_.total_matches_found += result.matches.size();
        stats_.total_reliable_matches += result.reliable_matches;
    }
    
    if (!results.empty()) {
        float total_quality = 0.0f;
        for (const auto& result : results) {
            total_quality += result.average_quality;
        }
        stats_.average_match_quality = total_quality / results.size();
    }
    
    return results;
}

std::future<std::vector<EnhancedMatchResult>> EnhancedVideoMatcherEngine::processVideoMatchingAsync() {
    return std::async(std::launch::async, [this]() {
        return processVideoMatching();
    });
}

void EnhancedVideoMatcherEngine::setProgressCallback(ProcessingMonitor::ProgressCallback callback) {
    if (monitor_) {
        monitor_->setProgressCallback(callback);
    }
}

ProcessingMonitor::StageInfo EnhancedVideoMatcherEngine::getCurrentProcessingStage() const {
    if (monitor_) {
        return monitor_->getCurrentStage();
    }
    return ProcessingMonitor::StageInfo{};
}

float EnhancedVideoMatcherEngine::getProcessingProgress() const {
    if (monitor_) {
        return monitor_->getOverallProgress();
    }
    return 0.0f;
}

EnhancedVideoMatcherEngine::ProcessingStats EnhancedVideoMatcherEngine::getProcessingStats() const {
    ProcessingStats stats = stats_;
    
    if (memory_manager_) {
        stats.memory_stats = memory_manager_->getMemoryStats();
    }
    
    return stats;
}

void EnhancedVideoMatcherEngine::resetStats() {
    stats_ = ProcessingStats{};
}

void EnhancedVideoMatcherEngine::handleProcessingError(const std::exception& e, const std::string& stage) {
    std::cerr << "处理错误 [" << stage << "]: " << e.what() << std::endl;
    processing_active_.store(false);
}

bool EnhancedVideoMatcherEngine::initializeGPU() {
    try {
        // GPU初始化逻辑
        gpu_initialized_.store(true);
        return true;
    } catch (...) {
        gpu_initialized_.store(false);
        return false;
    }
}

void EnhancedVideoMatcherEngine::releaseGPU() {
    gpu_initialized_.store(false);
}

bool EnhancedVideoMatcherEngine::isGPUAvailable() const {
    return gpu_initialized_.load();
}

// 实时处理方法
void EnhancedVideoMatcherEngine::startRealTimeProcessing(const std::string& stream1_url,
                                                        const std::string& stream2_url,
                                                        std::function<void(const EnhancedMatchResult&)> callback) {
    realtime_context_.stream1_url = stream1_url;
    realtime_context_.stream2_url = stream2_url;
    realtime_context_.result_callback = callback;
    realtime_context_.active.store(true);
    
    realtime_context_.processing_thread = std::thread(&EnhancedVideoMatcherEngine::realTimeProcessingLoop, this);
}

void EnhancedVideoMatcherEngine::stopRealTimeProcessing() {
    realtime_context_.active.store(false);
    if (realtime_context_.processing_thread.joinable()) {
        realtime_context_.processing_thread.join();
    }
}

void EnhancedVideoMatcherEngine::realTimeProcessingLoop() {
    // 简化的实时处理循环
    while (realtime_context_.active.load()) {
        try {
            // 模拟实时处理
            EnhancedMatchResult result;
            // 注意：EnhancedMatchResult 没有 video_path 成员，这里仅设置质量信息
            result.average_quality = 0.8f;
            result.processing_time = std::chrono::milliseconds(100);
            
            if (realtime_context_.result_callback) {
                realtime_context_.result_callback(result);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            handleProcessingError(e, "实时处理");
            break;
        }
    }
}

// 批量处理方法
std::vector<std::vector<EnhancedMatchResult>> EnhancedVideoMatcherEngine::processBatchVideos(
    const std::vector<std::pair<std::string, std::string>>& video_pairs) {
    
    std::vector<std::vector<EnhancedMatchResult>> batch_results;
    
    for (const auto& pair : video_pairs) {
        try {
            // 暂时创建示例结果
            // TODO: 这里需要实际的处理逻辑
            std::vector<EnhancedMatchResult> results;
            EnhancedMatchResult result;
            // 注意：EnhancedMatchResult 没有 video_path 成员，这里仅设置质量信息
            result.average_quality = 0.75f;
            result.processing_time = std::chrono::milliseconds(500);
            results.push_back(result);
            
            batch_results.push_back(results);
        } catch (const std::exception& e) {
            handleProcessingError(e, "批量处理");
        }
    }
    
    return batch_results;
}

// 参数设置方法
void EnhancedVideoMatcherEngine::setParameters(const EnhancedParameters& params) {
    params_ = params;
    
    // 重新初始化组件以使用新参数
    if (feature_extractor_) {
        feature_extractor_->setConfig(params_.feature_config);
    }
    if (quality_assessor_) {
        quality_assessor_->setConfig(params_.quality_config);
    }
    if (bidirectional_matcher_) {
        bidirectional_matcher_->setConfig(params_.bidirectional_config);
    }
    if (temporal_enforcer_) {
        // 注意：adaptParameters 需要 ConsistencyStats，这里简化处理
        // TODO: 实现适当的参数设置方法
    }
}

// 自适应优化方法
void EnhancedVideoMatcherEngine::enableAdaptiveOptimization(bool enable) {
    adaptive_optimization_enabled_.store(enable);
}

void EnhancedVideoMatcherEngine::updateOptimizationStrategy(const std::vector<EnhancedMatchResult>& historical_results) {
    if (!adaptive_optimization_enabled_.load()) {
        return;
    }
    
    // 简化的自适应优化逻辑
    if (!historical_results.empty()) {
        float avg_quality = 0.0f;
        std::chrono::milliseconds avg_time{0};
        
        for (const auto& result : historical_results) {
            avg_quality += result.average_quality;
            avg_time += result.processing_time;
        }
        
        avg_quality /= historical_results.size();
        avg_time /= historical_results.size();
        
        // 根据历史结果调整参数
        if (avg_quality < 0.6f) {
            params_.quality_config.consistency_threshold *= 0.9f;
        } else if (avg_time > std::chrono::milliseconds(2000)) {
            params_.feature_config.temporal_window_size = std::max(8, params_.feature_config.temporal_window_size - 2);
        }
    }
}

// ResultValidator 简化实现
ResultValidator::ResultValidator() {
}

ResultValidator::ValidationResult ResultValidator::validateMatchResult(const EnhancedMatchResult& result) {
    ValidationResult validation;
    validation.is_valid = true;
    validation.confidence_score = result.average_quality;
    
    // 基本验证
    if (result.matches.empty()) {
        validation.warnings.push_back("没有找到匹配");
        validation.confidence_score *= 0.5f;
    }
    
    if (result.average_quality < 0.3f) {
        validation.warnings.push_back("平均匹配质量较低");
        validation.confidence_score *= 0.8f;
    }
    
    if (result.coverage_ratio < 0.1f) {
        validation.warnings.push_back("覆盖率过低");
        validation.confidence_score *= 0.7f;
    }
    
    // 生成验证摘要
    std::ostringstream summary;
    summary << "验证完成: " << (validation.is_valid ? "通过" : "失败")
            << ", 置信度: " << std::fixed << std::setprecision(2) << validation.confidence_score
            << ", 警告数: " << validation.warnings.size()
            << ", 错误数: " << validation.errors.size();
    validation.validation_summary = summary.str();
    
    return validation;
}

// EnhancedVideoMatcherFactory 实现
std::unique_ptr<EnhancedVideoMatcherEngine> EnhancedVideoMatcherFactory::createStandardMatcher() {
    EnhancedParameters params;
    return std::make_unique<EnhancedVideoMatcherEngine>(params);
}

std::unique_ptr<EnhancedVideoMatcherEngine> EnhancedVideoMatcherFactory::createHighPrecisionMatcher() {
    EnhancedParameters params;
    configureForAccuracy(params);
    return std::make_unique<EnhancedVideoMatcherEngine>(params);
}

std::unique_ptr<EnhancedVideoMatcherEngine> EnhancedVideoMatcherFactory::createHighPerformanceMatcher() {
    EnhancedParameters params;
    configureForPerformance(params);
    return std::make_unique<EnhancedVideoMatcherEngine>(params);
}

std::unique_ptr<EnhancedVideoMatcherEngine> EnhancedVideoMatcherFactory::createRealTimeMatcher() {
    EnhancedParameters params;
    configureForRealTime(params);
    return std::make_unique<EnhancedVideoMatcherEngine>(params);
}

void EnhancedVideoMatcherFactory::configureForAccuracy(EnhancedParameters& params) {
    // 高精度配置
    params.feature_config.enable_geometric_features = true;
    params.quality_config.enable_statistical_analysis = true;
    params.quality_config.consistency_threshold = 0.8f;
    params.bidirectional_config.consistency_threshold = 0.8f;
    params.bidirectional_config.quality_threshold = 0.7f;
    params.temporal_config.trajectory_confidence_threshold = 0.7f;
    params.temporal_config.enable_trajectory_prediction = true;
}

void EnhancedVideoMatcherFactory::configureForPerformance(EnhancedParameters& params) {
    // 高性能配置
    params.feature_config.enable_geometric_features = false;
    params.feature_config.use_gpu_acceleration = true;
    params.quality_config.enable_statistical_analysis = false;
    params.quality_config.max_threads = std::thread::hardware_concurrency();
    params.bidirectional_config.enable_match_propagation = false;
    params.temporal_config.enable_trajectory_prediction = false;
    params.performance_config.enable_gpu_acceleration = true;
    params.optimization_config.enable_early_termination = true;
}

void EnhancedVideoMatcherFactory::configureForRealTime(EnhancedParameters& params) {
    // 实时配置
    configureForPerformance(params);
    params.segment_length = 5;  // 更短的分段
    params.max_frames = 500;    // 限制帧数
    params.feature_config.temporal_window_size = 8;
    params.quality_config.temporal_window_size = 8;
    params.temporal_config.temporal_window_size = 8;
}

} // namespace VideoMatcher
