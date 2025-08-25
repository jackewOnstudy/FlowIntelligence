#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <atomic>
#include <chrono>

#include "video_matcher.h"
#include "feature_extractor.h"
#include "match_quality_assessor.h"
#include "bidirectional_matcher.h"
#include "temporal_consistency_enforcer.h"
#include "parameters.h"

namespace VideoMatcher {

// 增强匹配结果
struct EnhancedMatchResult {
    // 基础匹配信息
    std::vector<MatchTriplet> matches;
    std::vector<QualityMetrics> quality_metrics;
    std::vector<FeatureDescriptor> features1, features2;
    
    // 双向匹配信息
    BidirectionalMatchResult bidirectional_result;
    
    // 时间一致性信息
    std::vector<MatchTrajectory> trajectories;
    float temporal_consistency_score = 0.0f;
    
    // 层级信息
    int hierarchy_level = 0;
    cv::Size grid_size;
    cv::Size stride;
    
    // 统计信息
    size_t total_candidates = 0;
    size_t reliable_matches = 0;
    float average_quality = 0.0f;
    float coverage_ratio = 0.0f;
    
    // 性能信息
    std::chrono::milliseconds processing_time{0};
    std::chrono::milliseconds feature_extraction_time{0};
    std::chrono::milliseconds matching_time{0};
    std::chrono::milliseconds quality_assessment_time{0};
    std::chrono::milliseconds consistency_enforcement_time{0};
    
    // 计算统计信息
    void computeStatistics();
    
    // 生成详细报告
    std::string generateDetailedReport() const;
    
    // 获取高质量匹配
    std::vector<MatchTriplet> getHighQualityMatches(float threshold = 0.7f) const;
    
    // 获取一致轨迹
    std::vector<MatchTrajectory> getConsistentTrajectories(float threshold = 0.6f) const;
};

// 增强参数配置
struct EnhancedParameters : public Parameters {
    // 特征提取配置
    MultiScaleFeatureExtractor::ExtractionConfig feature_config;
    
    // 质量评估配置
    QualityAssessmentConfig quality_config;
    
    // 双向匹配配置
    BidirectionalMatcher::MatchingConfig bidirectional_config;
    
    // 时间一致性配置
    TemporalConstraintConfig temporal_config;
    
    // 性能配置
    struct PerformanceConfig {
        bool enable_gpu_acceleration = false;
        bool enable_parallel_processing = true;
        int max_threads = std::thread::hardware_concurrency();
        size_t max_memory_usage = 4ULL * 1024 * 1024 * 1024; // 4GB
        bool enable_memory_optimization = true;
        bool enable_cache_optimization = true;
    } performance_config;
    
    // 优化配置
    struct OptimizationConfig {
        bool enable_adaptive_parameters = true;
        bool enable_progressive_matching = true;
        bool enable_early_termination = true;
        float early_termination_threshold = 0.95f;
        int max_iterations_per_level = 10;
        bool enable_result_validation = true;
    } optimization_config;
    
    // 默认构造函数
    EnhancedParameters() : Parameters() {
        // 初始化增强参数的默认值
        initializeEnhancedDefaults();
    }
    
private:
    void initializeEnhancedDefaults();
};

// 处理状态监控
class ProcessingMonitor {
public:
    enum class ProcessingStage {
        INITIALIZATION,
        VIDEO_LOADING,
        MOTION_DETECTION,
        FEATURE_EXTRACTION,
        BIDIRECTIONAL_MATCHING,
        QUALITY_ASSESSMENT,
        TEMPORAL_CONSISTENCY,
        RESULT_OPTIMIZATION,
        OUTPUT_GENERATION,
        COMPLETED
    };
    
    struct StageInfo {
        ProcessingStage stage;
        std::string description;
        float progress_percentage = 0.0f;
        std::chrono::milliseconds elapsed_time{0};
        std::chrono::milliseconds estimated_remaining{0};
        bool completed = false;
        std::string status_message;
    };
    
    ProcessingMonitor();
    
    // 阶段管理
    void startStage(ProcessingStage stage, const std::string& description);
    void updateProgress(float percentage, const std::string& message = "");
    void completeStage();
    void reportError(const std::string& error_message);
    
    // 状态查询
    StageInfo getCurrentStage() const;
    float getOverallProgress() const;
    std::vector<StageInfo> getAllStages() const;
    bool isCompleted() const;
    bool hasErrors() const;
    
    // 性能统计
    std::chrono::milliseconds getTotalElapsedTime() const;
    std::chrono::milliseconds getEstimatedRemainingTime() const;
    
    // 回调设置
    using ProgressCallback = std::function<void(const StageInfo&)>;
    void setProgressCallback(ProgressCallback callback);
    
private:
    std::vector<StageInfo> stages_;
    size_t current_stage_index_ = 0;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stage_start_time_;
    ProgressCallback progress_callback_;
    std::atomic<bool> has_errors_{false};
    std::string error_message_;
    mutable std::mutex monitor_mutex_;
    
    void notifyProgress();
    std::chrono::milliseconds estimateRemainingTime() const;
};

// 内存管理器
class AdvancedMemoryManager {
public:
    struct MemoryStats {
        size_t total_allocated = 0;
        size_t peak_usage = 0;
        size_t current_usage = 0;
        size_t cache_usage = 0;
        float fragmentation_ratio = 0.0f;
        size_t allocation_count = 0;
        size_t deallocation_count = 0;
    };
    
    explicit AdvancedMemoryManager(size_t max_memory_limit);
    
    // 内存分配管理
    template<typename T>
    std::unique_ptr<T[]> allocate(size_t count);
    
    template<typename T>
    void deallocate(std::unique_ptr<T[]>& ptr);
    
    // 缓存管理
    template<typename T>
    void cacheData(const std::string& key, std::unique_ptr<T[]> data, size_t size);
    
    template<typename T>
    std::unique_ptr<T[]> getCachedData(const std::string& key, size_t& size);
    
    // 内存监控
    MemoryStats getMemoryStats() const;
    bool isMemoryPressure() const;
    void triggerGarbageCollection();
    
    // 预分配池
    void preallocatePool(size_t element_size, size_t count);
    void* getFromPool(size_t size);
    void returnToPool(void* ptr, size_t size);
    
private:
    size_t max_memory_limit_;
    std::atomic<size_t> current_usage_{0};
    std::atomic<size_t> peak_usage_{0};
    mutable std::mutex memory_mutex_;
    
    // 缓存系统
    struct CacheEntry {
        std::unique_ptr<char[]> data;
        size_t size;
        std::chrono::steady_clock::time_point last_access;
        size_t access_count;
    };
    std::unordered_map<std::string, CacheEntry> cache_;
    
    // 内存池
    struct MemoryPool {
        std::vector<std::unique_ptr<char[]>> blocks;
        std::vector<bool> availability;
        size_t block_size;
        size_t block_count;
    };
    std::unordered_map<size_t, MemoryPool> memory_pools_;
    
    void cleanupCache();
    void optimizeMemoryLayout();
};

// 结果验证器
class ResultValidator {
public:
    struct ValidationResult {
        bool is_valid = true;
        float confidence_score = 0.0f;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        std::string validation_summary;
    };
    
    ResultValidator();
    
    // 验证匹配结果
    ValidationResult validateMatchResult(const EnhancedMatchResult& result);
    
    // 验证几何一致性
    bool validateGeometricConsistency(const std::vector<MatchTriplet>& matches,
                                     const cv::Size& grid_layout1,
                                     const cv::Size& grid_layout2);
    
    // 验证时间一致性
    bool validateTemporalConsistency(const std::vector<MatchTrajectory>& trajectories);
    
    // 验证质量分布
    bool validateQualityDistribution(const std::vector<QualityMetrics>& qualities);
    
    // 验证覆盖率
    bool validateCoverage(const std::vector<MatchTriplet>& matches,
                         size_t total_grids1, size_t total_grids2,
                         float min_coverage = 0.1f);
    
private:
    // 统计验证
    bool performStatisticalValidation(const std::vector<float>& values,
                                     float min_value, float max_value);
    
    // 异常检测
    std::vector<size_t> detectAnomalousMatches(const std::vector<MatchTriplet>& matches);
    
    // 拓扑验证
    bool validateTopology(const std::vector<MatchTriplet>& matches,
                         const cv::Size& layout1, const cv::Size& layout2);
};

// 增强的视频匹配引擎
class EnhancedVideoMatcherEngine {
public:
    explicit EnhancedVideoMatcherEngine(const EnhancedParameters& params = EnhancedParameters{});
    ~EnhancedVideoMatcherEngine();
    
    // 禁用复制，允许移动
    EnhancedVideoMatcherEngine(const EnhancedVideoMatcherEngine&) = delete;
    EnhancedVideoMatcherEngine& operator=(const EnhancedVideoMatcherEngine&) = delete;
    EnhancedVideoMatcherEngine(EnhancedVideoMatcherEngine&&) = default;
    EnhancedVideoMatcherEngine& operator=(EnhancedVideoMatcherEngine&&) = default;
    
    // 主要处理接口
    std::vector<EnhancedMatchResult> processVideoMatching();
    
    // 异步处理
    std::future<std::vector<EnhancedMatchResult>> processVideoMatchingAsync();
    
    // 实时处理 (流式)
    void startRealTimeProcessing(const std::string& stream1_url,
                                const std::string& stream2_url,
                                std::function<void(const EnhancedMatchResult&)> callback);
    void stopRealTimeProcessing();
    
    // 批量处理
    std::vector<std::vector<EnhancedMatchResult>> processBatchVideos(
        const std::vector<std::pair<std::string, std::string>>& video_pairs);
    
    // 配置管理
    void setParameters(const EnhancedParameters& params);
    const EnhancedParameters& getParameters() const { return params_; }
    
    // 进度监控
    void setProgressCallback(ProcessingMonitor::ProgressCallback callback);
    ProcessingMonitor::StageInfo getCurrentProcessingStage() const;
    float getProcessingProgress() const;
    
    // 性能统计
    struct ProcessingStats {
        size_t total_videos_processed = 0;
        size_t total_matches_found = 0;
        size_t total_reliable_matches = 0;
        std::chrono::milliseconds total_processing_time{0};
        std::chrono::milliseconds average_processing_time{0};
        float average_match_quality = 0.0f;
        float average_temporal_consistency = 0.0f;
        AdvancedMemoryManager::MemoryStats memory_stats;
    };
    
    ProcessingStats getProcessingStats() const;
    void resetStats();
    
    // 缓存管理
    void enableResultCaching(bool enable);
    void clearCache();
    size_t getCacheSize() const;
    
    // GPU资源管理
    bool initializeGPU();
    void releaseGPU();
    bool isGPUAvailable() const;
    
    // 自适应优化
    void enableAdaptiveOptimization(bool enable);
    void updateOptimizationStrategy(const std::vector<EnhancedMatchResult>& historical_results);
    
private:
    EnhancedParameters params_;
    
    // 核心组件
    std::unique_ptr<MultiScaleFeatureExtractor> feature_extractor_;
    std::unique_ptr<MatchQualityAssessment> quality_assessor_;
    std::unique_ptr<BidirectionalMatcher> bidirectional_matcher_;
    std::unique_ptr<TemporalConsistencyEnforcer> temporal_enforcer_;
    
    // 辅助组件
    std::unique_ptr<ProcessingMonitor> monitor_;
    std::unique_ptr<AdvancedMemoryManager> memory_manager_;
    std::unique_ptr<ResultValidator> validator_;
    
    // 性能统计
    mutable ProcessingStats stats_;
    
    // 状态管理
    std::atomic<bool> processing_active_{false};
    std::atomic<bool> gpu_initialized_{false};
    std::atomic<bool> adaptive_optimization_enabled_{false};
    
    // 线程管理
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> stop_requested_{false};
    
    // 缓存系统
    bool caching_enabled_ = true;
    std::unordered_map<std::string, EnhancedMatchResult> result_cache_;
    mutable std::mutex cache_mutex_;
    
    // 自适应参数
    struct AdaptiveState {
        float current_quality_threshold = 0.6f;
        float current_consistency_threshold = 0.7f;
        int current_feature_complexity = 1;
        std::vector<float> quality_history;
        std::vector<float> performance_history;
    } adaptive_state_;
    
    // 初始化方法
    void initializeComponents();
    void initializeThreadPool();
    void cleanupResources();
    
    // 核心处理流程
    EnhancedMatchResult processVideoLevel(int level,
                                         const cv::Mat& motion_status1,
                                         const cv::Mat& motion_status2,
                                         const cv::Size& grid_size,
                                         const cv::Size& stride);
    
    // 特征提取流程
    std::pair<std::vector<FeatureDescriptor>, std::vector<FeatureDescriptor>> 
    extractFeatures(const std::vector<cv::Mat>& frames1,
                   const std::vector<cv::Mat>& frames2,
                   const cv::Mat& motion_status1,
                   const cv::Mat& motion_status2,
                   const cv::Size& grid_size);
    
    // 匹配流程
    BidirectionalMatchResult performMatching(
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2);
    
    // 质量评估流程
    std::vector<QualityMetrics> assessMatchingQuality(
        const std::vector<MatchTriplet>& matches,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2);
    
    // 时间一致性流程
    std::vector<MatchTrajectory> enforceTemporalConsistency(
        const std::vector<std::vector<MatchTriplet>>& hierarchical_matches,
        const std::vector<std::vector<QualityMetrics>>& hierarchical_qualities);
    
    // 结果优化
    EnhancedMatchResult optimizeResult(const EnhancedMatchResult& raw_result);
    
    // 并行处理支持
    template<typename Func, typename... Args>
    auto executeInParallel(Func&& func, Args&&... args) 
        -> std::future<decltype(func(args...))>;
    
    // 缓存管理
    std::string generateCacheKey(const std::string& video1, const std::string& video2,
                                const EnhancedParameters& params);
    bool loadFromCache(const std::string& cache_key, EnhancedMatchResult& result);
    void saveToCache(const std::string& cache_key, const EnhancedMatchResult& result);
    
    // 自适应优化
    void analyzePerformance(const EnhancedMatchResult& result);
    void adjustParameters();
    void updateAdaptiveState(const EnhancedMatchResult& result);
    
    // 错误处理
    void handleProcessingError(const std::exception& e, const std::string& stage);
    
    // 资源监控
    void monitorResourceUsage();
    void optimizeResourceAllocation();
    
    // GPU操作
    void transferToGPU(const cv::Mat& data, cv::cuda::GpuMat& gpu_data);
    void transferFromGPU(const cv::cuda::GpuMat& gpu_data, cv::Mat& data);
    
    // 实时处理支持
    struct RealTimeContext {
        std::atomic<bool> active{false};
        std::thread processing_thread;
        std::function<void(const EnhancedMatchResult&)> result_callback;
        std::string stream1_url, stream2_url;
    } realtime_context_;
    
    void realTimeProcessingLoop();
};

// 工厂类
class EnhancedVideoMatcherFactory {
public:
    // 创建标准配置的匹配器
    static std::unique_ptr<EnhancedVideoMatcherEngine> createStandardMatcher();
    
    // 创建高精度配置的匹配器
    static std::unique_ptr<EnhancedVideoMatcherEngine> createHighPrecisionMatcher();
    
    // 创建高性能配置的匹配器
    static std::unique_ptr<EnhancedVideoMatcherEngine> createHighPerformanceMatcher();
    
    // 创建实时配置的匹配器
    static std::unique_ptr<EnhancedVideoMatcherEngine> createRealTimeMatcher();
    
    // 创建自定义配置的匹配器
    static std::unique_ptr<EnhancedVideoMatcherEngine> createCustomMatcher(
        const EnhancedParameters& params);
    
    // 获取推荐配置
    static EnhancedParameters getRecommendedParameters(
        const cv::Size& video_resolution,
        float expected_motion_level,
        bool require_real_time);
    
private:
    static void configureForAccuracy(EnhancedParameters& params);
    static void configureForPerformance(EnhancedParameters& params);
    static void configureForRealTime(EnhancedParameters& params);
};

} // namespace VideoMatcher