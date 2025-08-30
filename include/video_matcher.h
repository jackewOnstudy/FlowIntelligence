#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <unordered_map>  // 新增：高效哈希映射
#include <unordered_set>  // 新增：高效哈希集合
#include "parameters.h"
#include "time_alignment.h"  // 时间对齐模块

namespace VideoMatcher {

// 内存池管理类 - 新增优化
template<typename T>
class ObjectPool {
private:
    std::vector<std::unique_ptr<T>> pool_;
    std::mutex mutex_;
    
public:
    std::unique_ptr<T> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            auto obj = std::move(pool_.back());
            pool_.pop_back();
            return obj;
        }
        return std::make_unique<T>();
    }
    
    void release(std::unique_ptr<T> obj) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(obj));
    }
};

// 优化的匹配三元组结构 - 内存对齐优化
struct alignas(16) MatchTriplet {
    int grid1;
    int grid2;
    float distance;
    float padding_;  // 填充以确保16字节对齐
    
    MatchTriplet() : grid1(0), grid2(0), distance(0.0f), padding_(0.0f) {}
    MatchTriplet(int g1, int g2, float dist) : grid1(g1), grid2(g2), distance(dist), padding_(0.0f) {}
    
    // 显式定义拷贝语义以保持兼容性
    MatchTriplet(const MatchTriplet& other) = default;
    MatchTriplet& operator=(const MatchTriplet& other) = default;
    
    // 移动语义支持 - 新增优化
    MatchTriplet(MatchTriplet&& other) noexcept 
        : grid1(other.grid1), grid2(other.grid2), distance(other.distance), padding_(other.padding_) {}
    
    MatchTriplet& operator=(MatchTriplet&& other) noexcept {
        if (this != &other) {
            grid1 = other.grid1;
            grid2 = other.grid2;
            distance = other.distance;
            padding_ = other.padding_;
        }
        return *this;
    }
};

// 优化的Patch信息结构 - 减少内存占用
struct PatchInfo {
    cv::Point2i position;             // patch位置
    std::vector<bool> motion_sequence; // 运动状态序列 - 使用位压缩
    float confidence = 0.0f;          // 置信度
    uint16_t motion_count = 0;        // 运动像素计数 - 使用更小的类型
    
    PatchInfo() = default;
    PatchInfo(const cv::Point2i& pos) : position(pos) {}
    
    // 预分配内存 - 新增优化
    void reserve(size_t size) {
        motion_sequence.reserve(size);
    }
    
    // 移动语义支持
    PatchInfo(PatchInfo&& other) noexcept 
        : position(other.position), motion_sequence(std::move(other.motion_sequence)),
          confidence(other.confidence), motion_count(other.motion_count) {}
    
    PatchInfo& operator=(PatchInfo&& other) noexcept {
        if (this != &other) {
            position = other.position;
            motion_sequence = std::move(other.motion_sequence);
            confidence = other.confidence;
            motion_count = other.motion_count;
        }
        return *this;
    }
};

// 优化的匹配结果结构
struct MatchResult {
    PatchInfo patch1, patch2;
    float distance = 1.0f;           // 匹配距离
    float confidence = 0.0f;         // 匹配置信度
    uint8_t hierarchy_level = 0;     // 分层级别 - 使用更小的类型
    
    bool operator<(const MatchResult& other) const {
        return distance < other.distance;
    }
    
    // 移动语义支持
    MatchResult(MatchResult&& other) noexcept 
        : patch1(std::move(other.patch1)), patch2(std::move(other.patch2)),
          distance(other.distance), confidence(other.confidence), hierarchy_level(other.hierarchy_level) {}
    
    MatchResult& operator=(MatchResult&& other) noexcept {
        if (this != &other) {
            patch1 = std::move(other.patch1);
            patch2 = std::move(other.patch2);
            distance = other.distance;
            confidence = other.confidence;
            hierarchy_level = other.hierarchy_level;
        }
        return *this;
    }
};

// 内存优化的缓存管理类 - 新增优化
class MemoryCache {
private:
    std::unordered_map<std::string, cv::Mat> matrix_cache_;
    std::unordered_map<std::string, std::vector<bool>> sequence_cache_;
    std::mutex cache_mutex_;
    size_t max_cache_size_ = 100;  // 最大缓存项数
    
public:
    bool getMatrix(const std::string& key, cv::Mat& matrix) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = matrix_cache_.find(key);
        if (it != matrix_cache_.end()) {
            matrix = it->second;
            return true;
        }
        return false;
    }
    
    void putMatrix(const std::string& key, const cv::Mat& matrix) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (matrix_cache_.size() >= max_cache_size_) {
            matrix_cache_.erase(matrix_cache_.begin());
        }
        matrix_cache_[key] = matrix;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        matrix_cache_.clear();
        sequence_cache_.clear();
    }
};

// 距离计算类
class DistanceCalculator {
public:
    static float logicAndDistance(const std::vector<bool>& x, const std::vector<bool>& y);
    static float logicXorDistance(const std::vector<bool>& x, const std::vector<bool>& y);
    static float segmentSimilarity(const std::vector<bool>& segment1, const std::vector<bool>& segment2, 
                                  const std::string& distance_metric);
    
    // 批量距离计算 - 新增优化
    static std::vector<float> batchSegmentSimilarity(
        const std::vector<std::vector<bool>>& segments1,
        const std::vector<std::vector<bool>>& segments2,
        const std::string& distance_metric);
};

// 分段匹配类
class SegmentMatcher {
public:
    static std::vector<MatchTriplet> findMatchingGridWithSegment(
        const cv::Mat& motion_status_matrix1, const cv::Mat& motion_status_matrix2,
        const Parameters& parameters, const std::map<int, std::set<int>>& sorted_large_grid_corre_small_dict,
        int small_grid_cols, int large_grid_cols, bool shifting_flag);
    
    static std::vector<MatchTriplet> propagateMatchingResult(
        const std::vector<MatchTriplet>& match_result,
        const cv::Mat& motion_status_per_grid1, const cv::Mat& motion_status_per_grid2,
        int num_rows1, int num_cols1, int num_rows2, int num_cols2,
        const Parameters& parameters, bool shifting_flag);
    
    static std::vector<std::vector<bool>> segmentSequence(const std::vector<bool>& sequence, int segment_length);
    static std::vector<cv::Mat> segmentMatrix(const cv::Mat& matrix, int segment_length);
    
    // 内存池支持 - 新增优化
    static ObjectPool<std::vector<bool>>& getSequencePool() {
        static ObjectPool<std::vector<bool>> pool;
        return pool;
    }
};

// 工具函数类
class VideoMatcherUtils {
public:
    // 文件保存和加载
    static void saveNpyFiles(const std::string& video_path, const cv::Mat& motion_data, 
                            const cv::Size& grid_size, const std::string& output_path);
    static cv::Mat loadNpyFiles(const std::string& video_path, const cv::Size& grid_size, 
                               const std::string& output_path);
    
    // 运动检测
    static cv::Mat getMotionCountWithShiftingGrid(const std::string& video_path, 
                                                 const cv::Size& stride, const cv::Size& grid_size,
                                                 const Parameters& params, int& num_cols, int& num_rows);
    
    // 网格构建
    static cv::Mat get4nGridMotionCount(const cv::Mat& motion_count_per_grid, 
                                       int& col_grid_num, int& row_grid_num, bool shifting_flag);
    
    // 运动状态计算
    static cv::Mat getMotionStatus(const cv::Mat& motion_count, int motion_threshold);
    
    // Otsu自适应阈值相关功能
    static cv::Mat getMotionCountWithOtsu(const std::string& video_path, 
                                         const cv::Size& stride, const cv::Size& grid_size,
                                         const Parameters& params, int& num_cols, int& num_rows);
    static cv::Mat getMotionStatusWithOtsu(const cv::Mat& motion_count, float min_threshold = 1.0f);
    static cv::Mat getMotionStatusGlobalOtsu(const cv::Mat& motion_count, float min_threshold = 1.0f);
    static double calculateOtsuThreshold(const cv::Mat& data);
    
    // 结果处理 
    static std::vector<MatchTriplet> processTriplets(const std::vector<MatchTriplet>& triplets);
    
    // 匹配结果保存 
    static void saveMatchResultList(const std::string& video_path, const std::vector<MatchTriplet>& match_result,
                                   const cv::Size& grid_size, const std::string& output_path);

    
    // 匹配结果加载 
    static std::vector<MatchTriplet> loadMatchResultList(const std::string& video_path, const cv::Size& grid_size,
                                                        const std::string& output_path);

    // 结果可视化
    static void matchResultView(const std::string& path, const std::vector<MatchTriplet>& match_result,
                               const cv::Size& grid_size, const cv::Size& stride, 
                               const std::string& output_path, int which_video);
    
    // 拼接两个匹配结果图片并保存
    static void combinedMatchResultView(const std::string& video_path1, const std::string& video_path2,
                                      const std::vector<MatchTriplet>& match_result,
                                      const cv::Size& grid_size1, const cv::Size& grid_size2,
                                      const cv::Size& stride1, const cv::Size& stride2,
                                      const std::string& output_path);
    
    // 小网格索引计算
    static std::map<int, std::set<int>> getSmallGridIndex(const std::vector<MatchTriplet>& match_result,
                                                         const cv::Size& image_size, const cv::Size& small_grid_size,
                                                         const cv::Size& large_grid_size, bool shifting_flag, int which_video);
    
    static std::set<int> getSmallIndexInLarge(int K, int num_large_grids_per_row, int num_small_grids_per_row,
                                             int n_width, int n_height, bool shifting_flag);
    
    // 内存管理助手 - 新增优化
    static MemoryCache& getCache() {
        static MemoryCache cache;
        return cache;
    }
    
    // 批处理优化 - 新增优化
    static std::vector<cv::Mat> batchLoadNpyFiles(const std::vector<std::string>& video_paths,
                                                  const std::vector<cv::Size>& grid_sizes,
                                                  const std::string& output_path);
    
    // RANSAC筛选算法 - 新增功能
    static std::vector<MatchTriplet> ransacFilterMatchResults(const std::vector<MatchTriplet>& match_results,
                                                             const cv::Size& grid_size1, const cv::Size& grid_size2,
                                                             int num_cols1, int num_rows1, int num_cols2, int num_rows2,
                                                             bool shifting_flag, float ransac_threshold = 5.0f,
                                                             int max_iterations = 1000, float min_inlier_ratio = 0.5f);
};

// 主要的视频匹配器类 - 对应Python的cal_overlap_grid函数
class VideoMatcherEngine {
private:
    Parameters parameters_;
    std::unique_ptr<MemoryCache> cache_;  // 内存缓存 - 新增优化
    std::unique_ptr<TimeAlignmentEngine> time_alignment_engine_;  // 时间对齐引擎
    static const int ITERATE_TIMES = 4;
    
    // 预分配的内存缓冲区 - 新增优化
    std::vector<cv::Mat> motion_matrix_buffer_;
    std::vector<std::vector<MatchTriplet>> result_buffer_;
    
public:
    explicit VideoMatcherEngine(const Parameters& params = Parameters{});
    ~VideoMatcherEngine() = default;
    
    // 禁用拷贝，启用移动 - 新增优化
    VideoMatcherEngine(const VideoMatcherEngine&) = delete;
    VideoMatcherEngine& operator=(const VideoMatcherEngine&) = delete;
    VideoMatcherEngine(VideoMatcherEngine&&) = default;
    VideoMatcherEngine& operator=(VideoMatcherEngine&&) = default;
    
    // 主要处理函数 - 对应Python的cal_overlap_grid
    std::vector<std::vector<MatchTriplet>> calOverlapGrid();
    
    // 设置参数
    void setParameters(const Parameters& params) { parameters_ = params; }
    const Parameters& getParameters() const { return parameters_; }
    
    // 内存管理 - 新增优化
    void clearCache() { if (cache_) cache_->clear(); }
    void preallocateBuffers(size_t expected_size);
};

} // namespace VideoMatcher 