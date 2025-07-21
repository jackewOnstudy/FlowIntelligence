#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include "parameters.h"

namespace VideoMatcher {

// 匹配三元组结构
struct MatchTriplet {
    int grid1;
    int grid2;
    float distance;
    
    MatchTriplet() : grid1(0), grid2(0), distance(0.0f) {}
    MatchTriplet(int g1, int g2, float dist) : grid1(g1), grid2(g2), distance(dist) {}
};

// Patch信息结构
struct PatchInfo {
    cv::Point2i position;             // patch位置
    std::vector<bool> motion_sequence; // 运动状态序列
    float confidence = 0.0f;          // 置信度
    int motion_count = 0;             // 运动像素计数
    
    PatchInfo() = default;
    PatchInfo(const cv::Point2i& pos) : position(pos) {}
};

// 匹配结果结构
struct MatchResult {
    PatchInfo patch1, patch2;
    float distance = 1.0f;           // 匹配距离
    float confidence = 0.0f;         // 匹配置信度
    int hierarchy_level = 0;         // 分层级别
    
    bool operator<(const MatchResult& other) const {
        return distance < other.distance;
    }
};

// 距离计算类
class DistanceCalculator {
public:
    static float logicAndDistance(const std::vector<bool>& x, const std::vector<bool>& y);
    static float logicXorDistance(const std::vector<bool>& x, const std::vector<bool>& y);
    static float segmentSimilarity(const std::vector<bool>& segment1, const std::vector<bool>& segment2, 
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
    
    // 结果处理
    static std::vector<MatchTriplet> processTriplets(const std::vector<MatchTriplet>& triplets);
    
    // 结果可视化
    static void matchResultView(const std::string& path, const std::vector<MatchTriplet>& match_result,
                               const cv::Size& grid_size, const cv::Size& stride, 
                               const std::string& output_path, int which_video);
    
    // 小网格索引计算
    static std::map<int, std::set<int>> getSmallGridIndex(const std::vector<MatchTriplet>& match_result,
                                                         const cv::Size& image_size, const cv::Size& small_grid_size,
                                                         const cv::Size& large_grid_size, bool shifting_flag, int which_video);
    
    static std::set<int> getSmallIndexInLarge(int K, int num_large_grids_per_row, int num_small_grids_per_row,
                                             int n_width, int n_height, bool shifting_flag);
};

// 主要的视频匹配器类 - 对应Python的cal_overlap_grid函数
class VideoMatcherEngine {
private:
    Parameters parameters_;
    static const int ITERATE_TIMES = 4;
    
public:
    explicit VideoMatcherEngine(const Parameters& params = Parameters{});
    
    // 主要处理函数 - 对应Python的cal_overlap_grid
    std::vector<std::vector<MatchTriplet>> calOverlapGrid();
    
    // 设置参数
    void setParameters(const Parameters& params) { parameters_ = params; }
    const Parameters& getParameters() const { return parameters_; }
};

} // namespace VideoMatcher 