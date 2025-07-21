#include "video_matcher.h"
#include <algorithm>
#include <numeric>

namespace VideoMatcher {

float DistanceCalculator::logicAndDistance(const std::vector<bool>& x, const std::vector<bool>& y) {
    // 对应Python的logic_and_distance函数
    if (x.size() != y.size()) {
        throw std::invalid_argument("两个序列长度应相同");
    }
    
    if (x.empty()) {
        return 0.0f;
    }
    
    // 计算逻辑与的结果
    int and_count = 0;
    int sum_x = 0, sum_y = 0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] && y[i]) {
            and_count++;
        }
        if (x[i]) sum_x++;
        if (y[i]) sum_y++;
    }
    
    if (sum_x == 0 && sum_y == 0) {
        return 0.0f;
    }
    
    // Python版本的计算公式：1 - np.sum(np.logical_and(x, y)) / (np.sum(x) + np.sum(y)) * 2
    return 1.0f - (2.0f * and_count) / (sum_x + sum_y);
}

float DistanceCalculator::logicXorDistance(const std::vector<bool>& x, const std::vector<bool>& y) {
    // 对应Python的logic_xonr_distance函数
    if (x.size() != y.size()) {
        throw std::invalid_argument("两个序列长度应相同");
    }
    
    if (x.empty()) {
        return 0.0f;
    }
    
    int xor_count = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) {  // 异或操作
            xor_count++;
        }
    }
    
    return static_cast<float>(xor_count);
}

float DistanceCalculator::segmentSimilarity(const std::vector<bool>& segment1, 
                                          const std::vector<bool>& segment2, 
                                          const std::string& distance_metric) {
    // 对应Python的segment_similarity函数
    if (distance_metric == "logic_and") {
        return logicAndDistance(segment1, segment2);
    } else if (distance_metric == "logic_xonr") {
        return logicXorDistance(segment1, segment2);
    } else if (distance_metric == "hamming") {
        // 简单的汉明距离实现
        if (segment1.size() != segment2.size()) {
            return 1.0f;
        }
        int diff_count = 0;
        for (size_t i = 0; i < segment1.size(); ++i) {
            if (segment1[i] != segment2[i]) {
                diff_count++;
            }
        }
        return static_cast<float>(diff_count) / segment1.size();
    } else {
        throw std::invalid_argument("Unsupported distance metric: " + distance_metric);
    }
}



} // namespace VideoMatcher