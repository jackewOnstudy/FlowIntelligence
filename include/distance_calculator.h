#pragma once

#include "video_matcher.h"

namespace VideoMatcher {

class DistanceCalculator {
public:
    // 逻辑与运算计算序列距离
    float calculateLogicalDistance(
        const std::vector<bool>& seq1,
        const std::vector<bool>& seq2
    );
    
    // 分组匹配距离计算
    float calculateGroupDistance(
        const std::vector<bool>& seq1,
        const std::vector<bool>& seq2,
        int group_size = 10,
        int max_mismatch = 3
    );
    
    // 计算两个patch的综合距离
    float calculatePatchDistance(const PatchInfo& patch1, const PatchInfo& patch2);
    
    // SIMD优化的逻辑与计算
    int logicalAndCountSIMD(const std::vector<bool>& seq1, const std::vector<bool>& seq2);
    
private:
    // 标准逻辑与计算
    int logicalAndCount(const std::vector<bool>& seq1, const std::vector<bool>& seq2);
};

} // namespace VideoMatcher 