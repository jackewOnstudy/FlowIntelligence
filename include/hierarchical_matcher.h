#pragma once

#include "video_matcher.h"
#include "distance_calculator.h"

namespace VideoMatcher {

class HierarchicalMatcher {
private:
    std::shared_ptr<DistanceCalculator> distance_calculator_;
    std::vector<int> hierarchy_sizes_;
    float distance_threshold_;
    int num_threads_;
    
public:
    HierarchicalMatcher(
        std::shared_ptr<DistanceCalculator> distance_calc,
        const std::vector<int>& hierarchy_sizes,
        float distance_threshold,
        int num_threads
    );
    
    // 分层匹配主函数
    std::vector<MatchResult> hierarchicalMatch(
        const std::vector<PatchInfo>& patches1,
        const std::vector<PatchInfo>& patches2
    );
    
private:
    // 在特定层级进行匹配
    std::vector<MatchResult> matchAtLevel(
        const std::vector<PatchInfo>& patches1,
        const std::vector<PatchInfo>& patches2,
        int level_size,
        int hierarchy_level
    );
    
    // 细化匹配结果
    std::vector<MatchResult> refineMatch(
        const MatchResult& coarse_match,
        int refined_size
    );
    
    // 并行匹配处理
    std::vector<MatchResult> parallelMatch(
        const std::vector<PatchInfo>& patches1,
        const std::vector<PatchInfo>& patches2,
        int hierarchy_level
    );
    
    // 过滤重复匹配
    std::vector<MatchResult> removeDuplicates(const std::vector<MatchResult>& matches);
};

} // namespace VideoMatcher 