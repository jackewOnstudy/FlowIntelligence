#include <immintrin.h>

class SIMDOptimizer {
public:
    // AVX2优化的逻辑与计算
    int logicalAndCount_AVX2(const bool* seq1, const bool* seq2, int length) {
        int count = 0;
        int simd_length = length & ~31;  // 32的倍数
        
        for (int i = 0; i < simd_length; i += 32) {
            __m256i v1 = _mm256_loadu_si256((__m256i*)&seq1[i]);
            __m256i v2 = _mm256_loadu_si256((__m256i*)&seq2[i]);
            __m256i and_result = _mm256_and_si256(v1, v2);
            
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 0));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 1));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 2));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 3));
        }
        
        // 处理剩余元素
        for (int i = simd_length; i < length; i++) {
            if (seq1[i] && seq2[i]) count++;
        }
        
        return count;
    }
};