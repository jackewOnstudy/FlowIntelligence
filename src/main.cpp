#include "video_matcher.h"
#include <iostream>
#include <chrono>
#include <string>
#include <filesystem>

using namespace VideoMatcher;

void createOutputDirectories(const Parameters& params) {
    // 创建输出目录 - 对应Python中的目录创建逻辑
    std::filesystem::create_directories(params.motion_status_path);
    std::filesystem::create_directories(params.motion_counts_path);
    std::filesystem::create_directories(params.match_result_path);
    std::filesystem::create_directories(params.match_result_view_path);
    
    // 创建各种子目录
    std::filesystem::create_directories(params.base_output_folder + "/MatchThresholdsOtsu");
}

void printUsage(const char* program_name) {
    std::cout << "视频Patch匹配系统\n"
              << "使用方法: " << program_name << " [选项]\n"
              << "选项:\n"
              << "  --video1 <name>             第一个视频名称 (默认: T0A.mp4)\n"
              << "  --video2 <name>             第二个视频名称 (默认: B201A.mp4)\n"
              << "  --dataset-path <path>       数据集路径\n"
              << "  --output-path <path>        输出目录路径\n"
              << "  --grid-size <size>          初始网格大小 (默认: 8x8)\n"
              << "  --stride <size>             步长 (默认: 8x8)\n"
              << "  --motion-threshold <val>    运动阈值 (默认: 100,200,400,800)\n"
              << "  --segment-length <len>      分段长度 (默认: 10)\n"
              << "  --max-mismatches <count>    最大不匹配段数 (默认: 3)\n"
              << "  --propagate-step <step>     传播步长 (默认: 1)\n"
              << "  --max-frames <count>        最大处理帧数 (默认: 1000)\n"
              << "  --use-otsu-t1               在帧差分时使用Otsu阈值\n"
              << "  --use-otsu-t2               在运动状态判断时使用Otsu阈值\n"
              << "  --global-otsu               使用全局Otsu阈值（而非网格内Otsu）\n"
              << "  --enable-time-alignment     启用时间对齐补偿\n"
              << "  --max-time-offset <frames>  最大时间偏移搜索范围 (默认: 30)\n"
              << "  --time-align-threshold <val> 时间对齐相似性阈值 (默认: 0.6)\n"
              << "  -h, --help                  显示此帮助信息\n";
}

Parameters parseArguments(int argc, char* argv[]) {
    Parameters params;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "--video1" && i + 1 < argc) {
            params.video_name1 = argv[++i];
        } else if (arg == "--video2" && i + 1 < argc) {
            params.video_name2 = argv[++i];
        } else if (arg == "--dataset-path" && i + 1 < argc) {
            params.dataset_path = argv[++i];
        } else if (arg == "--output-path" && i + 1 < argc) {
            params.base_output_folder = argv[++i];
            // 更新相关路径
            params.motion_status_path = params.base_output_folder + "/MotionStatus";
            params.motion_counts_path = params.base_output_folder + "/MotionCounts";
            params.match_result_path = params.base_output_folder + "/MatchResult/List";
            params.match_result_view_path = params.base_output_folder + "/MatchResult/Pictures";
        } else if (arg == "--grid-size" && i + 1 < argc) {
            int size = std::stoi(argv[++i]);
            params.grid_size = cv::Size(size, size);
            params.grid_size2 = cv::Size(size, size);
        } else if (arg == "--stride" && i + 1 < argc) {
            int stride = std::stoi(argv[++i]);
            params.stride = cv::Size(stride, stride);
            params.stride2 = cv::Size(stride, stride);
        } else if (arg == "--segment-length" && i + 1 < argc) {
            params.segment_length = std::stoi(argv[++i]);
        } else if (arg == "--max-mismatches" && i + 1 < argc) {
            params.max_mismatches = std::stoi(argv[++i]);
        } else if (arg == "--propagate-step" && i + 1 < argc) {
            params.propagate_step = std::stoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            params.max_frames = std::stoi(argv[++i]);
        } else if (arg == "--use-otsu-t1") {
            params.use_otsu_t1 = true;
        } else if (arg == "--use-otsu-t2") {
            params.use_otsu_t2 = true;
        } else if (arg == "--global-otsu") {
            params.is_global_otsu = true;
        } else if (arg == "--enable-time-alignment") {
            params.enable_time_alignment = true;
        } else if (arg == "--max-time-offset" && i + 1 < argc) {
            params.max_time_offset = std::stoi(argv[++i]);
        } else if (arg == "--time-align-threshold" && i + 1 < argc) {
            params.time_alignment_similarity_threshold = std::stof(argv[++i]);
        }
    }
    
    return params;
}

int main(int argc, char* argv[]) {
    std::cout << "视频Patch匹配系统启动\n";
    
    try {
        // 解析参数
        Parameters params = parseArguments(argc, argv);
        
        // 创建输出目录
        createOutputDirectories(params);
        
        // 打印配置信息
        std::cout << "配置参数:\n"
                  << "  视频1: " << params.video_name1 << "\n"
                  << "  视频2: " << params.video_name2 << "\n"
                  << "  数据集路径: " << params.dataset_path << "\n"
                  << "  输出路径: " << params.base_output_folder << "\n"
                  << "  网格大小: " << params.grid_size.width << "x" << params.grid_size.height << "\n"
                  << "  步长: " << params.stride.width << "x" << params.stride.height << "\n"
                  << "  分段长度: " << params.segment_length << "\n"
                  << "  最大不匹配段数: " << params.max_mismatches << "\n"
                  << "  传播步长: " << params.propagate_step << "\n"
                  << "  最大处理帧数: " << params.max_frames << "\n"
                  << "  使用Otsu T1阈值: " << (params.use_otsu_t1 ? "是" : "否") << "\n"
                  << "  使用Otsu T2阈值: " << (params.use_otsu_t2 ? "是" : "否") << "\n"
                  << "  全局Otsu模式: " << (params.is_global_otsu ? "是" : "否") << "\n"
                  << "  启用时间对齐: " << (params.enable_time_alignment ? "是" : "否") << "\n";
        
        if (params.enable_time_alignment) {
            std::cout << "  最大时间偏移: " << params.max_time_offset << " 帧\n"
                      << "  相似性阈值: " << params.time_alignment_similarity_threshold << "\n";
        }
        std::cout << "\n";
        
        // 创建视频匹配器
        VideoMatcherEngine matcher(params);
        
        // 执行主要处理流程 - 对应Python的cal_overlap_grid函数
        std::cout << "开始执行视频匹配流程...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto all_match_results = matcher.calOverlapGrid();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n匹配流程完成!\n";
        std::cout << "总处理时间: " << duration.count() << " 毫秒\n";
        
        // 打印结果统计
        std::cout << "\n匹配结果统计:\n";
        for (size_t i = 0; i < all_match_results.size(); ++i) {
            std::cout << "  层级 " << i << ": " << all_match_results[i].size() << " 个匹配\n";
        }
        
        std::cout << "\n所有结果已保存到输出目录: " << params.base_output_folder << "\n";
        std::cout << "程序执行完成!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "未知错误\n";
        return 1;
    }
    
    return 0;
}