cmake_minimum_required(VERSION 3.16)
project(FlowIntelligence VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# 查找依赖包
find_package(OpenCV 4.0 REQUIRED)

# OpenMP支持 (可选)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    add_definitions(-D_OPENMP)
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 核心源文件 (仅原版FlowIntelligence所需)
set(CORE_SOURCES
    src/video_matcher.cpp
    src/video_matcher_utils.cpp
    src/segment_matcher.cpp
    src/time_alignment.cpp
    src/distance_calculator.cpp
)

# 核心头文件
set(CORE_HEADERS
    include/video_matcher.h
    include/parameters.h
    include/time_alignment.h
    include/distance_calculator.h
)

# 创建核心库
add_library(FlowIntelligenceCore STATIC ${CORE_SOURCES})

# 设置库属性
set_target_properties(FlowIntelligenceCore PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 链接库
target_link_libraries(FlowIntelligenceCore ${OpenCV_LIBS})

# 如果找到OpenMP，则链接
if(OpenMP_CXX_FOUND)
    target_link_libraries(FlowIntelligenceCore OpenMP::OpenMP_CXX)
endif()

# 主程序 (原版FlowIntelligence)
add_executable(FlowIntelligence src/main.cpp)
target_link_libraries(FlowIntelligence FlowIntelligenceCore)

# 安装规则
install(TARGETS FlowIntelligence FlowIntelligenceCore
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)

install(FILES ${CORE_HEADERS}
    DESTINATION include/FlowIntelligence
)

# 输出构建信息
message(STATUS "FlowIntelligence Simple Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  OpenMP support: ${OpenMP_CXX_FOUND}")
