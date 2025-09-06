# FlowIntelligence CMake配置文件
# 此文件由CMake自动生成，请勿手动编辑



# 包含目标文件
include("${CMAKE_CURRENT_LIST_DIR}/FlowIntelligenceTargets.cmake")

# 检查兼容性
check_required_components(FlowIntelligence)

# 设置包版本
set(FlowIntelligence_VERSION "2.0.0")
set(FlowIntelligence_VERSION_MAJOR 2)
set(FlowIntelligence_VERSION_MINOR 0)
set(FlowIntelligence_VERSION_PATCH 0)

# 设置包含目录
set(FlowIntelligence_INCLUDE_DIRS "/usr/local/include/FlowIntelligence")

# 设置库目录
set(FlowIntelligence_LIBRARY_DIRS "/usr/local/lib")

# 设置链接库
set(FlowIntelligence_LIBRARIES FlowIntelligence)

# 设置编译选项
set(FlowIntelligence_CXX_FLAGS " -mavx2")

# 设置依赖项
set(FlowIntelligence_DEPENDENCIES OpenCV)

# 输出包信息
message(STATUS "FlowIntelligence 2.0.0 found")
message(STATUS "  Include directories: ${FlowIntelligence_INCLUDE_DIRS}")
message(STATUS "  Library directories: ${FlowIntelligence_LIBRARY_DIRS}")
message(STATUS "  Libraries: ${FlowIntelligence_LIBRARIES}")
