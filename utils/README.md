+ analyze_match_results.py: 分析FlowIntelligence匹配结果，输入匹配结果目录
+ perspective_augmentation.py: 针对生成的多模态数据进行数据增强，输入，输入的多模态目录 `/mnt/mDisk2/newData/mm` 
generate_indentity_matrices.py：为rgb数据生成单位矩阵的透视变换文件，方便同一处理
+ calculate_relative_transforms.py: 根据原数据中的透视变换矩阵计算匹配模态之间的透视变换关系，并且将矩阵保存到匹配结果目录

+ get_img_from_video.py: 提取视频帧，方便进行基于图像的特征匹配方法

+ visualize_match_results.py: 可视化Flow Intelligence的匹配结果（热力图）