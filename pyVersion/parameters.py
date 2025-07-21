# 目前，distance_measure 可选 euclidean, dtw, jaccard, absolute, levenshtein, hamming and so on.
def generate_params(video_name1, video_name2, video_size=(1920, 1080), grid_size=(8, 8), **kwargs):
    return {
        'video_name1': video_name1,
        'video_name2': video_name2,
        'video_size1': video_size,
        'video_size2': video_size,
        'grid_size': grid_size,
        'stride': kwargs.get('stride', (8, 8)),
        'motion_threshold': kwargs.get('motion_threshold', [20, 80, 160, 400]),
        'distance_metric': kwargs.get('distance_metric', 'logic_and'),
        'GaussianBlurKernel': kwargs.get('GaussianBlurKernel', (11, 11)),
        'Binary_threshold': kwargs.get('Binary_threshold', 4),
        'stage_length': kwargs.get('stage_length', 9000),
        'segment_length': kwargs.get('segment_length', 300),
        'max_mismatches': kwargs.get('max_mismatches', 1),                             # 分段匹配，最大允许的不匹配次数
        'select_grid_factor': kwargs.get('select_grid_factor', 20),                    # 过滤运动较少的网格，运动阈值为序列总长度除以该参数
        'mismatch_distance_factor': kwargs.get('mismatch_distance_factor', 4),         # 统计所有网格第一段的距离，取这些距离的 mismatch_distance_factor 位数，作为距离过大不匹配的阈值
        'propagate_step': kwargs.get('propagate_step', 1),
    }

param_dict = {
     "param_SH7": {
        'video_name1': "SH7L.mp4",
        'video_name2': "SH7R.mp4",
        'video_size1': (1920, 1080),
        'video_size2': (1920, 1080),
        'grid_size': (8, 8),
        'grid_size2': (8, 8),
        'stride': (8, 8),
        'stride2': (8, 8),
        'motion_threshold1':[24, 80, 160, 400],
        'motion_threshold2':[24, 80, 160, 400],
        'distance_metric': 'logic_and',
        'GaussianBlurKernel' : (11, 11),
        'Binary_threshold' : 6,
        'stage_length' : 9000,
        'segment_length' : 100,
        'max_mismatches' : 1,
        'select_grid_factor' : 20,
        'mismatch_distance_factor' : 4,
        'propagate_step' : 1,
        #6 20
    },
     "param_SH8": {
        'video_name1': "SH8L.mp4",
        'video_name2': "SH8R.mp4",
        'video_size1': (1920, 1080),
        'video_size2': (1920, 1080),
        'grid_size': (8, 8),
        'grid_size2': (8, 8),
        'stride': (8, 8),
        'stride2': (8, 8),
        'motion_threshold1':[24, 80, 160, 400],
        'motion_threshold2':[24, 80, 160, 400],
        'distance_metric': 'logic_and',
        'GaussianBlurKernel' : (11, 11),
        'Binary_threshold' : 6,
        'stage_length' : 9000,
        'segment_length' : 100,
        'max_mismatches' : 1,
        'select_grid_factor' : 20,
        'mismatch_distance_factor' : 4,
        'propagate_step' : 1,
        #6 20
    },
    "param_T10": {
        'video_name1': "T10L.mp4",
        'video_name2': "T10R.mp4",
        'video_size1': (1920, 1080),
        'video_size2': (1920, 1080),
        'grid_size': (8, 8),
        'grid_size2': (8, 8),
        'stride': (8, 8),
        'stride2': (8, 8),
        'motion_threshold1':[24, 80, 160, 400],
        'motion_threshold2':[24, 80, 160, 400],
        'distance_metric': 'logic_and',
        'GaussianBlurKernel' : (11, 11),
        'Binary_threshold' : 6,
        'stage_length' : 9000,
        'segment_length' : 100,
        'max_mismatches' : 1,
        'select_grid_factor' : 50,
        'mismatch_distance_factor' : 4,
        'propagate_step' : 1,
        #6 20
    },
}


