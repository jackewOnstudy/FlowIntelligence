# FlowIntelligence

This is the pure version of FlowIntelligence, which includes only the original video matching feature, with all enhancements and extra dependencies removed.

## 🎯 Included Features

- **Original Video Matching**: Patch-based video alignment and matching algorithm
- **Motion Detection**: Basic motion area detection
- **Segment Matching**: Video segmentation and matching

## 📦 Dependencies

**Required Dependencies:**
- OpenCV 4.0+
- CMake 3.16+
- C++17 compiler

**Optional Dependencies:**
- OpenMP (for parallel acceleration)

## 🔨 Build Instructions

```bash
# Run the pure version build script
./build_simple.sh
```

## 🚀 Usage Instructions

After the build completes, the executable will be located in the `build/` directory:

```bash
cd build

./FlowIntelligence --video1 video1.mp4 --video2 video2.mp4 \
    --dataset-path /path/to/videos \
    --output-path /path/to/output

./FlowIntelligence --help
```

## ⚙️ Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--video1` | First video filename | T0A.mp4 |
| `--video2` | Second video filename | B201A.mp4 |
| `--dataset-path` | Directory where video files are located | - |
| `--output-path` | Output result directory | - |
| `--grid-size` | Initial grid size | 8x8 |
| `--stride` | Stride | 8x8 |
| `--segment-length` | Segment length | 10 |
| `--max-frames` | Maximum number of frames to process | 1000 |
| `--enable-time-alignment` | Enable time alignment | false |

## 📁 Output Results

The program will create the following subdirectories in the specified output directory:

- `MotionStatus/` - Motion status results
- `MotionCounts/` - Motion statistics data
- `MatchResult/List/` - Matching results list
- `MatchResult/Pictures/` - Visualized matching result images



