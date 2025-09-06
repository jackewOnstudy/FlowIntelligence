import cv2
import os
import shutil
from tqdm import tqdm


def save_frames_per_interval(video_path, file_name, interval, output_path):
    video_file = os.path.join(video_path, file_name)
    video = cv2.VideoCapture(video_file)

    if not video.isOpened():
        print("❌ Error opening video file:", video_file)
        return

    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = video_length // interval

    # 去掉扩展名，保留原始视频名字
    base_name, _ = os.path.splitext(file_name)

    # 输出文件夹：output_path/base_name/
    file_dir = os.path.join(output_path, base_name)
    os.makedirs(file_dir, exist_ok=True)

    # 检查并复制对应的透视矩阵npy文件
    npy_file_name = f"{base_name}_perspective_matrix.npy"
    npy_source_path = os.path.join(video_path, npy_file_name)
    
    if os.path.exists(npy_source_path):
        npy_dest_path = os.path.join(file_dir, npy_file_name)
        try:
            shutil.copy2(npy_source_path, npy_dest_path)
            print(f"📋 Copied perspective matrix: {npy_file_name}")
        except Exception as e:
            print(f"⚠️ Error copying {npy_file_name}: {e}")
    else:
        print(f"ℹ️ No perspective matrix file found for {file_name}")

    print(f"📹 Processing video: {file_name} | Total frames: {video_length} | Extracting {frame_count} frames")

    # 用 tqdm 显示进度条
    for i in tqdm(range(frame_count), desc=f"Extracting {file_name}", unit="frame"):
        frame_index = interval * i
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()

        if not ret:
            print(f"⚠️ Error reading frame {frame_index} in {file_name}")
            break

        # 命名规则：原视频名 + 帧号
        img_name = f"{base_name}_{frame_index}.jpg"
        save_file_path = os.path.join(file_dir, img_name)
        cv2.imwrite(save_file_path, frame)

    video.release()
    print(f"✅ Done processing {file_name}, saved in {file_dir}\n")


def process_all_videos(root_path, output_path, interval=1):
    """
    遍历 root_path 下所有文件和文件夹，找到 .mp4 文件并处理
    """
    print(f"🔍 Scanning {root_path} for .mp4 files ...")
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.lower().endswith(".mp4"):
                print("➡️ Found video:", os.path.join(dirpath, file))
                save_frames_per_interval(dirpath, file, interval, output_path)


# ================= 使用示例 =================
if __name__ == "__main__":
    root_path = r"/mnt/mDisk2/APIDIS_P/mm/"          # 输入根目录
    output_path = r"/mnt/mDisk2/APIDIS_P/img"    # 输出目录
    interval = 100                               # 每隔多少帧保存一张

    process_all_videos(root_path, output_path, interval)
