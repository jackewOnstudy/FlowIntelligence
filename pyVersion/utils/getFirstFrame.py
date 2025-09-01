import cv2

def get_first_frame(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return False
        
    # 读取第一帧
    ret, frame = cap.read()
    
    # 检查是否成功读取帧
    if not ret:
        print(f"错误: 无法读取视频帧 {video_path}")
        cap.release()
        return False
        
    # 构造保存路径
    import os
    # video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # save_dir = os.path.join(video_dir, "first_frame")
    save_dir = "/home/jackew/Project/FlowIntelligence/first_frame"
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存帧
    save_path = os.path.join(save_dir, f"{video_name}.jpg")
    cv2.imwrite(save_path, frame)
    
    # 释放视频对象
    cap.release()
    
    print(f"已保存第一帧到: {save_path}")
    return True

get_first_frame("/home/jackew/Project/FlowIntelligence/OTCBVS5L.mp4")
get_first_frame("/home/jackew/Project/FlowIntelligence/OTCBVS5R.mp4")