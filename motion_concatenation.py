import numpy as np
import cv2
import os
from pathlib import Path

def detect_motion_in_video(video_path, roi_top=0.3, roi_bottom=0.7):
    """
    直接從影片中偵測運動區域
    
    Args:
        video_path: 影片路徑
        roi_top: ROI 上邊界（畫面比例 0-1）
        roi_bottom: ROI 下邊界（畫面比例 0-1）
    
    Returns:
        每幀的運動強度列表
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    motion_scores = []
    prev_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # 轉換為灰度圖
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # 定義 ROI（關注中央區域的手部）
        h, w = gray.shape
        roi_y1 = int(h * roi_top)
        roi_y2 = int(h * roi_bottom)
        roi = gray[roi_y1:roi_y2, :] 
        
        if prev_gray is not None:
            # 計算光流或幀差
            prev_roi = prev_gray[roi_y1:roi_y2, :]
            diff = cv2.absdiff(roi, prev_roi)
            
            # 計算運動強度（非零像素的比例）
            motion_score = np.sum(diff > 30) / diff.size
            motion_scores.append(motion_score)
        else:
            motion_scores.append(0)
        
        prev_gray = gray
    
    cap.release()
    return frames, np.array(motion_scores)

def trim_static_frames(motion_scores, threshold=0.01, trim_start=True, trim_end=True):
    """
    移除開頭和結尾低於閾值的靜止幀
    
    Args:
        motion_scores: 運動分數數組
        threshold: 運動閾值
        trim_start: 是否修剪開頭
        trim_end: 是否修剪結尾
    
    Returns:
        (start_frame, end_frame) 應該保留的幀範圍
    """
    n_frames = len(motion_scores)
    
    # 找開頭：第一個超過閾值的幀
    start_frame = 0
    if trim_start:
        for i in range(n_frames):
            if motion_scores[i] > threshold:
                start_frame = i
                break
    
    # 找結尾：最後一個超過閾值的幀
    end_frame = n_frames
    if trim_end:
        for i in range(n_frames - 1, -1, -1):
            if motion_scores[i] > threshold:
                end_frame = i + 1  # +1 因為切片是左閉右開
                break
    
    print(f"  偵測到有效幀範圍: {start_frame} 到 {end_frame} (共 {end_frame - start_frame} 幀)")
    if trim_start:
        print(f"  移除開頭 {start_frame} 幀", end='')
    if trim_end:
        print(f", 移除結尾 {n_frames - end_frame} 幀" if trim_start else f"  移除結尾 {n_frames - end_frame} 幀")
    else:
        print()
    
    return start_frame, end_frame

def visualize_motion_scores(motion_scores, video_name, start_frame=None, end_frame=None, save_path=None):
    """
    視覺化運動分數（可選功能，用於調試）
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.plot(motion_scores, label='Motion Score')
    plt.axhline(y=0.024, color='r', linestyle='--', label='Threshold (0.024)')
    
    # 標示保留的範圍
    if start_frame is not None and end_frame is not None:
        plt.axvspan(start_frame, end_frame, alpha=0.2, color='green', label='Keep')
        if start_frame > 0:
            plt.axvspan(0, start_frame, alpha=0.2, color='red', label='Remove')
        if end_frame < len(motion_scores):
            plt.axvspan(end_frame, len(motion_scores), alpha=0.2, color='red')
    
    plt.xlabel('Frame')
    plt.ylabel('Motion Score')
    plt.title(f'Motion Analysis: {video_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"  運動分析圖已儲存至: {save_path}")
    else:
        plt.show()
    plt.close()

def concatenate_sign_videos(video_paths, output_path, blend_frames=10, 
                           motion_threshold=0.01,
                           use_auto_detect=True, visualize=False):
    """
    串接多個手語影片，自動移除開頭和結尾的靜止幀
    
    Args:
        video_paths: 影片路徑列表
        output_path: 輸出影片路徑
        blend_frames: 混合的幀數
        motion_threshold: 運動閾值
        use_auto_detect: 是否使用自動偵測
        visualize: 是否生成運動分析圖
    """
    print(f"\n開始處理 {len(video_paths)} 個影片...")
    print(f"設定: 混合 {blend_frames} 幀, 自動偵測: {use_auto_detect}, 運動閾值: {motion_threshold}")
    
    # 載入所有影片並分析運動
    all_frames = []
    all_ranges = []  # 儲存每個影片的有效範圍
    
    for idx, video_path in enumerate(video_paths):
        print(f"\n[{idx+1}/{len(video_paths)}] 載入並分析影片: {Path(video_path).name}")
        
        if not os.path.exists(video_path):
            print(f"  錯誤: 影片不存在!")
            continue
        
        # 分析影片運動
        frames, motion_scores = detect_motion_in_video(video_path)
        print(f"  載入了 {len(frames)} 幀")
        print(f"  平均運動強度: {np.mean(motion_scores):.4f}")
        print(f"  最大運動強度: {np.max(motion_scores):.4f}")
        
        # 判斷是否需要修剪開頭和結尾
        is_first = (idx == 0)
        is_last = (idx == len(video_paths) - 1)
        
        # 找到有效幀範圍
        if use_auto_detect:
            start_frame, end_frame = trim_static_frames(
                motion_scores, 
                motion_threshold,
                trim_start=(not is_first),  # 第一個影片不修剪開頭
                trim_end=(not is_last)      # 最後一個影片不修剪結尾
            )
        else:
            start_frame = 0 if is_first else 20
            end_frame = len(frames) if is_last else len(frames) - 20
        
        # 視覺化運動分數（可選）
        if visualize:
            video_name = Path(video_path).stem
            viz_path = f"/home/paohan/GUAVA/outputs/app/motion_analysis_{video_name}.png"
            visualize_motion_scores(motion_scores, video_name, start_frame, end_frame, viz_path)
        
        all_frames.append(frames)
        all_ranges.append((start_frame, end_frame))
    
    if len(all_frames) == 0:
        print("\n錯誤: 沒有成功載入任何影片!")
        return None, None
    
    # 串接影片
    concatenated_frames = []
    
    for i in range(len(all_frames)):
        current_frames = all_frames[i]
        start_frame, end_frame = all_ranges[i]
        
        if len(current_frames) == 0:
            continue
        
        print(f"\n處理第 {i+1} 個影片:")
        print(f"  使用幀範圍: {start_frame} 到 {end_frame}")
        
        # 提取有效幀
        valid_frames = current_frames[start_frame:end_frame]
        
        if i == 0:
            # 第一個影片：直接添加
            concatenated_frames.extend(valid_frames)
            print(f"  添加 {len(valid_frames)} 幀")
        else:
            # 後續影片：添加過渡效果
            if len(concatenated_frames) > 0 and len(valid_frames) > 0:
                prev_frame = concatenated_frames[-1]
                next_frame = valid_frames[0]
                
                print(f"  添加 {blend_frames} 幀過渡")
                for j in range(blend_frames):
                    alpha = (j + 1) / (blend_frames + 1)
                    blended_frame = cv2.addWeighted(
                        prev_frame, 1 - alpha,
                        next_frame, alpha, 0
                    )
                    concatenated_frames.append(blended_frame)
                
                # 添加剩餘幀
                concatenated_frames.extend(valid_frames)
                print(f"  添加 {len(valid_frames)} 幀")
    
    # 寫出影片
    if len(concatenated_frames) > 0:
        height, width = concatenated_frames[0].shape[:2]
        fps = 30
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in concatenated_frames:
            out.write(frame)
        
        out.release()
        print(f"\n✓ 成功!")
        print(f"✓ 總共 {len(concatenated_frames)} 幀 ({len(concatenated_frames)/fps:.2f} 秒)")
        print(f"✓ 已儲存至: {output_path}")
    else:
        print("\n錯誤: 沒有幀可以寫出!")
    
    return None, concatenated_frames

# 使用範例
if __name__ == "__main__":
    video_paths = [
        # "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_你/Gemini_Generated_Image_kzne4skzne4skzne_你_video.mp4",
        # "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_幫我/Gemini_Generated_Image_kzne4skzne4skzne_幫我_video.mp4",
        # "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_找/Gemini_Generated_Image_kzne4skzne4skzne_找_video.mp4",
        # "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_工作/Gemini_Generated_Image_kzne4skzne4skzne_工作_video.mp4",
        # "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_可以/Gemini_Generated_Image_kzne4skzne4skzne_可以_video.mp4"
    
        "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_大家/Gemini_Generated_Image_kzne4skzne4skzne_大家_video.mp4",
        "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_上課/Gemini_Generated_Image_kzne4skzne4skzne_上課_video.mp4" ,
        "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_準時/Gemini_Generated_Image_kzne4skzne4skzne_準時_video.mp4",
        "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/Gemini_Generated_Image_kzne4skzne4skzne_要/Gemini_Generated_Image_kzne4skzne4skzne_要_video.mp4"   
    ]
    
    output_path = "/home/paohan/GUAVA/outputs/app/concatenated_video_class_on_time.mp4"
    
    # 使用簡化的自動偵測
    concatenate_sign_videos(
        video_paths, 
        output_path, 
        blend_frames=15,
        motion_threshold=0.024,  # 低於此值視為靜止
        use_auto_detect=True,
        visualize=True  # 生成運動分析圖，會標示保留/移除的區域
    )