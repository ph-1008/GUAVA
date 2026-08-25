import numpy as np
import os
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

def find_tracking_dirs(base_dir):
    """尋找包含追蹤數據的目錄"""
    base_path = Path(base_dir)
    tracking_dirs = []
    
    for item in base_path.iterdir():
        if item.is_dir() and (item / "videos_info.json").exists():
            tracking_dirs.append(item)
    
    return sorted(tracking_dirs)

def load_tracking_data_from_dir(tracking_dir):
    """從 EHM-Tracker 輸出目錄載入追蹤數據"""
    tracking_dir = Path(tracking_dir)
    
    possible_files = [
        'optim_tracking_ehm.pkl',
        'tracking_results.npz',
        'tracking_params.pkl'
    ]
    
    for filename in possible_files:
        filepath = tracking_dir / filename
        if filepath.exists():
            print(f"  找到數據檔案: {filename}")
            if filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif filename.endswith('.npz'):
                return np.load(filepath, allow_pickle=True)
    
    raise FileNotFoundError(f"在 {tracking_dir} 中找不到追蹤數據檔案")

def extract_hand_positions_from_frames(tracking_data):
    """從逐幀數據中提取手部位置資訊"""
    frame_keys = sorted([k for k in tracking_data.keys() if k.startswith('frame_')])
    
    if not frame_keys:
        raise ValueError("找不到任何幀數據")
    
    print(f"  找到 {len(frame_keys)} 幀數據")
    
    left_hand_positions = []
    right_hand_positions = []
    body_positions = []
    
    for frame_key in frame_keys:
        frame_data = tracking_data[frame_key]
        
        if 'dwpose_rlt' in frame_data:
            dwpose = frame_data['dwpose_rlt']
            
            if isinstance(dwpose, dict) and 'keypoints' in dwpose:
                keypoints = dwpose['keypoints']
                
                if isinstance(keypoints, np.ndarray) and len(keypoints) >= 60:
                    left_wrist = keypoints[7]
                    right_wrist = keypoints[4]
                    neck = keypoints[1]
                    
                    left_hand_positions.append(left_wrist)
                    right_hand_positions.append(right_wrist)
                    body_positions.append(neck)
                    continue
        
        left_hand_positions.append(np.zeros(2))
        right_hand_positions.append(np.zeros(2))
        body_positions.append(np.zeros(2))
    
    left_hand_pos = np.array(left_hand_positions)
    right_hand_pos = np.array(right_hand_positions)
    body_pos = np.array(body_positions)
    
    left_variance = np.var(left_hand_pos, axis=0).sum()
    right_variance = np.var(right_hand_pos, axis=0).sum()
    print(f"  左手位置變異數: {left_variance:.2f}")
    print(f"  右手位置變異數: {right_variance:.2f}")
    
    if left_variance > 0 or right_variance > 0:
        print(f"  左手位置範圍: X=[{left_hand_pos[:,0].min():.1f}, {left_hand_pos[:,0].max():.1f}], Y=[{left_hand_pos[:,1].min():.1f}, {left_hand_pos[:,1].max():.1f}]")
        print(f"  右手位置範圍: X=[{right_hand_pos[:,0].min():.1f}, {right_hand_pos[:,0].max():.1f}], Y=[{right_hand_pos[:,1].min():.1f}, {right_hand_pos[:,1].max():.1f}]")
    
    return left_hand_pos, right_hand_pos, body_pos

def detect_sign_segment_by_peaks(tracking_data, window_size=5, min_peak_distance=10, offset_frames=3):
    """
    基於速度峰值的手語片段偵測
    
    策略：
    1. 計算手部速度
    2. 找到兩個主要的速度峰值（開始動作和結束動作）
    3. 擷取兩個峰值之間的片段，並往內縮減 offset_frames
    
    Args:
        window_size: 平滑窗口大小
        min_peak_distance: 峰值之間的最小距離（幀數）
        offset_frames: 從峰值往內縮減的幀數（第一個峰值+offset，第二個峰值-offset）
    """
    if isinstance(tracking_data, dict) and any(k.startswith('frame_') for k in tracking_data.keys()):
        left_hand_pos, right_hand_pos, body_pos = extract_hand_positions_from_frames(tracking_data)
        num_frames = len(left_hand_pos)
        
        # 計算速度（使用雙手總速度）
        velocities = []
        for i in range(num_frames - 1):
            left_vel = np.linalg.norm(left_hand_pos[i+1] - left_hand_pos[i])
            right_vel = np.linalg.norm(right_hand_pos[i+1] - right_hand_pos[i])
            velocities.append(left_vel + right_vel)
        
        velocities = np.array(velocities)
        
        # 平滑速度曲線
        if len(velocities) >= window_size:
            smoothed_velocity = savgol_filter(velocities, 
                                             window_length=min(window_size*2+1, len(velocities)//2*2+1),
                                             polyorder=2)
        else:
            smoothed_velocity = velocities
        
    else:
        raise ValueError("不支援的數據格式")
    
    # 檢查數據
    if np.all(velocities == 0):
        print(f"  ⚠️  警告：所有速度都是 0")
        return 0, num_frames - 1, velocities, np.array([])
    
    # 統計資訊
    velocity_mean = np.mean(smoothed_velocity)
    velocity_std = np.std(smoothed_velocity)
    velocity_max = np.max(smoothed_velocity)
    
    print(f"\n  📊 速度統計:")
    print(f"    平均={velocity_mean:.2f}, 標準差={velocity_std:.2f}")
    print(f"    最小={np.min(smoothed_velocity):.2f}, 最大={velocity_max:.2f}")
    
    # === 找峰值 ===
    peak_height_threshold = velocity_mean + velocity_std * 0.3
    
    peaks, properties = find_peaks(smoothed_velocity, 
                                   height=peak_height_threshold,
                                   distance=min_peak_distance,
                                   prominence=velocity_std * 0.5)
    
    print(f"\n  🔍 峰值偵測:")
    print(f"    峰值高度閾值: {peak_height_threshold:.2f}")
    print(f"    找到 {len(peaks)} 個峰值")
    
    if len(peaks) >= 2:
        # 取前兩個最高的峰值
        peak_heights = smoothed_velocity[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # 從高到低排序
        top_two_peaks = peaks[sorted_indices[:2]]
        top_two_peaks = np.sort(top_two_peaks)  # 按時間順序排列
        
        start_peak = top_two_peaks[0]
        end_peak = top_two_peaks[1]
        
        print(f"    兩個主要峰值位置: {start_peak} 和 {end_peak}")
        print(f"    峰值高度: {smoothed_velocity[start_peak]:.2f} 和 {smoothed_velocity[end_peak]:.2f}")
        
        # 擷取兩個峰值之間的片段，並往內縮減
        start_frame = start_peak + offset_frames  # 第一個峰值往後
        end_frame = end_peak - offset_frames + 1  # 第二個峰值往前（+1 因為是結束幀的下一幀）
        
        print(f"    往內縮減 {offset_frames} 幀")
        print(f"    調整後: {start_frame} ({start_peak}+{offset_frames}) → {end_frame} ({end_peak}-{offset_frames})")
        
    elif len(peaks) == 1:
        # 只有一個峰值，使用峰值前後的區間
        print(f"    只找到 1 個峰值，使用峰值前後區間")
        peak_idx = peaks[0]
        
        # 找峰值前後速度下降到閾值以下的點
        threshold = velocity_mean
        
        # 向前搜尋
        start_frame = 0
        for i in range(peak_idx, -1, -1):
            if smoothed_velocity[i] < threshold:
                start_frame = i + 1
                break
        
        # 向後搜尋
        end_frame = len(smoothed_velocity) - 1
        for i in range(peak_idx, len(smoothed_velocity)):
            if smoothed_velocity[i] < threshold:
                end_frame = i
                break
        
        # 往內縮減
        start_frame = min(start_frame + offset_frames, peak_idx)
        end_frame = max(end_frame - offset_frames, peak_idx + 1)
        
    else:
        # 沒有找到峰值，使用整個高於平均值的區間
        print(f"    未找到明顯峰值，使用高速度區間")
        active_frames = np.where(smoothed_velocity > velocity_mean)[0]
        
        if len(active_frames) > 0:
            start_frame = active_frames[0] + offset_frames
            end_frame = active_frames[-1] - offset_frames + 1
        else:
            # 使用全部幀
            start_frame = offset_frames
            end_frame = len(smoothed_velocity) - offset_frames
    
    # 確保邊界有效
    start_frame = max(0, start_frame)
    end_frame = min(num_frames - 1, end_frame)
    
    # 確保至少保留一些幀
    if end_frame - start_frame < 5:
        print(f"  ⚠️  片段過短（{end_frame - start_frame} 幀），調整至最少 5 幀")
        center = (start_frame + end_frame) // 2
        start_frame = max(0, center - 2)
        end_frame = min(num_frames - 1, center + 3)
    
    print(f"\n  ✂️  最終擷取: 第 {start_frame} 幀 → 第 {end_frame} 幀")
    
    return start_frame, end_frame, smoothed_velocity, peaks if len(peaks) > 0 else np.array([])

def visualize_detection(velocity, start_frame, end_frame, peaks, output_path, video_name, offset_frames=3):
    """視覺化偵測結果（標註峰值和偏移）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    
    # 上圖：速度曲線 + 峰值標記
    ax1.plot(velocity, label='Hand Velocity (Combined)', linewidth=2.5, color='#2E86AB')
    
    # 標記峰值
    if len(peaks) >= 2:
        # 取前兩個最高峰值
        peak_heights = velocity[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_two_peaks = peaks[sorted_indices[:2]]
        top_two_peaks = np.sort(top_two_peaks)
        
        # 標記所有峰值（灰色）
        ax1.plot(peaks, velocity[peaks], 'o', color='lightgray', markersize=10, 
                label=f'All Peaks ({len(peaks)})', markeredgecolor='gray', markeredgewidth=1)
        
        # 標記主要兩個峰值（紅色X）
        ax1.plot(top_two_peaks, velocity[top_two_peaks], 'X', color='#FF6B6B', markersize=15, 
                label=f'Main 2 Peaks', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        # 標註峰值數值和偏移
        for i, peak in enumerate(top_two_peaks):
            ax1.annotate(f'Peak {i+1}\n{velocity[peak]:.1f}', 
                        xy=(peak, velocity[peak]), 
                        xytext=(0, 15), 
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
            
            # 畫出從峰值到邊界的虛線
            if i == 0:  # 第一個峰值
                ax1.plot([peak, start_frame], [velocity[peak], velocity[start_frame]], 
                        'r--', linewidth=2, alpha=0.5)
                ax1.annotate(f'+{offset_frames}', 
                            xy=((peak + start_frame)/2, velocity[peak]*0.9),
                            fontsize=9, color='red', fontweight='bold')
            else:  # 第二個峰值
                ax1.plot([peak, end_frame], [velocity[peak], velocity[end_frame]], 
                        'r--', linewidth=2, alpha=0.5)
                ax1.annotate(f'-{offset_frames}', 
                            xy=((peak + end_frame)/2, velocity[peak]*0.9),
                            fontsize=9, color='red', fontweight='bold')
    elif len(peaks) > 0:
        ax1.plot(peaks, velocity[peaks], 'X', color='#FF6B6B', markersize=15, 
                label=f'Detected Peaks ({len(peaks)})', markeredgecolor='black', markeredgewidth=1.5)
        
        for peak in peaks:
            ax1.annotate(f'{velocity[peak]:.1f}', 
                        xy=(peak, velocity[peak]), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 標記開始和結束
    ax1.axvline(x=start_frame, color='#06A77D', linestyle='--', linewidth=3, 
                label=f'Sign Start: Frame {start_frame}', zorder=5)
    ax1.axvline(x=end_frame, color='#D62828', linestyle='--', linewidth=3, 
                label=f'Sign End: Frame {end_frame}', zorder=5)
    
    # 標記擷取區間和移除區間
    ax1.axvspan(start_frame, end_frame, alpha=0.25, color='green', label='Extracted Segment')
    
    if start_frame > 0:
        ax1.axvspan(0, start_frame, alpha=0.15, color='red', label='Removed (Preparation)')
    if end_frame < len(velocity) - 1:
        ax1.axvspan(end_frame, len(velocity)-1, alpha=0.15, color='orange', label='Removed (Relaxation)')
    
    # 標記平均線
    ax1.axhline(y=np.mean(velocity), color='gray', linestyle=':', linewidth=1.5, 
                alpha=0.7, label=f'Mean: {np.mean(velocity):.1f}')
    
    ax1.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Velocity (pixels)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Sign Language Segment Detection (Peak-Based, Offset={offset_frames}) - {video_name}', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 下圖：速度分布直方圖
    ax2.hist(velocity, bins=25, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.2)
    ax2.axvline(x=np.mean(velocity), color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean: {np.mean(velocity):.1f}')
    ax2.axvline(x=np.median(velocity), color='green', linestyle='--', linewidth=2.5, 
                label=f'Median: {np.median(velocity):.1f}')
    
    # 標記峰值在直方圖上
    if len(peaks) > 0:
        for peak in peaks:
            ax2.axvline(x=velocity[peak], color='#FF6B6B', linestyle=':', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Velocity (pixels)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Velocity Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_tracking_dir(tracking_dir, output_base_dir, window_size=5, min_peak_distance=10, offset_frames=3):
    """處理單個追蹤目錄"""
    tracking_dir = Path(tracking_dir)
    dir_name = tracking_dir.name
    
    print(f"\n{'='*60}")
    print(f"處理: {dir_name}")
    print(f"{'='*60}")
    
    try:
        tracking_data = load_tracking_data_from_dir(tracking_dir)
        start_frame, end_frame, velocity, peaks = detect_sign_segment_by_peaks(
            tracking_data,
            window_size=window_size,
            min_peak_distance=min_peak_distance,
            offset_frames=offset_frames
        )
        
        frame_keys = [k for k in tracking_data.keys() if k.startswith('frame_')]
        num_frames = len(frame_keys)
        
        removed_start = start_frame
        removed_end = num_frames - end_frame - 1
        extracted = end_frame - start_frame + 1
        
        print(f"\n  📊 分析結果:")
        print(f"  ├─ 原始幀數: {num_frames}")
        print(f"  ├─ 手語片段: 第 {start_frame} 幀 → 第 {end_frame} 幀")
        print(f"  ├─ 擷取幀數: {extracted} 幀 ({extracted/num_frames*100:.1f}%)")
        print(f"  ├─ 移除開頭: {removed_start} 幀 ({removed_start/num_frames*100:.1f}%) [準備階段]")
        print(f"  └─ 移除結尾: {removed_end} 幀 ({removed_end/num_frames*100:.1f}%) [放鬆階段]")
        
        output_dir = Path(output_base_dir) / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_path = output_dir / f"{dir_name}_detection.png"
        visualize_detection(velocity, start_frame, end_frame, peaks, str(vis_path), dir_name, offset_frames)
        print(f"  ✓ 視覺化已儲存")
        
        result_json = output_dir / "segment_info.json"
        with open(result_json, 'w', encoding='utf-8') as f:
            json.dump({
                'video_name': dir_name,
                'original_frames': int(num_frames),
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'extracted_frames': int(extracted),
                'removed_start_frames': int(removed_start),
                'removed_end_frames': int(removed_end),
                'extraction_ratio': float(extracted / num_frames),
                'has_motion': bool(np.any(velocity > 0)),
                'avg_velocity': float(np.mean(velocity)),
                'max_velocity': float(np.max(velocity)),
                'num_peaks': int(len(peaks)),
                'peak_positions': [int(p) for p in peaks] if len(peaks) > 0 else [],
                'offset_frames': int(offset_frames)
            }, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 結果已儲存")
        
        return {
            'name': dir_name,
            'original_frames': num_frames,
            'start': start_frame,
            'end': end_frame,
            'duration': extracted,
            'removed_start': removed_start,
            'removed_end': removed_end,
            'num_peaks': len(peaks)
        }
        
    except Exception as e:
        print(f"\n  ✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_process_tracking_dirs(input_dir, output_dir, window_size=5, min_peak_distance=10, offset_frames=3):
    """
    批次處理追蹤目錄
    
    Args:
        window_size: 速度平滑窗口大小（較大值會使曲線更平滑）
        min_peak_distance: 峰值之間的最小距離（幀數）
        offset_frames: 從峰值往內縮減的幀數
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tracking_dirs = find_tracking_dirs(input_path)
    
    print(f"\n{'='*60}")
    print(f"手語片段擷取工具（峰值偵測版 + 偏移）")
    print(f"{'='*60}")
    print(f"找到 {len(tracking_dirs)} 個追蹤目錄")
    print(f"參數設定:")
    print(f"  ├─ 平滑窗口: {window_size} 幀")
    print(f"  ├─ 最小峰值距離: {min_peak_distance} 幀")
    print(f"  └─ 偏移量: 第一峰+{offset_frames}, 第二峰-{offset_frames}")
    
    if len(tracking_dirs) == 0:
        print(f"\n在 {input_dir} 中找不到追蹤目錄")
        return
    
    results = []
    for idx, tracking_dir in enumerate(tracking_dirs, 1):
        print(f"\n進度: [{idx}/{len(tracking_dirs)}]")
        result = process_tracking_dir(tracking_dir, output_path, window_size, min_peak_distance, offset_frames)
        if result:
            results.append(result)
    
    summary_path = output_path / 'extraction_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"手語片段擷取摘要報告（峰值偵測版 + 偏移）\n")
        f.write(f"{'='*70}\n")
        f.write(f"參數設定:\n")
        f.write(f"  - 平滑窗口: {window_size} 幀\n")
        f.write(f"  - 最小峰值距離: {min_peak_distance} 幀\n")
        f.write(f"  - 偏移量: 第一峰+{offset_frames}, 第二峰-{offset_frames}\n")
        f.write(f"成功處理: {len(results)}/{len(tracking_dirs)} 個影片\n")
        f.write(f"{'='*70}\n\n")
        
        for r in results:
            f.write(f"影片: {r['name']}\n")
            f.write(f"{'─'*70}\n")
            f.write(f"  原始幀數: {r['original_frames']}\n")
            f.write(f"  偵測到 {r['num_peaks']} 個峰值\n")
            f.write(f"  擷取範圍: 第 {r['start']} 幀 → 第 {r['end']} 幀\n")
            f.write(f"  擷取幀數: {r['duration']} 幀 ({r['duration']/r['original_frames']*100:.1f}%)\n")
            f.write(f"  移除幀數:\n")
            f.write(f"    - 開頭準備階段: {r['removed_start']} 幀\n")
            f.write(f"    - 結尾放鬆階段: {r['removed_end']} 幀\n\n")
    
    print(f"\n\n{'='*60}")
    print(f"處理完成摘要")
    print(f"{'='*60}")
    print(f"✓ 成功處理: {len(results)}/{len(tracking_dirs)} 個影片")
    if results:
        avg_extraction_ratio = np.mean([r['duration']/r['original_frames'] for r in results])
        avg_removed_start = np.mean([r['removed_start'] for r in results])
        avg_removed_end = np.mean([r['removed_end'] for r in results])
        avg_num_peaks = np.mean([r['num_peaks'] for r in results])
        print(f"✓ 平均保留率: {avg_extraction_ratio*100:.1f}%")
        print(f"✓ 平均峰值數: {avg_num_peaks:.1f}")
        print(f"✓ 平均移除:")
        print(f"  ├─ 開頭準備階段: {avg_removed_start:.1f} 幀")
        print(f"  └─ 結尾放鬆階段: {avg_removed_end:.1f} 幀")
    print(f"\n📁 輸出位置:")
    print(f"  ├─ 結果目錄: {output_dir}")
    print(f"  └─ 摘要檔案: {summary_path.name}")
    print(f"\n💡 提示：")
    print(f"  - 調整 offset_frames 可改變往內縮減的幀數（目前: {offset_frames}）")
    print(f"  - 調整 window_size 可改變平滑程度（目前: {window_size}）")
    print(f"  - 調整 min_peak_distance 可改變峰值偵測靈敏度（目前: {min_peak_distance}）")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    input_directory = "/home/paohan/GUAVA/outputs/TWTSL_tracked/"
    output_directory = "/home/paohan/GUAVA/outputs/TWTSL_tracked/extracted_sign_segments"
    
    # 參數調整指南：
    # window_size: 5 (推薦), 3 (較少平滑), 7 (較多平滑)
    # min_peak_distance: 10 (推薦), 5 (更靈敏), 15 (更保守)
    # offset_frames: 3 (推薦), 可調整 1-5
    batch_process_tracking_dirs(input_directory, output_directory, 
                               window_size=5,
                               min_peak_distance=10,
                               offset_frames=3)