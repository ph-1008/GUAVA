#!/usr/bin/env python3
"""
完整的追踪数据串接工具（使用峰值偵測方法）
1. 使用峰值偵測來確定手語動作的核心片段
2. 在追踪参数空间生成平滑过渡
3. 输出可用于 GUAVA 渲染的完整追踪数据
"""

import numpy as np
import os
import json
import pickle
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter
import sys
sys.path.append('/home/paohan/GUAVA')
from utils.lmdb import LMDBEngine

def extract_hand_positions_from_frames(tracking_data):
    """從逐幀數據中提取手部位置資訊"""
    frame_keys = sorted([k for k in tracking_data.keys() if k.startswith('frame_')])
    
    if not frame_keys:
        raise ValueError("找不到任何幀數據")
    
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
    
    return left_hand_pos, right_hand_pos, body_pos

def detect_sign_segment_by_peaks(tracking_data, window_size=5, min_peak_distance=10, offset_frames=3):
    """
    基於速度峰值的手語片段偵測
    
    Returns:
        start_frame: 起始幀索引
        end_frame: 結束幀索引（不含）
        smoothed_velocity: 平滑後的速度曲線
        peaks: 偵測到的峰值位置
    """
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
    
    # 檢查數據
    if np.all(velocities == 0):
        print(f"    ⚠️  警告：所有速度都是 0，使用全部幀")
        return 0, num_frames, velocities, np.array([])
    
    # 統計資訊
    velocity_mean = np.mean(smoothed_velocity)
    velocity_std = np.std(smoothed_velocity)
    
    # === 找峰值 ===
    peak_height_threshold = velocity_mean + velocity_std * 0.3
    
    peaks, properties = find_peaks(smoothed_velocity, 
                                   height=peak_height_threshold,
                                   distance=min_peak_distance,
                                   prominence=velocity_std * 0.5)
    
    print(f"    速度統計: 平均={velocity_mean:.2f}, 標準差={velocity_std:.2f}")
    print(f"    找到 {len(peaks)} 個峰值")
    
    if len(peaks) >= 2:
        # 取前兩個最高的峰值
        peak_heights = smoothed_velocity[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_two_peaks = peaks[sorted_indices[:2]]
        top_two_peaks = np.sort(top_two_peaks)
        
        start_peak = top_two_peaks[0]
        end_peak = top_two_peaks[1]
        
        print(f"    兩個主要峰值位置: {start_peak} 和 {end_peak}")
        print(f"    峰值高度: {smoothed_velocity[start_peak]:.2f} 和 {smoothed_velocity[end_peak]:.2f}")
        
        # 擷取兩個峰值之間的片段，並往內縮減
        start_frame = start_peak + offset_frames
        end_frame = end_peak - offset_frames + 1
        
        print(f"    往內縮減 {offset_frames} 幀")
        print(f"    調整後: {start_frame} ({start_peak}+{offset_frames}) → {end_frame} ({end_peak}-{offset_frames})")
        
    elif len(peaks) == 1:
        print(f"    只找到 1 個峰值，使用峰值前後區間")
        peak_idx = peaks[0]
        threshold = velocity_mean
        
        start_frame = 0
        for i in range(peak_idx, -1, -1):
            if smoothed_velocity[i] < threshold:
                start_frame = i + 1
                break
        
        end_frame = len(smoothed_velocity) - 1
        for i in range(peak_idx, len(smoothed_velocity)):
            if smoothed_velocity[i] < threshold:
                end_frame = i
                break
        
        start_frame = min(start_frame + offset_frames, peak_idx)
        end_frame = max(end_frame - offset_frames, peak_idx + 1)
        
    else:
        print(f"    未找到明顯峰值，使用高速度區間")
        active_frames = np.where(smoothed_velocity > velocity_mean)[0]
        
        if len(active_frames) > 0:
            start_frame = active_frames[0] + offset_frames
            end_frame = active_frames[-1] - offset_frames + 1
        else:
            start_frame = offset_frames
            end_frame = len(smoothed_velocity) - offset_frames
    
    # 確保邊界有效
    start_frame = max(0, start_frame)
    end_frame = min(num_frames, end_frame)
    
    # 確保至少保留一些幀
    if end_frame - start_frame < 5:
        print(f"    ⚠️  片段過短（{end_frame - start_frame} 幀），調整至最少 5 幀")
        center = (start_frame + end_frame) // 2
        start_frame = max(0, center - 2)
        end_frame = min(num_frames, center + 3)
    
    return start_frame, end_frame, smoothed_velocity, peaks

def load_tracking_data(tracked_dir):
    """載入追踪数据"""
    tracking_pkl = os.path.join(tracked_dir, 'optim_tracking_ehm.pkl')
    id_share_pkl = os.path.join(tracked_dir, 'id_share_params.pkl')
    videos_info_json = os.path.join(tracked_dir, 'videos_info.json')
    
    with open(tracking_pkl, 'rb') as f:
        tracking_data = pickle.load(f)
    with open(id_share_pkl, 'rb') as f:
        id_share_params = pickle.load(f)
    with open(videos_info_json, 'r') as f:
        videos_info = json.load(f)
    
    return tracking_data, id_share_params, videos_info

def save_tracking_data(output_dir, tracking_data, id_share_params, videos_info):
    """保存追踪数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'optim_tracking_ehm.pkl'), 'wb') as f:
        pickle.dump(tracking_data, f)
    with open(os.path.join(output_dir, 'id_share_params.pkl'), 'wb') as f:
        pickle.dump(id_share_params, f)
    with open(os.path.join(output_dir, 'videos_info.json'), 'w') as f:
        json.dump(videos_info, f, indent=4)

def interpolate_tracking_params(param1, param2, alpha):
    """在两个参数之间进行插值"""
    if isinstance(param1, dict):
        result = {}
        for key in param1.keys():
            if isinstance(param1[key], dict):
                result[key] = interpolate_tracking_params(param1[key], param2[key], alpha)
            else:
                result[key] = param1[key] * (1 - alpha) + param2[key] * alpha
        return result
    else:
        return param1 * (1 - alpha) + param2 * alpha

def create_smooth_transition(tracking_data, frame_key1, frame_key2, num_transition_frames=15):
    """在两帧之间创建平滑过渡帧"""
    param1, param2 = tracking_data[frame_key1], tracking_data[frame_key2]
    transition_frames = []
    
    for i in range(1, num_transition_frames + 1):
        alpha = i / (num_transition_frames + 1)
        new_frame = {}
        
        # 插值所有参数
        for coeff_type in ['smplx_coeffs', 'flame_coeffs', 'left_mano_coeffs', 'right_mano_coeffs']:
            new_frame[coeff_type] = {}
            for key in param1[coeff_type].keys():
                if key == 'camera_RT_params':
                    new_frame[coeff_type][key] = param1[coeff_type][key].copy()
                else:
                    new_frame[coeff_type][key] = interpolate_tracking_params(
                        param1[coeff_type][key], param2[coeff_type][key], alpha
                    )
        
        # 复制其他必要信息
        new_frame['dwpose_rlt'] = param2['dwpose_rlt'].copy()
        for key in ['body_crop', 'head_crop', 'left_hand_crop', 'right_hand_crop',
                    'head_lmk_203', 'head_lmk_70', 'head_lmk_mp']:
            if key in param2:
                new_frame[key] = param2[key].copy()
        
        transition_frames.append(new_frame)
    
    return transition_frames

def concatenate_tracking_data_with_peaks(tracked_dirs, output_dir, 
                                        window_size=5,
                                        min_peak_distance=10,
                                        offset_frames=3,
                                        num_transition_frames=15,
                                        visualize=True):
    """
    使用峰值偵測方法串接追踪数据
    """
    print(f"\n{'='*70}")
    print(f"手語追蹤數據串接工具（峰值偵測版）")
    print(f"{'='*70}")
    print(f"處理 {len(tracked_dirs)} 個追蹤數據...")
    print(f"參數設定:")
    print(f"  ├─ 平滑窗口: {window_size} 幀")
    print(f"  ├─ 最小峰值距離: {min_peak_distance} 幀")
    print(f"  ├─ 峰值偏移量: {offset_frames} 幀")
    print(f"  └─ 過渡幀數: {num_transition_frames} 幀")
    
    all_tracking_data, all_ranges, all_frame_keys, all_lmdb_engines = [], [], [], []
    first_id_share_params = None
    all_velocities = []
    
    # 載入所有追踪数据並使用峰值偵測
    for idx, tracked_dir in enumerate(tracked_dirs):
        print(f"\n{'─'*70}")
        print(f"[{idx+1}/{len(tracked_dirs)}] 載入追蹤數據: {Path(tracked_dir).name}")
        print(f"{'─'*70}")
        
        if not os.path.exists(tracked_dir):
            print(f"  ✗ 錯誤: 目錄不存在!")
            continue
        
        tracking_data, id_share_params, videos_info = load_tracking_data(tracked_dir)
        
        # 載入LMDB
        lmdb_path = os.path.join(tracked_dir, 'img_lmdb')
        if not os.path.exists(lmdb_path):
            print(f"  ✗ 錯誤: 未找到 img_lmdb 目錄!")
            continue
        
        lmdb_engine = LMDBEngine(lmdb_path, write=False)
        all_lmdb_engines.append(lmdb_engine)
        
        if first_id_share_params is None:
            first_id_share_params = id_share_params
        
        video_name = list(videos_info.keys())[0]
        frame_keys = videos_info[video_name]['frames_keys']
        print(f"  載入了 {len(frame_keys)} 幀")
        
        # 使用峰值偵測確定片段範圍
        print(f"  使用峰值偵測分析手語動作...")
        
        # 判斷是否為首尾視頻
        is_first = (idx == 0)
        is_last = (idx == len(tracked_dirs) - 1)
        
        # 先執行正常的峰值偵測
        start_frame, end_frame, velocity, peaks = detect_sign_segment_by_peaks(
            tracking_data, window_size, min_peak_distance, offset_frames
        )
        
        # 根據位置調整邊界
        if is_first:
            # 第一個視頻：保留開頭（從第 0 幀開始）
            # 但第一個峰值還是要 +3，第二個峰值還是要 -3
            if len(peaks) >= 2:
                peak_heights = velocity[peaks]
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_two_peaks = peaks[sorted_indices[:2]]
                top_two_peaks = np.sort(top_two_peaks)
                
                start_peak = top_two_peaks[0]
                end_peak = top_two_peaks[1]
                
                # 第一個峰值 +3，第二個峰值 -3
                start_frame = start_peak + offset_frames
                end_frame = end_peak - offset_frames + 1
                
                # 但是從第 0 幀開始（保留開頭）
                start_frame = 0
                
                print(f"    第一個視頻：保留開頭（第 0 幀），結束於第二峰值-{offset_frames} (幀 {end_frame})")
            else:
                start_frame = 0
                
        elif is_last:
            # 最後一個視頻：保留結尾（到最後一幀）
            # 但第一個峰值還是要 +3，第二個峰值還是要 -3
            if len(peaks) >= 2:
                peak_heights = velocity[peaks]
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_two_peaks = peaks[sorted_indices[:2]]
                top_two_peaks = np.sort(top_two_peaks)
                
                start_peak = top_two_peaks[0]
                end_peak = top_two_peaks[1]
                
                # 第一個峰值 +3，第二個峰值 -3
                start_frame = start_peak + offset_frames
                end_frame = end_peak - offset_frames + 1
                
                # 但是到最後一幀（保留結尾）
                end_frame = len(frame_keys)
                
                print(f"    最後一個視頻：開始於第一峰值+{offset_frames} (幀 {start_frame})，保留結尾（第 {end_frame} 幀）")
            else:
                end_frame = len(frame_keys)
        else:
            # 中間視頻：使用正常偵測結果（已經包含 offset）
            print(f"    中間視頻：正常裁剪")
        
        print(f"  ✂️  擷取幀範圍: 第 {start_frame} 幀 → 第 {end_frame} 幀")
        print(f"  📊 擷取 {end_frame - start_frame} 幀 (原始: {len(frame_keys)} 幀)")
        print(f"  📉 移除開頭: {start_frame} 幀, 移除結尾: {len(frame_keys) - end_frame} 幀")
        
        # 可視化
        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 5))
            
            ax.plot(velocity, label='Hand Velocity (Combined)', linewidth=2.5, color='#2E86AB')
            
            # 標記峰值
            if len(peaks) >= 2:
                peak_heights = velocity[peaks]
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_two_peaks = peaks[sorted_indices[:2]]
                top_two_peaks = np.sort(top_two_peaks)
                
                ax.plot(peaks, velocity[peaks], 'o', color='lightgray', markersize=10, 
                       label=f'All Peaks ({len(peaks)})', markeredgecolor='gray', markeredgewidth=1)
                ax.plot(top_two_peaks, velocity[top_two_peaks], 'X', color='#FF6B6B', 
                       markersize=15, label=f'Main 2 Peaks', markeredgecolor='black', 
                       markeredgewidth=1.5, zorder=10)
                
                # 標註峰值和偏移
                for i, peak in enumerate(top_two_peaks):
                    offset_text = f"+{offset_frames}" if i == 0 else f"-{offset_frames}"
                    ax.annotate(f'Peak {i+1}\n{velocity[peak]:.1f}\n({offset_text})', 
                               xy=(peak, velocity[peak]), 
                               xytext=(0, 20), 
                               textcoords='offset points',
                               ha='center',
                               fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
            
            ax.axvline(x=start_frame, color='#06A77D', linestyle='--', linewidth=3, 
                      label=f'Start: Frame {start_frame}', zorder=5)
            ax.axvline(x=end_frame, color='#D62828', linestyle='--', linewidth=3, 
                      label=f'End: Frame {end_frame}', zorder=5)
            
            ax.axvspan(start_frame, end_frame, alpha=0.25, color='green', label='Extracted')
            if start_frame > 0:
                ax.axvspan(0, start_frame, alpha=0.15, color='red', label='Removed (Start)')
            if end_frame < len(frame_keys):
                ax.axvspan(end_frame, len(frame_keys), alpha=0.15, color='orange', label='Removed (End)')
            
            # 添加位置標籤
            position_text = "FIRST" if is_first else ("LAST" if is_last else "MIDDLE")
            ax.text(0.02, 0.98, position_text, transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Velocity (pixels)', fontsize=12, fontweight='bold')
            ax.set_title(f'Peak Detection: {Path(tracked_dir).name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, f'peak_analysis_{idx+1}_{Path(tracked_dir).name}.png')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 視覺化已保存: {Path(viz_path).name}")
        
        all_tracking_data.append(tracking_data)
        all_ranges.append((start_frame, end_frame))
        all_frame_keys.append(frame_keys)
        all_velocities.append(velocity)
    
    if len(all_tracking_data) == 0:
        print(f"\n✗ 錯誤: 沒有成功載入任何追踪数据!")
        return
    
    # 串接追踪数据
    print(f"\n{'='*70}")
    print(f"開始串接追踪參數...")
    print(f"{'='*70}")
    
    concatenated_tracking = {}
    concatenated_frame_keys = []
    frame_counter = 0
    
    for i in range(len(all_tracking_data)):
        tracking_data, frame_keys = all_tracking_data[i], all_frame_keys[i]
        start_frame, end_frame = all_ranges[i]
        
        print(f"\n處理第 {i+1}/{len(all_tracking_data)} 個視頻:")
        print(f"  使用幀範圍: {start_frame} → {end_frame}")
        
        valid_keys = frame_keys[start_frame:end_frame]
        
        if i == 0:
            # 第一個視頻：直接添加
            for key in valid_keys:
                new_key = f"frame_{frame_counter:06d}"
                concatenated_tracking[new_key] = tracking_data[key]
                concatenated_frame_keys.append(new_key)
                frame_counter += 1
            print(f"  ✓ 添加 {len(valid_keys)} 幀")
        else:
            # 後續視頻：添加過渡幀
            if len(concatenated_frame_keys) > 0 and len(valid_keys) > 0:
                prev_key = concatenated_frame_keys[-1]
                next_key_original = valid_keys[0]
                
                print(f"  生成 {num_transition_frames} 個平滑過渡幀...")
                
                transition_frames = create_smooth_transition(
                    {**concatenated_tracking, next_key_original: tracking_data[next_key_original]},
                    prev_key, next_key_original, num_transition_frames
                )
                
                for trans_frame in transition_frames:
                    new_key = f"frame_{frame_counter:06d}"
                    concatenated_tracking[new_key] = trans_frame
                    concatenated_frame_keys.append(new_key)
                    frame_counter += 1
                
                for key in valid_keys:
                    new_key = f"frame_{frame_counter:06d}"
                    concatenated_tracking[new_key] = tracking_data[key]
                    concatenated_frame_keys.append(new_key)
                    frame_counter += 1
                
                print(f"  ✓ 添加 {num_transition_frames} 個過渡幀 + {len(valid_keys)} 個內容幀")
    
    # 創建videos_info
    videos_info = {
        "concatenated_video": {
            "frames_num": len(concatenated_frame_keys),
            "frames_keys": concatenated_frame_keys
        }
    }
    
    # 保存結果
    print(f"\n{'='*70}")
    print(f"保存串接後的追踪数据...")
    print(f"{'='*70}")
    save_tracking_data(output_dir, concatenated_tracking, first_id_share_params, videos_info)
    print(f"  ✓ 追踪參數已保存")
    
    # 複製圖像到LMDB
    print(f"\n複製圖像數據到 LMDB...")
    output_lmdb_path = os.path.join(output_dir, 'img_lmdb')
    output_lmdb = LMDBEngine(output_lmdb_path, write=True)
    
    img_types = ['ori_image', 'body_image', 'body_mask', 'body_matting', 
                'head_image', 'left_hand_image', 'right_hand_image']
    
    frame_counter = 0
    for i in range(len(all_tracking_data)):
        lmdb_engine = all_lmdb_engines[i]
        frame_keys = all_frame_keys[i]
        start_frame, end_frame = all_ranges[i]
        valid_keys = frame_keys[start_frame:end_frame]
        
        print(f"  複製第 {i+1}/{len(all_tracking_data)} 個視頻的圖像 ({len(valid_keys)} 幀)...")
        
        if i == 0:
            # 第一個視頻：直接複製
            for old_key in tqdm(valid_keys, desc=f"    Video {i+1}"):
                new_key = f"frame_{frame_counter:06d}"
                for img_type in img_types:
                    try:
                        img_data = lmdb_engine[f'{old_key}/{img_type}']
                        output_lmdb.dump(f'{new_key}/{img_type}', payload=img_data, type='image')
                    except:
                        pass
                frame_counter += 1
        else:
            # 添加過渡幀圖像（使用下一個視頻的第一幀）
            if len(valid_keys) > 0:
                first_key = valid_keys[0]
                for _ in range(num_transition_frames):
                    new_key = f"frame_{frame_counter:06d}"
                    for img_type in img_types:
                        try:
                            img_data = lmdb_engine[f'{first_key}/{img_type}']
                            output_lmdb.dump(f'{new_key}/{img_type}', payload=img_data, type='image')
                        except:
                            pass
                    frame_counter += 1
                
                # 複製正常幀
                for old_key in tqdm(valid_keys, desc=f"    Video {i+1}"):
                    new_key = f"frame_{frame_counter:06d}"
                    for img_type in img_types:
                        try:
                            img_data = lmdb_engine[f'{old_key}/{img_type}']
                            output_lmdb.dump(f'{new_key}/{img_type}', payload=img_data, type='image')
                        except:
                            pass
                    frame_counter += 1
    
    output_lmdb.close()
    print(f"  ✓ LMDB 創建完成！")
    
    # 關閉所有輸入LMDB
    for lmdb_engine in all_lmdb_engines:
        if lmdb_engine is not None:
            lmdb_engine.close()
    
    # 生成摘要報告
    print(f"\n{'='*70}")
    print(f"串接完成摘要")
    print(f"{'='*70}")
    
    total_original_frames = sum([len(fk) for fk in all_frame_keys])
    total_extracted_frames = sum([r[1] - r[0] for r in all_ranges])
    total_transition_frames = num_transition_frames * (len(all_tracking_data) - 1)
    total_final_frames = len(concatenated_frame_keys)
    
    print(f"✓ 成功處理 {len(all_tracking_data)} 個視頻")
    print(f"✓ 原始總幀數: {total_original_frames} 幀")
    print(f"✓ 擷取內容幀: {total_extracted_frames} 幀 ({total_extracted_frames/total_original_frames*100:.1f}%)")
    print(f"✓ 生成過渡幀: {total_transition_frames} 幀")
    print(f"✓ 最終總幀數: {total_final_frames} 幀 ({total_final_frames/30:.2f} 秒 @ 30fps)")
    print(f"✓ 輸出目錄: {output_dir}")
    
    # 保存摘要
    summary_path = os.path.join(output_dir, 'concatenation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"手語追蹤數據串接摘要報告（峰值偵測版）\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"參數設定:\n")
        f.write(f"  - 平滑窗口: {window_size} 幀\n")
        f.write(f"  - 最小峰值距離: {min_peak_distance} 幀\n")
        f.write(f"  - 峰值偏移量: {offset_frames} 幀\n")
        f.write(f"  - 過渡幀數: {num_transition_frames} 幀\n\n")
        f.write(f"處理結果:\n")
        f.write(f"  - 處理視頻數: {len(all_tracking_data)}\n")
        f.write(f"  - 原始總幀數: {total_original_frames} 幀\n")
        f.write(f"  - 擷取內容幀: {total_extracted_frames} 幀 ({total_extracted_frames/total_original_frames*100:.1f}%)\n")
        f.write(f"  - 生成過渡幀: {total_transition_frames} 幀\n")
        f.write(f"  - 最終總幀數: {total_final_frames} 幀 ({total_final_frames/30:.2f} 秒)\n\n")
        f.write(f"各視頻詳情:\n")
        f.write(f"{'─'*70}\n")
        
        for i, (tracked_dir, (start, end)) in enumerate(zip(tracked_dirs, all_ranges)):
            name = Path(tracked_dir).name
            original = len(all_frame_keys[i])
            extracted = end - start
            removed_start = start
            removed_end = original - end
            position = "FIRST" if i == 0 else ("LAST" if i == len(tracked_dirs) - 1 else "MIDDLE")
            
            f.write(f"\n{i+1}. {name} [{position}]\n")
            f.write(f"   原始幀數: {original}\n")
            f.write(f"   擷取範圍: 第 {start} 幀 → 第 {end} 幀\n")
            f.write(f"   擷取幀數: {extracted} 幀 ({extracted/original*100:.1f}%)\n")
            f.write(f"   移除幀數: 開頭 {removed_start} 幀, 結尾 {removed_end} 幀\n")
    
    print(f"\n✓ 摘要報告已保存: {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"後續步驟：渲染視頻")
    print(f"{'='*70}")
    print(f"使用以下命令進行渲染:")
    print(f"\nPYTHONPATH=/home/paohan/GUAVA:$PYTHONPATH python main/test.py \\")
    print(f"  -d '0' -m assets/GUAVA -s outputs/concat_render \\")
    print(f"  --data_path {output_dir}")
    print(f"\n或使用指定形象進行 cross-reenactment:")
    print(f"\nPYTHONPATH=/home/paohan/GUAVA:$PYTHONPATH python main/test.py \\")
    print(f"  -d '0' -m assets/GUAVA -s outputs/concat_render_cross \\")
    print(f"  --data_path {output_dir} \\")
    print(f"  --source_data_path <源形象路徑> \\")
    print(f"  --skip_self_act --render_cross_act")
    print(f"{'='*70}\n")

# 使用範例
if __name__ == "__main__":
    tracked_dirs = [
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/你/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/幫我/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/找/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/工作/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/可以/",
    ]
    
    output_dir = "/home/paohan/GUAVA/outputs/concatenated_tracking_peak_based_final_2_0107_working"
    
    concatenate_tracking_data_with_peaks(
        tracked_dirs, output_dir,
        window_size=5,              # 速度平滑窗口
        min_peak_distance=10,       # 峰值最小距離
        offset_frames=3,            # 從峰值往內縮減的幀數
        num_transition_frames=15,   # 視頻間過渡幀數
        visualize=True              # 生成視覺化圖表
    )
