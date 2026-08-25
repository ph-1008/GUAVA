#!/usr/bin/env python3
"""
完整的追踪数据串接工具（使用峰值偵測方法 + 強化參數平滑）
增強版本：添加平滑效果驗證
"""

import numpy as np
import os
import json
import pickle
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
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

def calculate_smoothness_metric(param_array):
    """
    計算參數的平滑度指標（二階差分的標準差）
    值越小表示越平滑
    """
    if len(param_array) < 3:
        return 0.0
    
    # 計算二階差分（加速度）
    second_diff = np.diff(param_array, n=2, axis=0)
    # 計算所有維度的 RMS
    smoothness = np.sqrt(np.mean(second_diff ** 2))
    return smoothness

def smooth_tracking_params(param_array, sigma=1.0, method='gaussian', use_median=False):
    """
    平滑追蹤參數序列（強化版 + 驗證）
    
    Args:
        param_array: shape (num_frames, param_dim) 的參數陣列
        sigma: 平滑強度（標準差）
        method: 'gaussian' 或 'savgol'
        use_median: 是否先使用中值濾波器去除突變
    
    Returns:
        smoothed_array: 平滑後的參數陣列
        smoothness_before: 平滑前的平滑度指標
        smoothness_after: 平滑後的平滑度指標
    """
    if len(param_array) < 3:
        return param_array, 0.0, 0.0
    
    # 計算平滑前的指標
    smoothness_before = calculate_smoothness_metric(param_array)
    
    smoothed = np.zeros_like(param_array)
    original = param_array.copy()
    
    # === 第一步：中值濾波（去除突變/閃動） ===
    if use_median:
        for dim in range(param_array.shape[1]):
            # 使用 5 點中值濾波器
            smoothed[:, dim] = medfilt(param_array[:, dim], kernel_size=5)
        param_array = smoothed.copy()
    
    # === 第二步：高斯或 Savgol 平滑 ===
    if method == 'gaussian':
        for dim in range(param_array.shape[1]):
            smoothed[:, dim] = gaussian_filter1d(param_array[:, dim], sigma=sigma)
    
    elif method == 'savgol':
        window_length = min(11, len(param_array) if len(param_array) % 2 == 1 else len(param_array) - 1)
        if window_length < 3:
            return param_array, smoothness_before, smoothness_before
        
        for dim in range(param_array.shape[1]):
            try:
                smoothed[:, dim] = savgol_filter(param_array[:, dim], 
                                                 window_length=window_length, 
                                                 polyorder=2)
            except:
                smoothed[:, dim] = param_array[:, dim]
    
    # 計算平滑後的指標
    smoothness_after = calculate_smoothness_metric(smoothed)
    
    # 驗證是否真的有變平滑
    improvement = ((smoothness_before - smoothness_after) / smoothness_before * 100) if smoothness_before > 0 else 0
    
    return smoothed, smoothness_before, smoothness_after, improvement

def extract_params_to_array(tracking_data, frame_keys, param_type, param_name):
    """從追蹤數據中提取特定參數到陣列"""
    params_list = []
    
    first_key = frame_keys[0]
    if param_type not in tracking_data[first_key]:
        raise KeyError(f"{param_type} 不存在")
    if param_name not in tracking_data[first_key][param_type]:
        raise KeyError(f"{param_type}.{param_name} 不存在")
    
    for key in frame_keys:
        param = tracking_data[key][param_type][param_name]
        params_list.append(param.flatten())
    
    return np.array(params_list)

def apply_smoothed_params(tracking_data, frame_keys, param_type, param_name, smoothed_array):
    """將平滑後的參數應用回追蹤數據"""
    original_shape = tracking_data[frame_keys[0]][param_type][param_name].shape
    
    for i, key in enumerate(frame_keys):
        tracking_data[key][param_type][param_name] = smoothed_array[i].reshape(original_shape)

def smooth_all_tracking_params(tracking_data, frame_keys, 
                               body_sigma=1.5, 
                               hand_sigma=1.8, 
                               face_sigma=1.2,
                               use_median_for_hands=True):
    """
    對所有追蹤參數進行強化平滑處理（增強版：帶驗證）
    """
    print(f"    🔄 開始強化平滑追蹤參數...")
    print(f"    📊 平滑前後對比（改善率）:")
    
    smooth_configs = [
        ('smplx_coeffs', 'body_pose', body_sigma, False, '身體姿態'),
        ('smplx_coeffs', 'global_pose', body_sigma, False, '全局姿態'),
        ('smplx_coeffs', 'transl', body_sigma, False, '平移'),
        
        ('left_mano_coeffs', 'hand_pose', hand_sigma, use_median_for_hands, '左手姿態（含手指）'),
        ('left_mano_coeffs', 'global_orient', hand_sigma * 0.8, False, '左手旋轉'),
        
        ('right_mano_coeffs', 'hand_pose', hand_sigma, use_median_for_hands, '右手姿態（含手指）'),
        ('right_mano_coeffs', 'global_orient', hand_sigma * 0.8, False, '右手旋轉'),
        
        ('flame_coeffs', 'jaw_params', face_sigma, False, '下巴姿態'),
        ('flame_coeffs', 'neck_pose', face_sigma, False, '脖子姿態'),
        ('flame_coeffs', 'expression_params', face_sigma, False, '表情參數'),
    ]
    
    smoothed_count = 0
    total_improvement = 0
    
    for param_type, param_name, sigma, use_median, desc in smooth_configs:
        try:
            # 提取參數
            param_array = extract_params_to_array(tracking_data, frame_keys, param_type, param_name)
            
            # 平滑並獲取統計資訊
            smoothed_array, before, after, improvement = smooth_tracking_params(
                param_array, 
                sigma=sigma, 
                method='gaussian',
                use_median=use_median
            )
            
            # 應用回去
            apply_smoothed_params(tracking_data, frame_keys, param_type, param_name, smoothed_array)
            
            smoothed_count += 1
            total_improvement += improvement
            
            median_text = " [中值+高斯]" if use_median else ""
            # 顯示改善效果
            print(f"      ✓ {desc:20s} σ={sigma:.1f}{median_text:15s} | 改善: {improvement:6.1f}%")
            
        except KeyError:
            pass
        except Exception as e:
            print(f"      ⚠️  {desc} 平滑失敗: {e}")
    
    avg_improvement = total_improvement / smoothed_count if smoothed_count > 0 else 0
    print(f"    ✓ 成功平滑 {smoothed_count} 組參數，平均改善率: {avg_improvement:.1f}%")

def detect_sign_segment_by_peaks(tracking_data, window_size=5, min_peak_distance=10, offset_frames=3):
    """基於速度峰值的手語片段偵測"""
    left_hand_pos, right_hand_pos, body_pos = extract_hand_positions_from_frames(tracking_data)
    num_frames = len(left_hand_pos)
    
    velocities = []
    for i in range(num_frames - 1):
        left_vel = np.linalg.norm(left_hand_pos[i+1] - left_hand_pos[i])
        right_vel = np.linalg.norm(right_hand_pos[i+1] - right_hand_pos[i])
        velocities.append(left_vel + right_vel)
    
    velocities = np.array(velocities)
    
    if len(velocities) >= window_size:
        smoothed_velocity = savgol_filter(velocities, 
                                         window_length=min(window_size*2+1, len(velocities)//2*2+1),
                                         polyorder=2)
    else:
        smoothed_velocity = velocities
    
    if np.all(velocities == 0):
        print(f"    ⚠️  警告：所有速度都是 0，使用全部幀")
        return 0, num_frames, velocities, np.array([])
    
    velocity_mean = np.mean(smoothed_velocity)
    velocity_std = np.std(smoothed_velocity)
    
    peak_height_threshold = velocity_mean + velocity_std * 0.3
    
    peaks, properties = find_peaks(smoothed_velocity, 
                                   height=peak_height_threshold,
                                   distance=min_peak_distance,
                                   prominence=velocity_std * 0.5)
    
    print(f"    速度統計: 平均={velocity_mean:.2f}, 標準差={velocity_std:.2f}")
    print(f"    找到 {len(peaks)} 個峰值")
    
    if len(peaks) >= 2:
        peak_heights = smoothed_velocity[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_two_peaks = peaks[sorted_indices[:2]]
        top_two_peaks = np.sort(top_two_peaks)
        
        start_peak = top_two_peaks[0]
        end_peak = top_two_peaks[1]
        
        print(f"    兩個主要峰值位置: {start_peak} 和 {end_peak}")
        print(f"    峰值高度: {smoothed_velocity[start_peak]:.2f} 和 {smoothed_velocity[end_peak]:.2f}")
        
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
    
    start_frame = max(0, start_frame)
    end_frame = min(num_frames, end_frame)
    
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
    """保存追踪数据（強制寫入）"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 確保完全寫入
    tracking_pkl_path = os.path.join(output_dir, 'optim_tracking_ehm.pkl')
    with open(tracking_pkl_path, 'wb') as f:
        pickle.dump(tracking_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    id_share_pkl_path = os.path.join(output_dir, 'id_share_params.pkl')
    with open(id_share_pkl_path, 'wb') as f:
        pickle.dump(id_share_params, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    videos_info_json_path = os.path.join(output_dir, 'videos_info.json')
    with open(videos_info_json_path, 'w') as f:
        json.dump(videos_info, f, indent=4)
    
    # 驗證檔案大小
    tracking_size = os.path.getsize(tracking_pkl_path) / (1024 * 1024)  # MB
    print(f"  ✓ 追踪參數已保存 ({tracking_size:.2f} MB)")
    
    # 驗證可讀取
    with open(tracking_pkl_path, 'rb') as f:
        _ = pickle.load(f)
    print(f"  ✓ 檔案驗證通過")

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
        
        for coeff_type in ['smplx_coeffs', 'flame_coeffs', 'left_mano_coeffs', 'right_mano_coeffs']:
            new_frame[coeff_type] = {}
            for key in param1[coeff_type].keys():
                if key == 'camera_RT_params':
                    new_frame[coeff_type][key] = param1[coeff_type][key].copy()
                else:
                    new_frame[coeff_type][key] = interpolate_tracking_params(
                        param1[coeff_type][key], param2[coeff_type][key], alpha
                    )
        
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
                                        smooth_params=True,
                                        body_sigma=2.5,
                                        hand_sigma=3.0,
                                        face_sigma=2.0,
                                        global_smooth_sigma=1.5,
                                        use_median_for_hands=True,
                                        visualize=True):
    """使用峰值偵測方法串接追踪数据（增強驗證版）"""
    print(f"\n{'='*70}")
    print(f"手語追蹤數據串接工具（峰值偵測版 + 強化參數平滑 + 驗證）")
    print(f"{'='*70}")
    print(f"處理 {len(tracked_dirs)} 個追蹤數據...")
    print(f"參數設定:")
    print(f"  ├─ 平滑窗口: {window_size} 幀")
    print(f"  ├─ 最小峰值距離: {min_peak_distance} 幀")
    print(f"  ├─ 峰值偏移量: {offset_frames} 幀")
    print(f"  ├─ 過渡幀數: {num_transition_frames} 幀")
    print(f"  ├─ 參數平滑: {'啟用（強化版 + 驗證）' if smooth_params else '停用'}")
    if smooth_params:
        print(f"  │  ├─ 身體平滑: σ={body_sigma}")
        print(f"  │  ├─ 手部平滑: σ={hand_sigma} {'+ 中值濾波' if use_median_for_hands else ''}")
        print(f"  │  ├─ 臉部平滑: σ={face_sigma}")
        print(f"  │  └─ 全局平滑: σ={global_smooth_sigma}")
    print(f"  └─ 視覺化: {'啟用' if visualize else '停用'}")
    
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
        
        # === 對單個視頻的參數進行強化平滑 ===
        if smooth_params:
            smooth_all_tracking_params(tracking_data, frame_keys, 
                                      body_sigma=body_sigma,
                                      hand_sigma=hand_sigma,
                                      face_sigma=face_sigma,
                                      use_median_for_hands=use_median_for_hands)
        
        print(f"  使用峰值偵測分析手語動作...")
        
        is_first = (idx == 0)
        is_last = (idx == len(tracked_dirs) - 1)
        
        start_frame, end_frame, velocity, peaks = detect_sign_segment_by_peaks(
            tracking_data, window_size, min_peak_distance, offset_frames
        )
        
        if is_first:
            if len(peaks) >= 2:
                peak_heights = velocity[peaks]
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_two_peaks = peaks[sorted_indices[:2]]
                top_two_peaks = np.sort(top_two_peaks)
                
                start_peak = top_two_peaks[0]
                end_peak = top_two_peaks[1]
                
                start_frame = start_peak + offset_frames
                end_frame = end_peak - offset_frames + 1
                start_frame = 0
                
                print(f"    第一個視頻：保留開頭（第 0 幀），結束於第二峰值-{offset_frames} (幀 {end_frame})")
            else:
                start_frame = 0
                
        elif is_last:
            if len(peaks) >= 2:
                peak_heights = velocity[peaks]
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_two_peaks = peaks[sorted_indices[:2]]
                top_two_peaks = np.sort(top_two_peaks)
                
                start_peak = top_two_peaks[0]
                end_peak = top_two_peaks[1]
                
                start_frame = start_peak + offset_frames
                end_frame = end_peak - offset_frames + 1
                end_frame = len(frame_keys)
                
                print(f"    最後一個視頻：開始於第一峰值+{offset_frames} (幀 {start_frame})，保留結尾（第 {end_frame} 幀）")
            else:
                end_frame = len(frame_keys)
        else:
            print(f"    中間視頻：正常裁剪")
        
        print(f"  ✂️  擷取幀範圍: 第 {start_frame} 幀 → 第 {end_frame} 幀")
        print(f"  📊 擷取 {end_frame - start_frame} 幀 (原始: {len(frame_keys)} 幀)")
        print(f"  📉 移除開頭: {start_frame} 幀, 移除結尾: {len(frame_keys) - end_frame} 幀")
        
        # 可視化
        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(velocity, label='Hand Velocity', linewidth=2.5, color='#2E86AB')
            
            if len(peaks) >= 2:
                peak_heights = velocity[peaks]
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_two_peaks = peaks[sorted_indices[:2]]
                top_two_peaks = np.sort(top_two_peaks)
                
                ax.plot(peaks, velocity[peaks], 'o', color='lightgray', markersize=10, 
                       label=f'All Peaks ({len(peaks)})')
                ax.plot(top_two_peaks, velocity[top_two_peaks], 'X', color='#FF6B6B', 
                       markersize=15, label=f'Main 2 Peaks', zorder=10)
            
            ax.axvline(x=start_frame, color='#06A77D', linestyle='--', linewidth=3, 
                      label=f'Start: {start_frame}')
            ax.axvline(x=end_frame, color='#D62828', linestyle='--', linewidth=3, 
                      label=f'End: {end_frame}')
            
            ax.axvspan(start_frame, end_frame, alpha=0.25, color='green', label='Extracted')
            
            position_text = "FIRST" if is_first else ("LAST" if is_last else "MIDDLE")
            if smooth_params:
                position_text += f"\n✨ Enhanced Smoothing"
            ax.text(0.02, 0.98, position_text, transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Velocity (pixels)', fontsize=12, fontweight='bold')
            ax.set_title(f'Peak Detection: {Path(tracked_dir).name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.3)
            
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
            for key in valid_keys:
                new_key = f"frame_{frame_counter:06d}"
                concatenated_tracking[new_key] = tracking_data[key]
                concatenated_frame_keys.append(new_key)
                frame_counter += 1
            print(f"  ✓ 添加 {len(valid_keys)} 幀")
        else:
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
    
    # === 對最終串接結果進行強化全局平滑 ===
    if smooth_params and global_smooth_sigma > 0:
        print(f"\n{'='*70}")
        print(f"對最終串接結果進行強化全局平滑...")
        print(f"{'='*70}")
        smooth_all_tracking_params(concatenated_tracking, concatenated_frame_keys,
                                  body_sigma=global_smooth_sigma,
                                  hand_sigma=global_smooth_sigma * 1.2,
                                  face_sigma=global_smooth_sigma * 0.8,
                                  use_median_for_hands=True)
    
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
    
    # 複製圖像到LMDB（省略代碼，與原版相同）
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
    
    for lmdb_engine in all_lmdb_engines:
        if lmdb_engine is not None:
            lmdb_engine.close()
    
    # 生成摘要（省略，與原版相同）
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
    print(f"✓ 參數平滑: 已應用（強化版 + 驗證）")
    print(f"✓ 輸出目錄: {output_dir}")

# 使用範例
if __name__ == "__main__":
    tracked_dirs = [
        "/home/paohan/GUAVA/EHM-Tracker/results/working/你/",
        "/home/paohan/GUAVA/EHM-Tracker/results/working/幫我/",
        "/home/paohan/GUAVA/EHM-Tracker/results/working/找/",
        "/home/paohan/GUAVA/EHM-Tracker/results/working/工作/",
        "/home/paohan/GUAVA/EHM-Tracker/results/working/可以/"

        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/你/",
        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/幫我/",
        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/找/",
        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/工作/",
        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/可以/",

        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/大家/",
        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/上課/",
        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/準時/",
        # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/要/"
    ]
    
    # 使用新的輸出目錄（避免覆蓋）
    # output_dir = "/home/paohan/GUAVA/outputs/concatenated_smoothed_0115_on_time"
    output_dir = "/home/paohan/GUAVA/outputs/concatenated_smoothed_0318_working_2"
    
    concatenate_tracking_data_with_peaks(
        tracked_dirs, output_dir,
        window_size=5,
        min_peak_distance=10,
        offset_frames=3,
        num_transition_frames=15,
        
        # === 強化平滑參數 ===
        smooth_params=True,
        body_sigma=1.5,
        hand_sigma=1.8,              # 提高到 5.0
        face_sigma=1.2,
        global_smooth_sigma=1.5,
        use_median_for_hands=True,
        
        visualize=True
    )