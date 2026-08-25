#!/usr/bin/env python3
"""
完整的追踪数据串接工具
1. 使用视频运动检测来确定帧范围（与 motion_concatenation.py 一致）
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
import sys
sys.path.append('/home/paohan/GUAVA')
from utils.lmdb import LMDBEngine

def detect_motion_from_lmdb(lmdb_engine, frame_keys, roi_top=0.3, roi_bottom=0.7):
    """
    从LMDB中的图像检测运动强度（与motion_concatenation.py逻辑一致）
    """
    motion_scores = []
    prev_gray = None
    
    for i, frame_key in enumerate(tqdm(frame_keys, desc="  分析运动")):
        if i == 0:
            motion_scores.append(0)
            continue
        
        try:
            # 从LMDB读取body_image
            img_data = lmdb_engine[f'{frame_key}/body_image']
            img_np = img_data.numpy().transpose(1, 2, 0)  # CHW -> HWC
            img_np = (img_np * 255).astype(np.uint8)
            
            # 转换为灰度图
            if img_np.shape[2] == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np[:,:,0]
            
            # 定义ROI
            h, w = gray.shape
            roi_y1 = int(h * roi_top)
            roi_y2 = int(h * roi_bottom)
            roi = gray[roi_y1:roi_y2, :]
            
            if prev_gray is not None:
                prev_roi = prev_gray[roi_y1:roi_y2, :]
                diff = cv2.absdiff(roi, prev_roi)
                motion_score = np.sum(diff > 30) / diff.size
                motion_scores.append(motion_score)
            else:
                motion_scores.append(0)
            
            prev_gray = gray
        except Exception as e:
            motion_scores.append(0)
    
    return np.array(motion_scores)

def trim_static_frames_video_style(motion_scores, threshold=0.024, trim_start=True, trim_end=True):
    """
    使用与motion_concatenation.py完全相同的逻辑删除静止帧
    """
    n_frames = len(motion_scores)
    
    start_frame = 0
    if trim_start:
        for i in range(n_frames):
            if motion_scores[i] > threshold:
                start_frame = i
                break
    
    end_frame = n_frames
    if trim_end:
        for i in range(n_frames - 1, -1, -1):
            if motion_scores[i] > threshold:
                end_frame = i + 1
                break
    
    print(f"  检测到有效帧范围: {start_frame} 到 {end_frame} (共 {end_frame - start_frame} 帧)")
    print(f"  平均运动强度: {np.mean(motion_scores):.6f}, 最大: {np.max(motion_scores):.6f}")
    
    return start_frame, end_frame, motion_scores

def load_tracking_data(tracked_dir):
    """载入追踪数据"""
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

def concatenate_tracking_data_with_video_motion(tracked_dirs, output_dir, 
                                               motion_threshold=0.024,
                                               num_transition_frames=15,
                                               visualize=False):
    """
    使用视频运动检测串接追踪数据
    """
    print(f"\n开始处理 {len(tracked_dirs)} 个追踪数据...")
    print(f"设定: 过渡帧数 {num_transition_frames}, 运动阈值: {motion_threshold}")
    print(f"注意: 使用与 motion_concatenation.py 相同的运动检测逻辑")
    
    all_tracking_data, all_ranges, all_frame_keys, all_lmdb_engines = [], [], [], []
    first_id_share_params = None
    
    # 载入所有追踪数据并分析运动
    for idx, tracked_dir in enumerate(tracked_dirs):
        print(f"\n[{idx+1}/{len(tracked_dirs)}] 载入追踪数据: {Path(tracked_dir).name}")
        
        if not os.path.exists(tracked_dir):
            print(f"  错误: 目录不存在!")
            continue
        
        tracking_data, id_share_params, videos_info = load_tracking_data(tracked_dir)
        
        # 载入LMDB
        lmdb_path = os.path.join(tracked_dir, 'img_lmdb')
        if not os.path.exists(lmdb_path):
            print(f"  错误: 未找到 img_lmdb 目录!")
            continue
        
        lmdb_engine = LMDBEngine(lmdb_path, write=False)
        all_lmdb_engines.append(lmdb_engine)
        
        if first_id_share_params is None:
            first_id_share_params = id_share_params
        
        video_name = list(videos_info.keys())[0]
        frame_keys = videos_info[video_name]['frames_keys']
        print(f"  载入了 {len(frame_keys)} 帧")
        
        # 使用视频运动检测（与motion_concatenation.py一致）
        print(f"  从图像检测运动强度...")
        motion_scores = detect_motion_from_lmdb(lmdb_engine, frame_keys)
        
        # 判断是否需要修剪
        is_first, is_last = (idx == 0), (idx == len(tracked_dirs) - 1)
        start_frame, end_frame, motion_scores = trim_static_frames_video_style(
            motion_scores, motion_threshold,
            trim_start=(not is_first), trim_end=(not is_last)
        )
        
        # 可视化
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(motion_scores, label='Motion Score')
            plt.axhline(y=motion_threshold, color='r', linestyle='--', 
                       label=f'Threshold ({motion_threshold})')
            plt.axvspan(start_frame, end_frame, alpha=0.2, color='green', label='Keep')
            if start_frame > 0:
                plt.axvspan(0, start_frame, alpha=0.2, color='red', label='Remove')
            if end_frame < len(frame_keys):
                plt.axvspan(end_frame, len(frame_keys), alpha=0.2, color='red')
            plt.xlabel('Frame')
            plt.ylabel('Motion Score')
            plt.title(f'Motion Analysis (Video-based): {Path(tracked_dir).name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            viz_path = os.path.join(output_dir, f'motion_analysis_video_{idx}.png')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(viz_path)
            plt.close()
            print(f"  运动分析图已保存至: {viz_path}")
        
        all_tracking_data.append(tracking_data)
        all_ranges.append((start_frame, end_frame))
        all_frame_keys.append(frame_keys)
    
    if len(all_tracking_data) == 0:
        print("\n错误: 没有成功载入任何追踪数据!")
        return
    
    # 串接追踪数据
    print(f"\n开始串接追踪参数...")
    concatenated_tracking = {}
    concatenated_frame_keys = []
    frame_counter = 0
    
    for i in range(len(all_tracking_data)):
        tracking_data, frame_keys = all_tracking_data[i], all_frame_keys[i]
        start_frame, end_frame = all_ranges[i]
        
        print(f"\n处理第 {i+1} 个追踪数据:")
        print(f"  使用帧范围: {start_frame} 到 {end_frame}")
        
        valid_keys = frame_keys[start_frame:end_frame]
        
        if i == 0:
            # 第一个视频：直接添加
            for key in valid_keys:
                new_key = f"frame_{frame_counter:06d}"
                concatenated_tracking[new_key] = tracking_data[key]
                concatenated_frame_keys.append(new_key)
                frame_counter += 1
            print(f"  添加 {len(valid_keys)} 帧")
        else:
            # 后续视频：添加过渡帧
            if len(concatenated_frame_keys) > 0 and len(valid_keys) > 0:
                prev_key = concatenated_frame_keys[-1]
                next_key_original = valid_keys[0]
                
                print(f"  生成 {num_transition_frames} 个平滑过渡帧...")
                print(f"    从第 {i} 个视频的第 {end_frame-1} 帧")
                print(f"    到第 {i+1} 个视频的第 {start_frame} 帧")
                
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
                
                print(f"  添加 {len(valid_keys)} 帧")
    
    # 创建videos_info
    videos_info = {
        "concatenated_video": {
            "frames_num": len(concatenated_frame_keys),
            "frames_keys": concatenated_frame_keys
        }
    }
    
    # 保存结果
    print(f"\n保存串接后的追踪数据...")
    save_tracking_data(output_dir, concatenated_tracking, first_id_share_params, videos_info)
    
    # 复制图像到LMDB
    print(f"\n复制图像数据到 LMDB...")
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
        
        print(f"  复制第 {i+1} 个视频的图像 ({len(valid_keys)} 帧)...")
        
        if i == 0:
            # 第一个视频：直接复制
            for old_key in tqdm(valid_keys, desc=f"  Video {i+1}"):
                new_key = f"frame_{frame_counter:06d}"
                for img_type in img_types:
                    try:
                        img_data = lmdb_engine[f'{old_key}/{img_type}']
                        output_lmdb.dump(f'{new_key}/{img_type}', payload=img_data, type='image')
                    except:
                        pass
                frame_counter += 1
        else:
            # 添加过渡帧图像（使用下一个视频的第一帧）
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
                
                # 复制正常帧
                for old_key in tqdm(valid_keys, desc=f"  Video {i+1}"):
                    new_key = f"frame_{frame_counter:06d}"
                    for img_type in img_types:
                        try:
                            img_data = lmdb_engine[f'{old_key}/{img_type}']
                            output_lmdb.dump(f'{new_key}/{img_type}', payload=img_data, type='image')
                        except:
                            pass
                    frame_counter += 1
    
    output_lmdb.close()
    print(f"  LMDB 创建完成！")
    
    # 关闭所有输入LMDB
    for lmdb_engine in all_lmdb_engines:
        if lmdb_engine is not None:
            lmdb_engine.close()
    
    print(f"\n✓ 成功!")
    print(f"✓ 总共 {len(concatenated_frame_keys)} 帧 ({len(concatenated_frame_keys)/30:.2f} 秒)")
    print(f"✓ 已保存至: {output_dir}")
    print(f"\n现在可以使用以下命令进行渲染:")
    print(f"  PYTHONPATH=/home/paohan/GUAVA:$PYTHONPATH")
    print(f"  python main/test.py -d '0' -m assets/GUAVA -s outputs/concat_render --data_path {output_dir}")
    print(f"\n或使用指定形象进行 cross-reenactment:")
    print(f"  python main/test.py -d '0' -m assets/GUAVA -s outputs/concat_render_cross \\")
    print(f"    --data_path {output_dir} \\")
    print(f"    --source_data_path <源形象路径> \\")
    print(f"    --skip_self_act --render_cross_act")

# 使用范例
if __name__ == "__main__":
    tracked_dirs = [
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/你/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/幫我/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/找/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/工作/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/可以/",

    # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/大家/",
    # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/上課/",
    # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/準時/",
    # "/home/paohan/GUAVA/outputs/app/tracked_driven_video/要"
    ]
    
    output_dir = "/home/paohan/GUAVA/outputs/concatenated_tracking_final"
    # output_dir = "/home/paohan/GUAVA/outputs/concatenated_tracking_final_class_on_time"
    
    concatenate_tracking_data_with_video_motion(
        tracked_dirs, output_dir,
        motion_threshold=0.024,  # 与 motion_concatenation.py 相同
        num_transition_frames=15,
        visualize=True
    )
