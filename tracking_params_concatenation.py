import numpy as np
import os
import json
import pickle
from pathlib import Path
from scipy.interpolate import interp1d
import shutil
from tqdm import tqdm
import sys
sys.path.append('/home/paohan/GUAVA')
from utils.lmdb import LMDBEngine

def load_tracking_data(tracked_dir):
    """
    载入追踪数据
    
    Args:
        tracked_dir: 追踪结果目录 (包含 optim_tracking_ehm.pkl 等文件)
    
    Returns:
        tracking_data, id_share_params, videos_info
    """
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
    """
    保存追踪数据
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tracking_pkl = os.path.join(output_dir, 'optim_tracking_ehm.pkl')
    id_share_pkl = os.path.join(output_dir, 'id_share_params.pkl')
    videos_info_json = os.path.join(output_dir, 'videos_info.json')
    
    with open(tracking_pkl, 'wb') as f:
        pickle.dump(tracking_data, f)
    
    with open(id_share_pkl, 'wb') as f:
        pickle.dump(id_share_params, f)
    
    with open(videos_info_json, 'w') as f:
        json.dump(videos_info, f, indent=4)

def detect_motion_in_tracking(tracking_data, frame_keys):
    """
    从追踪参数中检测运动强度
    通过计算相邻帧之间的参数变化量来判断
    """
    motion_scores = []
    
    for i, frame_key in enumerate(frame_keys):
        if i == 0:
            motion_scores.append(0)
            continue
        
        prev_key = frame_keys[i-1]
        curr_key = frame_key
        
        # 计算身体姿态变化
        prev_body_pose = tracking_data[prev_key]['smplx_coeffs']['body_pose']
        curr_body_pose = tracking_data[curr_key]['smplx_coeffs']['body_pose']
        body_motion = np.sum((curr_body_pose - prev_body_pose) ** 2)
        
        # 计算手部姿态变化
        prev_lhand = tracking_data[prev_key]['smplx_coeffs']['left_hand_pose']
        curr_lhand = tracking_data[curr_key]['smplx_coeffs']['left_hand_pose']
        lhand_motion = np.sum((curr_lhand - prev_lhand) ** 2)
        
        prev_rhand = tracking_data[prev_key]['smplx_coeffs']['right_hand_pose']
        curr_rhand = tracking_data[curr_key]['smplx_coeffs']['right_hand_pose']
        rhand_motion = np.sum((curr_rhand - prev_rhand) ** 2)
        
        # 计算表情变化
        prev_expr = tracking_data[prev_key]['flame_coeffs']['expression_params']
        curr_expr = tracking_data[curr_key]['flame_coeffs']['expression_params']
        expr_motion = np.sum((curr_expr - prev_expr) ** 2)
        
        # 综合运动分数（手部权重更高，因为手语主要看手部）
        total_motion = body_motion * 0.2 + (lhand_motion + rhand_motion) * 0.6 + expr_motion * 0.2
        motion_scores.append(total_motion)
    
    return np.array(motion_scores)

def trim_static_frames_from_tracking(tracking_data, frame_keys, motion_threshold=0.01, 
                                     trim_start=True, trim_end=True):
    """
    基于运动强度修剪静止帧
    """
    motion_scores = detect_motion_in_tracking(tracking_data, frame_keys)
    n_frames = len(frame_keys)
    
    # 找开头：第一个超过阈值的帧
    start_frame = 0
    if trim_start:
        for i in range(n_frames):
            if motion_scores[i] > motion_threshold:
                start_frame = i
                break
    
    # 找结尾：最后一个超过阈值的帧
    end_frame = n_frames
    if trim_end:
        for i in range(n_frames - 1, -1, -1):
            if motion_scores[i] > motion_threshold:
                end_frame = i + 1
                break
    
    print(f"  检测到有效帧范围: {start_frame} 到 {end_frame} (共 {end_frame - start_frame} 帧)")
    print(f"  平均运动强度: {np.mean(motion_scores):.6f}")
    print(f"  最大运动强度: {np.max(motion_scores):.6f}")
    
    return start_frame, end_frame, motion_scores

def interpolate_tracking_params(param1, param2, alpha):
    """
    在两个参数之间进行插值
    
    Args:
        param1, param2: 参数字典或数组
        alpha: 插值系数 (0-1)
    """
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

def create_smooth_transition(tracking_data, frame_key1, frame_key2, num_transition_frames=10):
    """
    在两帧之间创建平滑过渡帧
    
    Args:
        tracking_data: 追踪数据字典
        frame_key1: 起始帧键
        frame_key2: 结束帧键
        num_transition_frames: 过渡帧数量
    
    Returns:
        transition_frames: 过渡帧数据列表
    """
    param1 = tracking_data[frame_key1]
    param2 = tracking_data[frame_key2]
    
    transition_frames = []
    
    for i in range(1, num_transition_frames + 1):
        alpha = i / (num_transition_frames + 1)
        
        # 创建新的帧数据
        new_frame = {}
        
        # 插值 SMPLX 参数
        new_frame['smplx_coeffs'] = {}
        for key in param1['smplx_coeffs'].keys():
            if key == 'camera_RT_params':
                # 相机参数特殊处理（保持第一个视频的相机参数）
                new_frame['smplx_coeffs'][key] = param1['smplx_coeffs'][key].copy()
            else:
                new_frame['smplx_coeffs'][key] = interpolate_tracking_params(
                    param1['smplx_coeffs'][key],
                    param2['smplx_coeffs'][key],
                    alpha
                )
        
        # 插值 FLAME 参数
        new_frame['flame_coeffs'] = {}
        for key in param1['flame_coeffs'].keys():
            if key == 'camera_RT_params':
                new_frame['flame_coeffs'][key] = param1['flame_coeffs'][key].copy()
            else:
                new_frame['flame_coeffs'][key] = interpolate_tracking_params(
                    param1['flame_coeffs'][key],
                    param2['flame_coeffs'][key],
                    alpha
                )
        
        # 插值 MANO 参数
        for hand_key in ['left_mano_coeffs', 'right_mano_coeffs']:
            new_frame[hand_key] = {}
            for key in param1[hand_key].keys():
                if key == 'camera_RT_params':
                    new_frame[hand_key][key] = param1[hand_key][key].copy()
                else:
                    new_frame[hand_key][key] = interpolate_tracking_params(
                        param1[hand_key][key],
                        param2[hand_key][key],
                        alpha
                    )
        
        # 复制 dwpose_rlt (2D关键点可以线性插值或直接复制)
        new_frame['dwpose_rlt'] = param1['dwpose_rlt'].copy()
        
        transition_frames.append(new_frame)
    
    return transition_frames

def concatenate_tracking_data(tracked_dirs, output_dir, 
                              motion_threshold=0.01,
                              num_transition_frames=15,
                              use_auto_detect=True,
                              visualize=False,
                              copy_images=True):
    """
    串接多个追踪数据，自动移除静止帧并生成平滑过渡
    
    Args:
        tracked_dirs: 追踪数据目录列表
        output_dir: 输出目录
        motion_threshold: 运动阈值
        num_transition_frames: 过渡帧数量
        use_auto_detect: 是否使用自动检测
        visualize: 是否生成可视化
        copy_images: 是否复制图像到LMDB（需要用于渲染）
    """
    print(f"\n开始处理 {len(tracked_dirs)} 个追踪数据...")
    print(f"设定: 过渡帧数 {num_transition_frames}, 运动阈值: {motion_threshold}")
    
    all_tracking_data = []
    all_ranges = []
    all_frame_keys = []
    all_lmdb_engines = []
    first_id_share_params = None
    
    # 载入所有追踪数据
    for idx, tracked_dir in enumerate(tracked_dirs):
        print(f"\n[{idx+1}/{len(tracked_dirs)}] 载入追踪数据: {Path(tracked_dir).name}")
        
        if not os.path.exists(tracked_dir):
        tracking_data, id_share_params, videos_info = load_tracking_data(tracked_dir)
        
        # 载入 LMDB 数据库
        lmdb_path = os.path.join(tracked_dir, 'img_lmdb')
        if os.path.exists(lmdb_path):
            lmdb_engine = LMDBEngine(lmdb_path, write=False)
            all_lmdb_engines.append(lmdb_engine)
        else:
            print(f"  警告: 未找到 img_lmdb 目录")
            all_lmdb_engines.append(None)
        
        # 保存第一个视频的身份参数
        if first_id_share_params is None:
            first_id_share_params = id_share_params
        
        # 获取帧键列表
        video_name = list(videos_info.keys())[0]
        frame_keys = videos_info[video_name]['frames_keys']
        # 获取帧键列表
        video_name = list(videos_info.keys())[0]
        frame_keys = videos_info[video_name]['frames_keys']
        
        print(f"  载入了 {len(frame_keys)} 帧")
        
        # 判断是否需要修剪
        is_first = (idx == 0)
        is_last = (idx == len(tracked_dirs) - 1)
        
        if use_auto_detect:
            start_frame, end_frame, motion_scores = trim_static_frames_from_tracking(
                tracking_data,
                frame_keys,
                motion_threshold,
                trim_start=(not is_first),
                trim_end=(not is_last)
            )
        else:
            start_frame = 0 if is_first else 20
            end_frame = len(frame_keys) if is_last else len(frame_keys) - 20
            motion_scores = None
        
        # 可视化运动分数
        if visualize and motion_scores is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(motion_scores, label='Motion Score')
            plt.axhline(y=motion_threshold, color='r', linestyle='--', label=f'Threshold ({motion_threshold})')
            plt.axvspan(start_frame, end_frame, alpha=0.2, color='green', label='Keep')
            if start_frame > 0:
                plt.axvspan(0, start_frame, alpha=0.2, color='red', label='Remove')
            if end_frame < len(frame_keys):
                plt.axvspan(end_frame, len(frame_keys), alpha=0.2, color='red')
            plt.xlabel('Frame')
            plt.ylabel('Motion Score')
            plt.title(f'Motion Analysis: {Path(tracked_dir).name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            viz_path = os.path.join(output_dir, f'motion_analysis_{idx}.png')
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
    concatenated_tracking = {}
    concatenated_frame_keys = []
    frame_counter = 0
    
    for i in range(len(all_tracking_data)):
        tracking_data = all_tracking_data[i]
        frame_keys = all_frame_keys[i]
        start_frame, end_frame = all_ranges[i]
        
        print(f"\n处理第 {i+1} 个追踪数据:")
        print(f"  使用帧范围: {start_frame} 到 {end_frame}")
        
        # 提取有效帧
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
                
                print(f"  生成 {num_transition_frames} 个过渡帧...")
                transition_frames = create_smooth_transition(
                    {**concatenated_tracking, next_key_original: tracking_data[next_key_original]},
                    prev_key,
                    next_key_original,
                    num_transition_frames
                )
                
                # 添加过渡帧
                for trans_frame in transition_frames:
                    new_key = f"frame_{frame_counter:06d}"
                    concatenated_tracking[new_key] = trans_frame
                    concatenated_frame_keys.append(new_key)
                    frame_counter += 1
                
                # 添加剩余帧
                for key in valid_keys:
                    new_key = f"frame_{frame_counter:06d}"
                    concatenated_tracking[new_key] = tracking_data[key]
                    concatenated_frame_keys.append(new_key)
    # 保存结果
    print(f"\n保存串接后的追踪数据...")
    save_tracking_data(output_dir, concatenated_tracking, first_id_share_params, videos_info)
    
    # 复制图像到 LMDB
    if copy_images and len(all_lmdb_engines) > 0:
        print(f"\n复制图像数据到 LMDB...")
        output_lmdb_path = os.path.join(output_dir, 'img_lmdb')
        output_lmdb = LMDBEngine(output_lmdb_path, write=True)
        
        frame_counter = 0
        for i in range(len(all_tracking_data)):
            lmdb_engine = all_lmdb_engines[i]
            if lmdb_engine is None:
                print(f"  跳过第 {i+1} 个视频（无 LMDB）")
                continue
            
            frame_keys = all_frame_keys[i]
            start_frame, end_frame = all_ranges[i]
            valid_keys = frame_keys[start_frame:end_frame]
            
            print(f"  复制第 {i+1} 个视频的图像 ({len(valid_keys)} 帧)...")
            
            if i == 0:
                # 第一个视频：直接复制
                for old_key in tqdm(valid_keys, desc=f"  Video {i+1}"):
                    new_key = f"frame_{frame_counter:06d}"
                    # 复制所有图像数据
                    for img_type in ['ori_image', 'body_image', 'body_mask', 'body_matting', 
    concatenate_tracking_data(
        tracked_dirs,
        output_dir,
        motion_threshold=0.005,  # 根据实际情况调整
        num_transition_frames=15,  # 过渡帧数量（约0.5秒）
        use_auto_detect=True,
        visualize=True,
        copy_images=True  # 复制图像数据到LMDB（渲染需要）
    )       else:
                # 后续视频：添加过渡帧（使用第一帧图像）和正常帧
                if len(valid_keys) > 0:
                    # 为过渡帧使用当前视频的第一帧图像
                    first_key = valid_keys[0]
                    for _ in range(num_transition_frames):
                        new_key = f"frame_{frame_counter:06d}"
                        for img_type in ['ori_image', 'body_image', 'body_mask', 'body_matting', 
                                        'head_image', 'left_hand_image', 'right_hand_image']:
                            try:
                                img_data = lmdb_engine[f'{first_key}/{img_type}']
                                output_lmdb.dump(f'{new_key}/{img_type}', payload=img_data, type='image')
                            except:
                                pass
                        frame_counter += 1
                    
                    # 复制正常帧
                    for old_key in tqdm(valid_keys, desc=f"  Video {i+1}"):
                        new_key = f"frame_{frame_counter:06d}"
                        for img_type in ['ori_image', 'body_image', 'body_mask', 'body_matting', 
                                        'head_image', 'left_hand_image', 'right_hand_image']:
                            try:
                                img_data = lmdb_engine[f'{old_key}/{img_type}']
                                output_lmdb.dump(f'{new_key}/{img_type}', payload=img_data, type='image')
                            except:
                                pass
                        frame_counter += 1
        
        output_lmdb.close()
        print(f"  LMDB 创建完成！")
        
        # 关闭所有输入 LMDB
        for lmdb_engine in all_lmdb_engines:
            if lmdb_engine is not None:
                lmdb_engine.close()
    
    print(f"\n✓ 成功!")
    print(f"✓ 总共 {len(concatenated_frame_keys)} 帧 ({len(concatenated_frame_keys)/30:.2f} 秒)")
    print(f"✓ 已保存至: {output_dir}")
    print(f"\n现在可以使用以下命令进行渲染:")
    print(f"export PYTHONPATH=/home/paohan/GUAVA:$PYTHONPATH")
    print(f"python main/test.py -d '0' -m assets/GUAVA -s outputs/concatenated_render --data_path {output_dir}")
        }
    }
    
    # 保存结果
    print(f"\n保存串接后的追踪数据...")
    save_tracking_data(output_dir, concatenated_tracking, first_id_share_params, videos_info)
    
    print(f"\n✓ 成功!")
    print(f"✓ 总共 {len(concatenated_frame_keys)} 帧 ({len(concatenated_frame_keys)/30:.2f} 秒)")
    print(f"✓ 已保存至: {output_dir}")
    print(f"\n现在可以使用以下命令进行渲染:")
    print(f"python main/test.py -d '0' -m assets/GUAVA -s outputs/concatenated_render --data_path {output_dir}")

# 使用范例
if __name__ == "__main__":
    # 追踪数据目录列表（每个目录应包含 optim_tracking_ehm.pkl, id_share_params.pkl, videos_info.json）
    # 这些是 EHM-Tracker 的输出目录，不是渲染后的视频文件
    tracked_dirs = [
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/你/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/幫我/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/找/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/工作/",
        "/home/paohan/GUAVA/outputs/app/tracked_driven_video/可以/",
    ]
    

    
    output_dir = "/home/paohan/GUAVA/outputs/concatenated_tracking_1224"
    
    concatenate_tracking_data(
        tracked_dirs,
        output_dir,
        motion_threshold=0.005,  # 根据实际情况调整
        num_transition_frames=15,  # 过渡帧数量（约0.5秒）
        use_auto_detect=True,
        visualize=True
    )
