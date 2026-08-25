import pickle
import numpy as np
import os
import json
import lmdb
from pathlib import Path

def load_tracking_params_from_pkl(video_folder):
    """从 PKL 文件加载追踪参数"""
    pkl_path = os.path.join(video_folder, 'optim_tracking_ehm.pkl')
    
    if not os.path.exists(pkl_path):
        print(f"找不到 {pkl_path}")
        return None
    
    with open(pkl_path, 'rb') as f:
        tracking_data = pickle.load(f)
    
    print(f"\n追踪数据结构:")
    print(f"  类型: {type(tracking_data)}")
    if isinstance(tracking_data, dict):
        print(f"  键数量: {len(tracking_data.keys())}")
        first_key = list(tracking_data.keys())[0]
        print(f"  第一个键: {first_key}")
        print(f"  第一个值类型: {type(tracking_data[first_key])}")
        if isinstance(tracking_data[first_key], dict):
            print(f"  第一帧参数: {tracking_data[first_key].keys()}")
    
    return tracking_data

def load_tracking_params_from_lmdb(video_folder):
    """从 LMDB 加载追踪参数"""
    lmdb_path = os.path.join(video_folder, 'img_lmdb')
    
    if not os.path.exists(lmdb_path):
        print(f"找不到 {lmdb_path}")
        return None
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    # 收集所有帧的参数
    params_by_frame = {}
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                key_str = key.decode('utf-8')
                data = pickle.loads(value)
                
                # 提取帧号
                if key_str.isdigit():
                    frame_idx = int(key_str)
                    params_by_frame[frame_idx] = data
                    
            except Exception as e:
                continue
    
    env.close()
    
    if not params_by_frame:
        print("LMDB 中没有找到帧数据")
        return None
    
    # 重组成按参数名分组的格式
    params = {}
    sorted_frames = sorted(params_by_frame.keys())
    first_frame_data = params_by_frame[sorted_frames[0]]
    
    print(f"\n第一帧包含的参数: {list(first_frame_data.keys())}")
    
    for param_name in first_frame_data.keys():
        if param_name in ['img', 'img_path', 'image', 'body_image', 'head_image', 'left_hand_image', 'right_hand_image', 'body_mask', 'body_matting', 'ori_image']:
            continue
        
        param_values = []
        for frame_idx in sorted_frames:
            if param_name in params_by_frame[frame_idx]:
                param_values.append(params_by_frame[frame_idx][param_name])
        
        if param_values:
            try:
                params[param_name] = np.array(param_values)
            except:
                params[param_name] = param_values
    
    return params, sorted_frames

def extract_params_from_tracking_dict(tracking_dict):
    """从追踪字典中提取参数数组"""
    if not tracking_dict:
        return None
    
    frame_keys = sorted(tracking_dict.keys())
    params = {}
    
    first_frame = tracking_dict[frame_keys[0]]
    
    # 遍历第一帧的所有参数
    for param_category in first_frame.keys():
        if isinstance(first_frame[param_category], dict):
            # 如果是嵌套字典（如 smplx_coeffs, flame_coeffs）
            for sub_param in first_frame[param_category].keys():
                param_name = f"{param_category}.{sub_param}"
                param_values = []
                
                for frame_key in frame_keys:
                    if param_category in tracking_dict[frame_key] and \
                       sub_param in tracking_dict[frame_key][param_category]:
                        val = tracking_dict[frame_key][param_category][sub_param]
                        if isinstance(val, (np.ndarray, list)):
                            param_values.append(val)
                
                if param_values:
                    try:
                        params[param_name] = np.array(param_values)
                    except:
                        params[param_name] = param_values
        else:
            # 直接是参数
            param_values = []
            for frame_key in frame_keys:
                if param_category in tracking_dict[frame_key]:
                    val = tracking_dict[frame_key][param_category]
                    if isinstance(val, (np.ndarray, list)):
                        param_values.append(val)
            
            if param_values:
                try:
                    params[param_category] = np.array(param_values)
                except:
                    params[param_category] = param_values
    
    return params, frame_keys

def detect_jumps(params, threshold=3.0):
    """检测参数中的跳动"""
    jump_info = {}
    
    for key, values in params.items():
        if not isinstance(values, np.ndarray):
            continue
        
        # 跳过标量或空数组
        if values.ndim < 2 or len(values) < 2:
            continue
        
        # 计算帧间差异
        diffs = np.abs(np.diff(values, axis=0))
        
        # 找出跳动超过阈值的位置
        max_diffs = np.max(diffs.reshape(diffs.shape[0], -1), axis=1)
        mean_diff = np.mean(max_diffs)
        std_diff = np.std(max_diffs)
        
        # 使用统计方法检测异常值
        jump_threshold = mean_diff + threshold * std_diff
        jump_frames = np.where(max_diffs > jump_threshold)[0]
        
        if len(jump_frames) > 0:
            jump_info[key] = {
                'jump_frames': jump_frames.tolist(),
                'max_jump': float(np.max(max_diffs)),
                'mean_diff': float(mean_diff),
                'std_diff': float(std_diff),
                'shape': values.shape,
                'jump_values': [float(max_diffs[i]) for i in jump_frames[:5]]
            }
    
    return jump_info

def fix_jumps(params, jump_info, method='interpolate'):
    """修复参数跳动"""
    fixed_params = {}
    
    for key, values in params.items():
        if key not in jump_info:
            fixed_params[key] = values
            continue
        
        values = values.copy()
        info = jump_info[key]
        jump_frames = info['jump_frames']
        
        print(f"\n修复 {key} 的跳动:")
        print(f"  跳动帧: {jump_frames[:10]}{'...' if len(jump_frames) > 10 else ''}")
        print(f"  最大跳动值: {info['max_jump']:.4f}")
        
        for frame_idx in jump_frames:
            if method == 'interpolate' and frame_idx > 0 and frame_idx < len(values) - 1:
                # 线性插值
                values[frame_idx + 1] = (values[frame_idx] + values[frame_idx + 2]) / 2
            elif method == 'copy_prev' and frame_idx > 0:
                # 使用前一帧的值
                values[frame_idx + 1] = values[frame_idx]
        
        fixed_params[key] = values
        print(f"  ✓ 已修复 {len(jump_frames)} 个跳动帧")
    
    return fixed_params

def save_fixed_params_to_pkl(video_folder, original_tracking_dict, fixed_params, frame_keys):
    """将修复后的参数保存回 PKL 文件"""
    output_pkl_path = os.path.join(video_folder, 'optim_tracking_ehm_fixed.pkl')
    
    import copy
    fixed_tracking_dict = copy.deepcopy(original_tracking_dict)
    
    for frame_idx, frame_key in enumerate(frame_keys):
        for param_name, param_values in fixed_params.items():
            if '.' in param_name:
                category, sub_param = param_name.split('.', 1)
                if category in fixed_tracking_dict[frame_key]:
                    fixed_tracking_dict[frame_key][category][sub_param] = param_values[frame_idx]
            else:
                fixed_tracking_dict[frame_key][param_name] = param_values[frame_idx]
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(fixed_tracking_dict, f)
    
    print(f"\n✓ 修复后的数据已保存到: {output_pkl_path}")

def main():
    # 只检测指定的文件夹
    video_folder = '/home/paohan/GUAVA/outputs/app/tracked_driven_video/要'
    
    print(f"\n{'='*80}")
    print(f"检测视频: {os.path.basename(video_folder)}")
    print(f"{'='*80}")
    
    # 优先尝试从 PKL 文件加载
    tracking_dict = load_tracking_params_from_pkl(video_folder)
    
    if tracking_dict:
        print("\n从 PKL 文件提取参数...")
        result = extract_params_from_tracking_dict(tracking_dict)
    else:
        print("\n从 LMDB 加载追踪参数...")
        result = load_tracking_params_from_lmdb(video_folder)
    
    if result is None:
        print(f"无法加载追踪数据")
        return
    
    params, frame_indices = result
    
    print(f"\n共加载 {len(frame_indices)} 帧数据")
    print("\n参数列表:")
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")
    
    print("\n检测跳动...")
    jump_info = detect_jumps(params, threshold=2.0)
    
    if not jump_info:
        print("✓ 未检测到明显的参数跳动")
        return
    
    print("\n发现以下参数有跳动:")
    for key, info in jump_info.items():
        print(f"\n{key}:")
        print(f"  形状: {info['shape']}")
        print(f"  跳动帧数: {len(info['jump_frames'])}")
        print(f"  跳动位置: {info['jump_frames'][:10]}{'...' if len(info['jump_frames']) > 10 else ''}")
        print(f"  最大跳动: {info['max_jump']:.4f}")
        print(f"  平均差异: {info['mean_diff']:.4f}")
        print(f"  标准差: {info['std_diff']:.4f}")
        print(f"  跳动值示例: {info['jump_values']}")
    
    # 保存检测结果
    output_file = os.path.join(video_folder, 'jump_detection_report.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(jump_info, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 检测报告已保存到: {output_file}")
    
    # 询问是否修复
    print("\n是否要修复这些跳动? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("\n选择修复方法:")
        print("1. interpolate - 线性插值")
        print("2. copy_prev - 复制前一帧")
        method_choice = input("请选择 (1/2): ").strip()
        
        method = 'interpolate' if method_choice == '1' else 'copy_prev'
        
        fixed_params = fix_jumps(params, jump_info, method=method)
        
        # 保存修复后的参数
        if tracking_dict:
            save_fixed_params_to_pkl(video_folder, tracking_dict, fixed_params, frame_indices)
        else:
            print("注意: 当前仅支持修复 PKL 格式的追踪数据")
    else:
        print("\n已取消修复")

if __name__ == '__main__':
    main()