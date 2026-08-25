import pickle
import numpy as np

pkl_path = '/home/paohan/GUAVA/outputs/app/tracked_driven_video/1022_1min/optim_tracking_ehm.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"总帧数: {len(data)}")
print(f"\n第一帧的键: {list(data.keys())[:3]}")

# 获取第一帧
first_key = list(data.keys())[0]
first_frame = data[first_key]

print(f"\n第一帧结构:")
print(f"类型: {type(first_frame)}")
if isinstance(first_frame, dict):
    print(f"键: {first_frame.keys()}")
    
    # 检查每个字段的大小
    for key, value in first_frame.items():
        if isinstance(value, (np.ndarray, list)):
            arr = np.array(value)
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  {key}: {type(value)}")
            if isinstance(value, dict):
                for k, v in value.items():
                    v_arr = np.array(v)
                    print(f"    {k}: shape={v_arr.shape}, dtype={v_arr.dtype}")