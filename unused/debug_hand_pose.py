import pickle
import numpy as np

pkl_path = '/home/paohan/GUAVA/outputs/app/tracked_driven_video/1022_1min/optim_tracking_ehm.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

first_key = list(data.keys())[0]
first_frame = data[first_key]
smplx_params = first_frame['smplx_coeffs']

left_hand = smplx_params['left_hand_pose']
right_hand = smplx_params['right_hand_pose']

print(f"left_hand_pose type: {type(left_hand)}")
print(f"left_hand_pose shape: {np.array(left_hand).shape}")
print(f"left_hand_pose ndim: {np.array(left_hand).ndim}")
print(f"left_hand_pose dtype: {np.array(left_hand).dtype}")
print(f"left_hand_pose sample: {np.array(left_hand)[:2]}")

print(f"\nright_hand_pose type: {type(right_hand)}")
print(f"right_hand_pose shape: {np.array(right_hand).shape}")
print(f"right_hand_pose ndim: {np.array(right_hand).ndim}")
print(f"right_hand_pose dtype: {np.array(right_hand).dtype}")
print(f"right_hand_pose sample: {np.array(right_hand)[:2]}")

# 检查最后一个维度
left_flat = np.array(left_hand).reshape(-1)
print(f"\nleft_hand flattened shape: {left_flat.shape}")
print(f"left_hand flattened values (first 20): {left_flat[:20]}")