import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from pytorch3d.io import save_obj

# 添加项目路径
sys.path.insert(0, '/home/paohan/GUAVA/models/modules')
sys.path.insert(0, '/home/paohan/GUAVA/EHM-Tracker')

from smplx.SMPLX import SMPLX

def export_smplx_to_obj(tracking_pkl_path, output_obj_path):
    """
    导出SMPL-X为OBJ格式
    """
    print(f"Loading tracking data from: {tracking_pkl_path}")
    with open(tracking_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data)}")
    print(f"Data keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载SMPL-X模型
    model_path = '/home/paohan/GUAVA/assets/SMPLX/SMPLX_NEUTRAL_2020.npz'
    model = SMPLX(model_path, num_betas=10, device=device)
    
    # 提取参数
    if isinstance(data, dict):
        body_pose = data.get('body_pose', np.zeros((1, 63)))
        global_orient = data.get('global_orient', np.zeros((1, 3)))
        betas = data.get('betas', data.get('shape', np.zeros((1, 10))))
        transl = data.get('transl', np.zeros((1, 3)))
        
        print(f"body_pose shape: {body_pose.shape}")
        print(f"global_orient shape: {global_orient.shape}")
        print(f"betas shape: {betas.shape}")
    else:
        print("Warning: Unexpected data format")
        return
    
    # 转换为tensor
    body_pose_tensor = torch.from_numpy(body_pose).to(device).float()
    global_orient_tensor = torch.from_numpy(global_orient).to(device).float()
    betas_tensor = torch.from_numpy(betas).to(device).float()
    transl_tensor = torch.from_numpy(transl).to(device).float()
    
    # 生成SMPL-X网格
    print("Generating SMPL-X mesh...")
    with torch.no_grad():
        output = model(
            body_pose=body_pose_tensor,
            global_orient=global_orient_tensor,
            betas=betas_tensor,
            transl=transl_tensor,
            return_verts=True
        )
    
    vertices = output['vertices'].cpu().numpy()[0]
    faces = model.faces
    
    print(f"Vertices shape: {vertices.shape}")
    print(f"Faces shape: {faces.shape}")
    
    # 保存为OBJ
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    
    verts_tensor = torch.from_numpy(vertices).float()
    faces_tensor = torch.from_numpy(faces).long()
    
    save_obj(output_obj_path, verts_tensor, faces_tensor)
    print(f"✅ OBJ exported to: {output_obj_path}")

if __name__ == "__main__":
    pkl_path = "/home/paohan/GUAVA/outputs/app/tracked_driven_video/1022_1min/optim_tracking_ehm.pkl"
    output_obj = "/home/paohan/GUAVA/outputs/export/1022_1min.obj"
    
    if os.path.exists(pkl_path):
        export_smplx_to_obj(pkl_path, output_obj)
    else:
        print(f"Error: File not found: {pkl_path}")
        print("\nAvailable pickle files:")
        for root, dirs, files in os.walk('/home/paohan/GUAVA/outputs'):
            for file in files:
                if file.endswith('.pkl'):
                    print(f"  {os.path.join(root, file)}")