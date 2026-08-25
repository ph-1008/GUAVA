import os
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
import trimesh

class SMPLXToFBX:
    def __init__(self, smplx_model_path='assets/SMPLX', device='cuda'):
        """
        初始化SMPL-X转换器
        """
        self.device = device
        self.smplx_model_path = smplx_model_path
        
        try:
            from smplx import SMPLX
            
            model_file_2020 = os.path.join(smplx_model_path, 'SMPLX_NEUTRAL_2020.npz')
            model_file_std = os.path.join(smplx_model_path, 'SMPLX_NEUTRAL.npz')
            
            if os.path.exists(model_file_2020) and not os.path.exists(model_file_std):
                print(f"📦 检测到 SMPLX_NEUTRAL_2020.npz，创建符号链接...")
                os.symlink('SMPLX_NEUTRAL_2020.npz', model_file_std)
            
            if not os.path.exists(model_file_std):
                raise FileNotFoundError(f"模型文件不存在: {model_file_std}")
            
            print(f"📦 加载SMPL-X模型...")
            
            # 不使用手部，简化模型
            self.smplx_model = SMPLX(
                model_path=smplx_model_path,
                gender='neutral',
                batch_size=1,
                num_betas=10,
                use_face_contour=False,
                ext='npz'
            ).to(device)
            print(f"✅ SMPL-X模型加载成功")
        except Exception as e:
            print(f"⚠️  SMPL-X模型加载失败: {e}")
            raise

    def load_tracking_data(self, pkl_path: str) -> Dict:
        """加载跟踪数据pickle文件"""
        print(f"📂 正在加载: {pkl_path}")
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"文件不存在: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 加载成功，包含 {len(data)} 帧数据")
        return data

    def smplx_params_to_vertices(self, smplx_params: Dict, debug=False) -> Tuple[np.ndarray, np.ndarray]:
        """将SMPL-X参数转换为网格顶点"""
        try:
            # 提取参数
            betas = smplx_params.get('shape', np.zeros(10))
            if isinstance(betas, (list, np.ndarray)):
                betas = np.array(betas).flatten()
            else:
                betas = np.zeros(10)
            
            if betas.shape[0] < 10:
                betas = np.pad(betas, (0, 10 - betas.shape[0]))
            else:
                betas = betas[:10]
            
            # global_pose: (3,)
            global_pose = smplx_params.get('global_pose', np.zeros(3))
            global_pose = np.array(global_pose).flatten()[:3]
            if global_pose.shape[0] < 3:
                global_pose = np.pad(global_pose, (0, 3 - global_pose.shape[0]))
            
            # body_pose: (21, 3) -> (63,)
            body_pose = smplx_params.get('body_pose', np.zeros((21, 3)))
            body_pose = np.array(body_pose).reshape(-1)[:63]
            if body_pose.shape[0] < 63:
                body_pose = np.pad(body_pose, (0, 63 - body_pose.shape[0]))
            
            # expression: (50,) -> 取前10个
            exp = smplx_params.get('exp', np.zeros(50))
            exp = np.array(exp).flatten()
            if exp.shape[0] < 10:
                exp = np.pad(exp, (0, 10 - exp.shape[0]))
            else:
                exp = exp[:10]
            
            if debug:
                print(f"\n📊 处理后的数据形状:")
                print(f"  betas: {betas.shape}")
                print(f"  global_pose: {global_pose.shape}")
                print(f"  body_pose: {body_pose.shape}")
                print(f"  exp: {exp.shape}")
            
            # 转换为tensor
            betas_t = torch.from_numpy(betas).float().unsqueeze(0).to(self.device)
            body_pose_t = torch.from_numpy(body_pose).float().unsqueeze(0).to(self.device)
            global_orient = torch.from_numpy(global_pose).float().unsqueeze(0).to(self.device)
            expression = torch.from_numpy(exp).float().unsqueeze(0).to(self.device)
            
            # 生成SMPL-X网格 - 只传递身体和表情参数，不传手部
            with torch.no_grad():
                output = self.smplx_model(
                    betas=betas_t,
                    body_pose=body_pose_t,
                    global_orient=global_orient,
                    expression=expression,
                    return_verts=True
                )
            
            vertices = output.vertices[0].detach().cpu().numpy()
            faces = self.smplx_model.faces
            
            return vertices, faces
            
        except Exception as e:
            print(f"❌ 参数转换失败: {e}")
            raise

    def fix_mesh_orientation(self, mesh):
        """修正网格方向 - 翻转Y轴"""
        # 翻转Y轴以修正上下颠倒
        mesh.vertices[:, 1] *= -1
        # 翻转面的法线方向（反向面的顶点顺序）
        mesh.faces = mesh.faces[:, ::-1]
        return mesh

    def create_animation_sequence(self, tracking_data: Dict, output_dir: str = 'outputs/fbx', max_frames: int = None):
        """为每一帧创建网格文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        frame_meshes = []
        frame_list = sorted(tracking_data.keys())
        
        if max_frames:
            frame_list = frame_list[:max_frames]
        
        print(f"🎬 正在处理 {len(frame_list)} 帧...")
        
        for idx, frame_id in enumerate(frame_list):
            frame_data = tracking_data[frame_id]
            
            if isinstance(frame_data, dict) and 'smplx_coeffs' in frame_data:
                smplx_params = frame_data['smplx_coeffs']
            else:
                smplx_params = frame_data
            
            try:
                vertices, faces = self.smplx_params_to_vertices(smplx_params, debug=(idx == 0))
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                # 修正方向
                mesh = self.fix_mesh_orientation(mesh)
                
                frame_meshes.append({
                    'frame_id': frame_id,
                    'mesh': mesh,
                    'vertices': vertices
                })
                
                if (idx + 1) % max(1, len(frame_list) // 10) == 0:
                    print(f"  ✓ 已处理 {idx + 1}/{len(frame_list)} 帧")
                    
            except Exception as e:
                if idx < 3:
                    print(f"⚠️  帧 {frame_id} 处理失败: {str(e)[:100]}")
                continue
        
        print(f"✅ 成功处理 {len(frame_meshes)}/{len(frame_list)} 帧")
        return frame_meshes

    def export_to_glb(self, frame_meshes: list, output_path: str):
        """导出为GLB格式"""
        print(f"💾 正在导出GLB: {output_path}")
        
        if not frame_meshes:
            print("❌ 没有可导出的网格数据")
            return
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        first_mesh = frame_meshes[0]['mesh']
        first_mesh.export(output_path)
        
        print(f"✅ GLB导出完成: {output_path}")

    def export_obj_sequence(self, frame_meshes: list, output_dir: str):
        """导出为OBJ序列（每帧一个文件）"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 正在导出OBJ序列到: {output_dir}")
        
        for idx, frame_data in enumerate(frame_meshes):
            mesh = frame_data['mesh']
            
            output_path = os.path.join(output_dir, f"frame_{idx:06d}.obj")
            mesh.export(output_path)
            
            if (idx + 1) % max(1, len(frame_meshes) // 10) == 0:
                print(f"  ✓ 已导出 {idx + 1}/{len(frame_meshes)} 帧")
        
        print(f"✅ OBJ序列导出完成")

    def export_to_fbx(self, frame_meshes: list, output_path: str):
        """导出为FBX格式（使用trimesh）"""
        print(f"💾 正在导出FBX: {output_path}")
        
        if not frame_meshes:
            print("❌ 没有可导出的网格数据")
            return
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        try:
            # 直接使用trimesh导出
            first_mesh = frame_meshes[0]['mesh']
            first_mesh.export(output_path)
            print(f"✅ FBX导出完成: {output_path}")
            
        except Exception as e:
            print(f"⚠️  FBX导出失败: {e}")
            print("   尝试导出为OBJ格式...")
            try:
                first_mesh = frame_meshes[0]['mesh']
                obj_path = output_path.replace('.fbx', '.obj')
                first_mesh.export(obj_path)
                print(f"✅ 已导出为OBJ: {obj_path}")
            except Exception as e2:
                print(f"❌ OBJ导出也失败: {e2}")

    def export_to_fbx_blender(self, frame_meshes: list, output_path: str):
        """使用Blender Python API导出FBX动画"""
        try:
            import bpy
        except ImportError:
            print("❌ 需要使用Blender的Python环境运行此函数")
            print("使用方法: blender --python smplx_to_fbx.py")
            self.export_to_fbx(frame_meshes, output_path)
            return
        
        print(f"🎬 正在创建Blender动画: {output_path}")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # 清空场景
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # 为每一帧创建对象
        scene = bpy.context.scene
        fps = 30
        scene.render.fps = fps
        scene.frame_end = len(frame_meshes)
        
        for idx, frame_data in enumerate(frame_meshes):
            mesh = frame_data['mesh']
            
            # 创建Blender网格
            bpy_mesh = bpy.data.meshes.new(f"mesh_{idx}")
            bpy_mesh.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
            
            obj = bpy.data.objects.new(f"body_{idx}", bpy_mesh)
            bpy.context.collection.objects.link(obj)
            
            # 设置关键帧
            obj.hide_render = False
            obj.keyframe_insert(data_path="hide_render", frame=idx)
            if idx > 0:
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_render", frame=idx - 1)
        
        # 导出为FBX
        bpy.ops.export_scene.fbx(filepath=output_path)
        print(f"✅ FBX导出完成: {output_path}")

    def export_to_bvh(self, tracking_data: Dict, output_path: str, fps: int = 30):
        """导出为BVH格式动画（包含骨骼和关键帧）"""
        print(f"💾 正在导出BVH: {output_path}")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        try:
            # SMPL-X的22个关节对应于22个骨骼
            # 骨骼父子关系 - 必须与运动数据顺序对应
            skeleton_tree = {
                'Hips': None,  # ROOT - 6个通道 (Xposition Yposition Zposition Xrotation Yrotation Zrotation)
                'LeftUpLeg': 'Hips',      # body_pose[0:3]
                'LeftLeg': 'LeftUpLeg',   # body_pose[3:6]
                'LeftFoot': 'LeftLeg',    # body_pose[6:9]
                'LeftToeBase': 'LeftFoot',  # body_pose[9:12]
                'RightUpLeg': 'Hips',     # body_pose[12:15]
                'RightLeg': 'RightUpLeg', # body_pose[15:18]
                'RightFoot': 'RightLeg',  # body_pose[18:21]
                'RightToeBase': 'RightFoot', # body_pose[21:24]
                'Spine': 'Hips',          # body_pose[24:27]
                'Spine1': 'Spine',        # body_pose[27:30]
                'Spine2': 'Spine1',       # body_pose[30:33]
                'Neck': 'Spine2',         # body_pose[33:36]
                'Head': 'Neck',           # body_pose[36:39]
                'LeftShoulder': 'Spine2', # body_pose[39:42]
                'LeftArm': 'LeftShoulder', # body_pose[42:45]
                'LeftForeArm': 'LeftArm', # body_pose[45:48]
                'LeftHand': 'LeftForeArm', # body_pose[48:51]
                'RightShoulder': 'Spine2', # body_pose[51:54]
                'RightArm': 'RightShoulder', # body_pose[54:57]
                'RightForeArm': 'RightArm', # body_pose[57:60]
                'RightHand': 'RightForeArm' # body_pose[60:63]
            }
            
            # 提取关键帧数据
            frame_list = sorted(tracking_data.keys())
            num_frames = len(frame_list)
            
            motion_data = []
            
            print(f"  正在提取 {num_frames} 帧的运动数据...")
            
            for idx, frame_id in enumerate(frame_list):
                frame_data = tracking_data[frame_id]
                
                if isinstance(frame_data, dict) and 'smplx_coeffs' in frame_data:
                    smplx_params = frame_data['smplx_coeffs']
                else:
                    smplx_params = frame_data
                
                # 提取全局旋转和身体姿态
                global_pose = smplx_params.get('global_pose', np.zeros(3))
                global_pose = np.array(global_pose).flatten()[:3]
                
                body_pose = smplx_params.get('body_pose', np.zeros((21, 3)))
                body_pose = np.array(body_pose).reshape(-1)[:63]
                
                # 组合为完整的骨骼数据 (global_pose + body_pose)
                # 总共66个值: 3(全局位置) + 3(全局旋转) + 63(身体姿态)
                frame_rotations = np.concatenate([global_pose, body_pose])
                motion_data.append(frame_rotations)
                
                if (idx + 1) % max(1, num_frames // 10) == 0:
                    print(f"    ✓ 已提取 {idx + 1}/{num_frames} 帧")
            
            motion_data = np.array(motion_data)
            
            # 写入BVH文件
            self._write_bvh_file(output_path, skeleton_tree, motion_data, fps)
            print(f"✅ BVH导出完成: {output_path}")
            
        except Exception as e:
            print(f"❌ BVH导出失败: {e}")
            import traceback
            traceback.print_exc()

    def _write_bvh_file(self, filepath: str, skeleton_tree: dict, 
                        motion_data: np.ndarray, fps: int):
        """写入BVH文件"""
        
        hierarchy_str = "HIERARCHY\n"
        hierarchy_str += "ROOT Hips\n"
        hierarchy_str += "{\n"
        hierarchy_str += "  OFFSET 0.0 0.0 0.0\n"
        hierarchy_str += "  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
        
        # 递归生成完整骨骼树
        def write_joint_recursive(parent_name, level=1):
            indent = "  " * level
            result = ""
            
            # 查找所有直接子节点
            children = [name for name, parent in skeleton_tree.items() 
                       if parent == parent_name]
            
            for child_name in children:
                result += f"{indent}JOINT {child_name}\n"
                result += f"{indent}" + "{\n"
                result += f"{indent}  OFFSET 0.0 0.0 0.0\n"
                result += f"{indent}  CHANNELS 3 Xrotation Yrotation Zrotation\n"
                
                # 递归处理子节点
                result += write_joint_recursive(child_name, level + 1)
                
                result += f"{indent}" + "}\n"
            
            return result
        
        # 生成完整的骨骼树
        hierarchy_str += write_joint_recursive('Hips', 1)
        hierarchy_str += "}\n"
        
        # 生成动作数据部分 (MOTION)
        motion_str = "MOTION\n"
        motion_str += f"Frames: {motion_data.shape[0]}\n"
        motion_str += f"Frame Time: {1.0 / fps}\n"
        
        # 写入每一帧的数据
        for frame_idx in range(motion_data.shape[0]):
            frame_values = motion_data[frame_idx]
            # 需要全局位置数据 - 使用平均位置或为0
            # 格式: Xposition Yposition Zposition Xrotation Yrotation Zrotation (ROOT)
            #       然后是其他关节的 Xrotation Yrotation Zrotation
            
            # 提取前3个作为全局旋转，补充位置为0
            root_values = [0.0, 0.0, 0.0] + frame_values[:3].tolist() + frame_values[3:].tolist()
            motion_str += " ".join([f"{v:.6f}" for v in root_values]) + "\n"
        
        # 写入文件
        with open(filepath, 'w') as f:
            f.write(hierarchy_str)
            f.write(motion_str)


def main():
    """主函数"""
    
    SMPLX_MODEL_PATH = 'assets/SMPLX'
    PKL_FILE = '/home/paohan/GUAVA/outputs/app/tracked_driven_video/1022_1min/optim_tracking_ehm.pkl'
    OUTPUT_DIR = 'outputs/fbx'
    
    try:
        print("🚀 初始化SMPL-X转换器...")
        converter = SMPLXToFBX(smplx_model_path=SMPLX_MODEL_PATH, device='cuda')
        
        print("\n📂 加载跟踪数据...")
        tracking_data = converter.load_tracking_data(PKL_FILE)
        
        print("\n🎬 创建动画序列...")
        frame_meshes = converter.create_animation_sequence(
            tracking_data, 
            OUTPUT_DIR,
            max_frames=None  # None处理所有帧
        )
        
        if frame_meshes:
            # print("\n💾 导出GLB格式...")
            # converter.export_to_glb(
            #     frame_meshes,
            #     os.path.join(OUTPUT_DIR, 'animation.glb')
            # )
            
            # print("\n💾 导出OBJ序列...")
            # converter.export_obj_sequence(
            #     frame_meshes,
            #     os.path.join(OUTPUT_DIR, 'obj_sequence')
            # )

            # print("\n💾 导出FBX格式...")
            # converter.export_to_fbx(
            #     frame_meshes,
            #     os.path.join(OUTPUT_DIR, 'animation.fbx')
            # )
            
            print("\n💾 导出BVH格式...")
            converter.export_to_bvh(
                tracking_data,
                os.path.join(OUTPUT_DIR, 'animation.bvh'),
                fps=30
            )
            
            print(f"\n✅ 导出完成！输出目录: {OUTPUT_DIR}")
        else:
            print("❌ 没有成功处理的帧数据")
            
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
    