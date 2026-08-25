import pickle
import numpy as np
from pathlib import Path

PKL_PATH = "/home/paohan/GUAVA/outputs/app/tracked_driven_video/1022_1min/optim_tracking_ehm.pkl"   # 改成你的實際路徑
OUT_NPZ  = "smplx_motion_from_ehm.npz"

def main():
    pkl_path = Path(PKL_PATH)
    data = pickle.load(open(pkl_path, "rb"))

    frame_keys = sorted(data.keys())  # frame_000000, frame_000001, ...
    n_frames = len(frame_keys)
    print(f"Loaded {n_frames} frames")

    # 這些 shape 來自我幫你看過的資料
    body_pose_list = []
    global_pose_list = []
    left_hand_list = []
    right_hand_list = []

    for k in frame_keys:
        smplx = data[k]["smplx_coeffs"]
        body_pose_list.append(smplx["body_pose"])        # (21,3)
        global_pose_list.append(smplx["global_pose"])    # (3,)
        left_hand_list.append(smplx["left_hand_pose"])   # (15,3)
        right_hand_list.append(smplx["right_hand_pose"]) # (15,3)

    body_pose = np.stack(body_pose_list, axis=0)         # (T,21,3)
    global_pose = np.stack(global_pose_list, axis=0)     # (T,3)
    left_hand = np.stack(left_hand_list, axis=0)         # (T,15,3)
    right_hand = np.stack(right_hand_list, axis=0)       # (T,15,3)

    np.savez(
        OUT_NPZ,
        frame_keys=np.array(frame_keys),
        body_pose=body_pose,
        global_pose=global_pose,
        left_hand=left_hand,
        right_hand=right_hand,
    )
    print(f"Saved motion to {OUT_NPZ}")

if __name__ == "__main__":
    main()
