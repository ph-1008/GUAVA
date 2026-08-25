import pickle
import numpy as np

pkl_path = "/home/paohan/GUAVA/outputs/app/tracked_driven_video/找/optim_tracking_ehm.pkl"
# pkl_path = "/home/paohan/GUAVA/EHM-Tracker/results/working/找/optim_tracking_ehm.pkl"
# pkl_path = "yZB4pNR4VAI/pre/optim_tracking_ehm.pkl"


with open(pkl_path, "rb") as f:
    data = pickle.load(f)

frames = sorted(data.keys())

# 取雙手 2D keypoints: shape = (T, 2, 21, 2)
hands = np.array([
    data[fr]["dwpose_rlt"]["hands"]
    for fr in frames
])

# 攤平成 (T, 42, 2)
pose = hands.reshape(len(frames), -1, 2)

# velocity: p_t - p_{t-1}
velocity = np.diff(pose, axis=0)

# acceleration: v_t - v_{t-1}
acceleration = np.diff(velocity, axis=0)

# jerk: a_t - a_{t-1}
jerk = np.diff(acceleration, axis=0)

mean_velocity = np.mean(np.linalg.norm(velocity, axis=-1))
mean_acceleration = np.mean(np.linalg.norm(acceleration, axis=-1))
mean_jerk = np.mean(np.linalg.norm(jerk, axis=-1))

print("Mean Velocity:", mean_velocity)
print("Mean Acceleration:", mean_acceleration)
print("Mean Jerk:", mean_jerk)

# ============================

def collect_motion_params(data):
    frames = sorted(data.keys())
    params = []

    for fr in frames:
        smplx = data[fr]["smplx_coeffs"]

        vec = np.concatenate([
            smplx["global_pose"].reshape(-1),
            smplx["body_pose"].reshape(-1),
            smplx["left_hand_pose"].reshape(-1),
            smplx["right_hand_pose"].reshape(-1),
        ])

        params.append(vec)

    return np.array(params)

motion = collect_motion_params(data)

velocity = np.diff(motion, axis=0)
acceleration = np.diff(velocity, axis=0)
jerk = np.diff(acceleration, axis=0)

print("Param Velocity:", np.mean(np.linalg.norm(velocity, axis=-1)))
print("Param Acceleration:", np.mean(np.linalg.norm(acceleration, axis=-1)))
print("Param Jerk:", np.mean(np.linalg.norm(jerk, axis=-1)))