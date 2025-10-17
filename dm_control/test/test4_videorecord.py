import numpy as np
import cv2
from dm_control import suite
from dm_control import mujoco

def save_video(frames, filename="nubots_simulation.mp4", fps=30):
    """
    Save a sequence of frames as an MP4 video using OpenCV.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"✅ Video saved as {filename}")

# --- Load environment ---
env = suite.load(domain_name="nubots", task_name="stand")
action_spec = env.action_spec()
print("Action Spec:", action_spec)
# --- Reset environment ---
time_step = env.reset()

# --- Collect frames ---
frames = []
frame_count = 30  # e.g., 10 seconds at 30 fps

print("🎥 Recording Nubots video...")
for _ in range(frame_count):
    # Render the current frame
    frame = env.physics.render(height=480, width=640, camera_id=0)
    frames.append(frame)

    # Random action
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)

# --- Save to video ---
save_video(frames, filename="dm_control/nubots_stand.mp4", fps=30)
