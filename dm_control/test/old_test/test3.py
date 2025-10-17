import os
import numpy as np
from dm_control import suite
from dm_control.locomotion import soccer as dm_soccer
# from dm_control.soccer import load
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control import viewer
from PIL import Image
import cv2

def save_video(frames, filename="soccer_simulation.mp4", fps=30):
    """
    Save a sequence of frames as an MP4 video using OpenCV.
    Args:
        frames (list of np.array): List of video frames (each frame is a numpy array).
        filename (str): Name of the output video file.
        fps (int): Frames per second for the video.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR
    out.release()
    print(f"Video saved as {filename}")

# Load the dm_control soccer environment
# env = dm_soccer.load()
env = dm_soccer.load(team_size=2,
                     time_limit=1.5,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.NUBOTS)

# Reset the environment
time_step = env.reset()
action_specs = env.action_spec()
# Frame collection
frames = []
frame_count = 300  # Number of frames to record

print("Recording video...")
for _ in range(frame_count):
    # Render the environment to a frame
    frame = env.physics.render(height=480, width=640, camera_id=3)
    frames.append(frame)
    actions = []
    # Step through the environment with random actions
    # action = np.random.uniform(-1, 1, env.action_spec().shape)
    for action_spec in action_specs:
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)    
    time_step = env.step(action)

# Save the recorded frames as a video
save_video(frames)