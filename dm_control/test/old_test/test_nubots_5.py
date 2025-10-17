import imageio
from dm_control import mujoco
from dm_control.suite import nubots

env = nubots.stand()
time_step = env.reset()

frames = []
for _ in range(100):
    print("Time step:", time_step)
    action = env.action_spec().generate_value()
    time_step = env.step(action)
    pixels = mujoco.engine.render(env.physics, height=480, width=640, camera_id=0)
    frames.append(pixels)

imageio.mimsave("nubots.gif", frames, fps=20)
