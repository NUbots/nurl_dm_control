import numpy as np
import torch
from dm_control import suite
from rl_algorithm.ppo import PPO
import matplotlib.pyplot as plt  # 添加 matplotlib 导入

# Hyperparameters
num_episodes = 1000
max_steps = 200

env = suite.load(domain_name="nubots_CMU", task_name="stand")
observation_spec = env.observation_spec()
action_spec = env.action_spec()
obs_dim = sum(np.prod(v.shape) for v in observation_spec.values())
act_dim = action_spec.shape[0]

ppo = PPO(obs_dim, act_dim)

def flatten_obs(obs):
    return np.concatenate([np.ravel(obs[k]) for k in observation_spec.keys()])

for episode in range(num_episodes):
    obs = env.reset().observation
    state = flatten_obs(obs)
    episode_reward = 0
    states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []
    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, log_prob = ppo.model.act(state_tensor)
        value = ppo.model.get_value(state_tensor)
        action_np = action.detach().numpy()[0]
        time_step = env.step(action_np)
        next_state = flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        done = time_step.last()
        states.append(state)
        actions.append(action_np)
        log_probs.append(log_prob.item())
        rewards.append(reward)
        masks.append(1.0 - float(done))
        values.append(value.item())
        state = next_state
        episode_reward += reward

        # 渲染图像并显示
        img = env.physics.render(height=480, width=640, camera_id=0)
        plt.imshow(img)
        plt.axis('off')
        plt.pause(0.001)
        plt.clf()
        # Allow user to close the program by closing the matplotlib window
        if not plt.fignum_exists(1):
            print("Matplotlib window closed. Exiting RL loop.")
            exit(0)

        if done:
            break
    # Compute GAE and update PPO
    next_value = ppo.model.get_value(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item()
    values.append(next_value)
    returns = []
    gae = 0
    gamma, lam = ppo.gamma, ppo.lam
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values[i])
    advantages = np.array(returns) - np.array(values[:-1])
    ppo.update(states, actions, log_probs, returns, advantages)
    print(f"Episode {episode}, Reward: {episode_reward}")


