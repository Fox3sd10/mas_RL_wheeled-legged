"""验证 Cod-Stand-v0 环境能否构建 + 跑几步（随机动作），不训练。"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import gymnasium as gym
import torch
import agent_tasks  # 触发注册
from isaaclab_tasks.utils import parse_env_cfg

# IsaacLab 用 parse_env_cfg 读取注册项里的 env_cfg_entry_point 并实例化
env_cfg = parse_env_cfg("Cod-Stand-v0", device="cuda:0", num_envs=64)
env = gym.make("Cod-Stand-v0", cfg=env_cfg)
print(">>> ENV BUILT OK")
print(">>> num_envs      =", env.unwrapped.num_envs)
print(">>> num_actions   =", env.unwrapped.action_space.shape)
print(">>> num_obs       =", env.unwrapped.observation_space.shape)

obs, _ = env.reset()
print(">>> reset OK, obs keys =", list(obs.keys()) if isinstance(obs, dict) else "tensor")
obs_t = obs["policy"] if isinstance(obs, dict) else obs
print(">>> policy obs shape =", obs_t.shape)

for i in range(50):
    action = 2 * torch.rand(env.unwrapped.action_space.shape, device="cuda:0") - 1
    obs, rew, term, trunc, info = env.step(action)
    if i % 10 == 0:
        print(f">>> step {i:3d}  reward={rew.mean().item():.4f}")

print(">>> ENV STEP OK, final mean reward per step:", rew.mean().item())
env.close()
app.close()
