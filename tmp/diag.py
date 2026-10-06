"""诊断：构建 Cod-Stand 环境，跑几步，检查 obs/reward 是否有 NaN。"""
import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from agent_tasks.manager_based.cod_stand import *  # noqa: 触发 gym.register
from agent_tasks.manager_based.cod_stand.cod_stand_env_cfg import CodStandEnvCfg

cfg = CodStandEnvCfg()
cfg.scene.num_envs = 16
env = gym.make("Cod-Stand-v0", cfg=cfg).unwrapped

env.reset()
_oc = env.observation_manager.compute()
print("obs keys:", list(_oc.keys()), "policy shape:", _oc["policy"].shape, flush=True)

for step in range(30):
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    obs, rew, term, trunc, info = env.step(act)
    o = obs["policy"] if isinstance(obs, dict) else obs
    bad_obs = torch.isnan(o).any().item() or torch.isinf(o).any().item()
    bad_rew = torch.isnan(rew).any().item() or torch.isinf(rew).any().item()
    if bad_obs or bad_rew or step < 3 or step % 10 == 0:
        print(f"step {step}: obs_nan={bad_obs} rew_nan={bad_rew} rew_mean={rew.mean().item():.4f}", flush=True)
        if bad_obs:
            print("  NaN obs idx:", torch.isnan(o).nonzero()[:10].tolist(), flush=True)
    if step == 5:
        cmd = env.command_manager.get_command("base_velocity")
        print("  command sample:", cmd[:3].tolist(), flush=True)
        print("  reward per-term mean:", [f"{x:.4f}" for x in env.reward_manager._episode_sums.keys()] if hasattr(env.reward_manager, "_episode_sums") else "n/a", flush=True)

env.close()
simulation_app.close()
print("DONE", flush=True)
