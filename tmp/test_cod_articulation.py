"""
只读实测：验证 COD-2026-RoboMaster-Balance USD 能否在 IsaacLab 正常作为 articulation 加载。
不修改任何文件。
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--steps", type=int, default=200)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

USD = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd"


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="/World/envs/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
        actuators={},
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))

    # Before creating the scene, spawn the robot manually to inspect prim nesting.
    from pxr import Usd
    stage = sim.stage

    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print(">>> SCENE BUILT OK")

    robot = scene["robot"]
    print(">>> ALL PRIM PATHS under /World/envs/env_0/Robot:")
    for prim in sim.stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith("/World/envs/env_0/Robot"):
            print(f"    {prim.GetTypeName():24s} {p}")
            if str(prim.GetPath()).count("/") > 6:
                break
    print(f">>> num_bodies = {robot.num_bodies}")
    print(f">>> num_joints = {robot.num_joints}")
    print(f">>> joint_names = {robot.joint_names}")
    print(f">>> body_names = {robot.body_names}")

    # default joint positions
    print(f">>> default_joint_pos shape = {robot.data.default_joint_pos.shape}")

    for i in range(args.steps):
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if i % 50 == 0:
            pos = robot.data.root_pos_w[0]
            print(f">>> step {i:4d} root_pos_z={pos[2].item():.4f} "
                  f"joint_pos[:6]={robot.data.joint_pos[0,:6].tolist()}")

    print(">>> SIM DONE (no crash)")
    print(f">>> final root_pos_w = {robot.data.root_pos_w[0].tolist()}")
    sim.stop()


main()
