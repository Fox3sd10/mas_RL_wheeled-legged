"""验证 cod_balance.py 的 ArticulationCfg 能否正确 spawn + 建立 articulation。
只读验证：不训练，只 spawn 出来跑几步看是否稳定。"""
import argparse
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": args.headless})

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from agent_world.assets.cod_balance import COD_BALANCE_CFG


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200.0, device="cuda:0"))
    sim.set_camera_view([2.0, 0.0, 1.0], [0.0, 0.0, 0.3])

    # 地面（用一个大薄板替代 GroundPlaneCfg，避免该版本材质绑定 bug）
    ground = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    ground.func("/World/ground", ground, translation=(0.0, 0.0, -0.05))
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)

    # spawn 机器人
    cfg = COD_BALANCE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg)

    sim.reset()

    print(">>> num_joints   =", robot.num_joints)
    print(">>> joint_names  =", robot.joint_names)
    print(">>> num_bodies   =", robot.num_bodies)
    print(">>> body_names   =", robot.body_names)

    # 找关节索引
    def idx(name):
        return robot.joint_names.index(name)

    # 简单 PD 让它站：所有主动腿关节保持 0 位置
    for i in range(600):
        root_pos = robot.data.root_pos_w
        if i % 100 == 0:
            print(f">>> step {i:4d}  root_z={root_pos[0,2].item():.4f}")
        # 目标：腿关节回到 0，轮子 0 速度
        tgt_q = robot.data.default_joint_pos.clone()
        tgt_qd = torch.zeros_like(tgt_q)
        robot.set_joint_position_target(tgt_q)
        robot.set_joint_velocity_target(tgt_qd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    print(">>> SIM DONE (no crash)")
    print(">>> final root_pos =", robot.data.root_pos_w[0].tolist())


if __name__ == "__main__":
    main()
    app.close()
