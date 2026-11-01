# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-scene PhysX evaluation environment with measured, history-aware termination."""

import json
from pathlib import Path

import numpy as np
import torch
from scene import compose
from success import TOOL_NAMES, CleanupSuccess

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import matrix_from_quat


class WorkbenchCleanupEnv:
    """Expose tool wrenches and drawer effort, observations, reset, and success termination.

    No robot is present. A later robot evaluation can replace the wrench action source
    while reusing ``observe`` and ``CleanupSuccess`` unchanged. The two accessories are
    fixed distractors. The tools and all drawer links are simulated dynamic bodies.
    """

    def __init__(self, sim, asset_root: Path, output: Path):
        self.sim = sim
        self.dt = sim.get_physics_dt()
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.geometry = compose(get_current_stage(), asset_root)
        ground = sim_utils.GroundPlaneCfg(color=(0.12, 0.14, 0.16))
        ground.func("/World/Ground", ground)
        light = sim_utils.DomeLightCfg(intensity=1800.0)
        light.func("/World/Light", light)
        self.bench = Articulation(
            ArticulationCfg(
                prim_path="/World/Workbench/Bench",
                spawn=None,
                actuators={
                    "force_control": ImplicitActuatorCfg(
                        joint_names_expr=[".*"], stiffness=0.0, damping=0.0, joint_effort_limit=100.0
                    )
                },
            )
        )
        self.tools = {
            name: RigidObject(RigidObjectCfg(prim_path=f"/World/Tools/{name}", spawn=None)) for name in TOOL_NAMES
        }
        sim.set_camera_view(eye=[1.20, -1.35, 1.35], target=[0.04, 0.0, 0.73])
        get_current_stage().GetRootLayer().Export(str(output / "scene.usda"))
        (output / "asset_geometry.json").write_text(json.dumps(self.geometry, indent=2) + "\n")
        sim.reset()
        self.drawer_id = self.bench.data.joint_names.index("joint_drawersA_01_joint_00")
        self.initial_tool_poses = {name: obj.data.root_link_pose_w.torch.clone() for name, obj in self.tools.items()}
        self.corners_local = {
            name: torch.tensor(self.geometry[name]["corners"], dtype=torch.float32, device=sim.device)
            for name in TOOL_NAMES
        }
        max_vertices = max(len(vertices) for vertices in self.corners_local.values())
        self.corners_local = {
            name: torch.cat([vertices, vertices[:1].repeat(max_vertices - len(vertices), 1)])
            for name, vertices in self.corners_local.items()
        }
        self.success_monitor = CleanupSuccess()
        self.steps = 0
        self.reset()

    def observe(self) -> dict:
        """Read full tool extents, velocities, drawer opening, and the current success state."""
        corners, bottoms, speeds, angular = [], [], [], []
        q = self.bench.data.joint_pos.torch[0, self.drawer_id].item()
        for name in TOOL_NAMES:
            obj = self.tools[name]
            rotation = matrix_from_quat(obj.data.root_link_quat_w.torch)[0]
            world = self.corners_local[name] @ rotation.T + obj.data.root_link_pos_w.torch[0]
            bottoms.append(world[:, 2].min().item())
            relative = world.clone()
            relative[:, 0] -= q
            corners.append(relative.cpu().numpy())
            speeds.append(obj.data.root_com_lin_vel_w.torch[0].norm().item())
            angular.append(obj.data.root_com_ang_vel_w.torch[0].norm().item())
        return {
            "corners": np.asarray(corners),
            "world_bottoms": np.asarray(bottoms),
            "linear_speeds": np.asarray(speeds),
            "angular_speeds": np.asarray(angular),
            "drawer_q": q,
            "drawer_speed": self.bench.data.joint_vel.torch[0, self.drawer_id].item(),
        }

    def reset(self) -> dict:
        """Restore the initial scene and clear all episode history and external wrenches."""
        for name, obj in self.tools.items():
            obj.write_root_link_pose_to_sim_index(root_pose=self.initial_tool_poses[name])
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.sim.device))
            obj.reset()
            obj.permanent_wrench_composer.reset()
        self.bench.write_joint_position_to_sim_index(position=torch.zeros_like(self.bench.data.joint_pos.torch))
        self.bench.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(self.bench.data.joint_vel.torch))
        self.bench.reset()
        self.steps = 0
        self.sim.forward()
        observation = self.observe()
        self.success_monitor.reset(observation["corners"], observation["drawer_q"])
        return observation

    def step(self, wrenches: dict | None = None, drawer_effort: float = 0.0) -> tuple[dict, bool]:
        """Apply world-frame tool forces/torques and drawer force for one PhysX step.

        ``wrenches`` maps tool names to pairs of three-vectors (force in N, torque
        in Nm). All unspecified tool wrenches are zero. No pose target is written.
        Non-target drawers receive a weak restoring effort to keep them closed.
        """
        for name, obj in self.tools.items():
            obj.permanent_wrench_composer.reset()
            if wrenches and name in wrenches:
                force, torque = wrenches[name]
                obj.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=torch.as_tensor(force, device=self.sim.device).reshape(1, 1, 3),
                    torques=torch.as_tensor(torque, device=self.sim.device).reshape(1, 1, 3),
                    is_global=True,
                )
            obj.write_data_to_sim()
        efforts = -80.0 * self.bench.data.joint_pos.torch - 12.0 * self.bench.data.joint_vel.torch
        efforts[0, self.drawer_id] = float(np.clip(drawer_effort, -70, 70))
        self.bench.set_joint_effort_target_index(target=efforts)
        self.bench.write_data_to_sim()
        self.sim.step(render=False)
        self.bench.update(self.dt)
        for obj in self.tools.values():
            obj.update(self.dt)
        self.steps += 1
        obs = self.observe()
        for value in obs.values():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError("Non-finite PhysX state")
        terminated = self.success_monitor.update(**obs, dt=self.dt)
        return obs, terminated

    def summary(self) -> dict:
        """Return serializable measured task progress."""
        obs = self.observe()
        return {
            "simulation_time_s": self.steps * self.dt,
            "drawer_q_m": obs["drawer_q"],
            "drawer_speed_m_s": obs["drawer_speed"],
            "inside": self.success_monitor.contained(obs["corners"]).tolist(),
            "tool_speeds_m_s": obs["linear_speeds"].tolist(),
            "opened": bool(self.success_monitor.opened),
            "placed_one_by_one": [TOOL_NAMES[i] for i in self.success_monitor.placed],
            "invalid_sequence": bool(self.success_monitor.invalid_sequence),
            "success": bool(self.success_monitor.success),
            "tool_world_bottoms_m": obs["world_bottoms"].tolist(),
        }
