# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded wrench demonstration and physical negative controls for cleanup success."""

import copy
import json
import math
import subprocess
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from success import TOOL_NAMES

from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
)


class ForceDemo:
    """Apply a virtual force operator, release every tool, and assert measured task completion."""

    def __init__(self, env, record=True):
        self.env = env
        self.record = record
        self.holds = {}
        self.drawer_target = 0.0
        self.phase = "Settle the initial clutter"
        self.events = []
        self.writer = None
        self.frames = 0
        self.max_force = 0.0
        self.max_torque = 0.0
        self.started = time.time()

    def capture(self, filename=None):
        """Capture actual Kit/RTX output and add a measured-progress caption."""
        self.env.sim.render()
        pixels = self.env.sim.visualizers[0].render_rgb_array()
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 58), fill=(15, 23, 32))
        draw.text((20, 10), "PHYSX WORKBENCH CLEANUP | Force operator (no robot)", fill="white")
        draw.text((20, 32), self.phase, fill=(175, 220, 255))
        summary = self.env.summary()
        draw.rectangle((0, image.height - 34, image.width, image.height), fill=(15, 23, 32))
        draw.text(
            (20, image.height - 23),
            f"Placed: {len(summary['placed_one_by_one'])}/5   Drawer: {summary['drawer_q_m']:.3f} m"
            f"   Success: {summary['success']}   Sim time: {summary['simulation_time_s']:.1f} s",
            fill="white",
        )
        if filename:
            image.save(self.env.output / filename)
        if self.record:
            if self.writer is None:
                self.writer = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{image.width}x{image.height}",
                        "-r",
                        "20",
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(self.env.output / "cleanup_demo.mp4"),
                    ],
                    stdin=subprocess.PIPE,
                )
            self.writer.stdin.write(np.asarray(image).tobytes())
            self.frames += 1

    def mark(self, phase):
        """Log a stage boundary using measured state, not intended controller targets."""
        self.phase = phase
        event = {"phase": phase, **self.env.summary()}
        self.events.append(event)
        print("PHASE", json.dumps(event), flush=True)
        (self.env.output / "events.json").write_text(json.dumps(self.events, indent=2) + "\n")
        if self.record:
            self.capture(f"phase_{len(self.events):02d}.png")

    def step(self, seconds):
        """Step PhysX with bounded PD wrenches, gravity compensation, and joint efforts."""
        for _ in range(round(seconds / self.env.dt)):
            commands = {}
            for name, (position, quaternion) in self.holds.items():
                obj = self.env.tools[name]
                p = obj.data.root_link_pos_w.torch
                q = obj.data.root_link_quat_w.torch
                v = obj.data.root_com_lin_vel_w.torch
                omega = obj.data.root_com_ang_vel_w.torch
                mass = obj.data.body_mass.torch.sum().item()
                gain = 65.0
                damping = 2 * math.sqrt(gain * mass)
                force = gain * (position - p) - damping * v
                force[:, 2] += mass * 9.81
                force = force.clamp(-25.0, 25.0)
                _, error = compute_pose_error(p, q, position, quaternion)
                inertia = obj.data.body_inertia.torch.reshape(1, 3, 3)
                acceleration = quat_apply_inverse(q, 180 * error - 28 * omega)
                torque = quat_apply(q, torch.bmm(inertia, acceleration[:, :, None]).squeeze(-1)).clamp(-0.4, 0.4)
                commands[name] = (force[0], torque[0])
                self.max_force = max(self.max_force, force.norm().item())
                self.max_torque = max(self.max_torque, torque.norm().item())
            q = self.env.bench.data.joint_pos.torch[0, self.env.drawer_id].item()
            qd = self.env.bench.data.joint_vel.torch[0, self.env.drawer_id].item()
            effort = np.clip(300 * (self.drawer_target - q) - 50 * qd, -70, 70)
            self.env.step(commands, float(effort))
            if self.record and self.env.steps % 6 == 0:
                self.capture()

    def move(self, name, position, quaternion=None, seconds=1.7):
        """Move a force-controller target smoothly; never write object poses at runtime."""
        obj = self.env.tools[name]
        start = obj.data.root_link_pos_w.torch.clone()
        goal = torch.tensor([position], device=self.env.sim.device, dtype=torch.float32)
        if quaternion is None:
            quaternion = self.holds[name][1] if name in self.holds else obj.data.root_link_quat_w.torch.clone()
        count = round(seconds / self.env.dt)
        for i in range(count):
            alpha = (i + 1) / count
            alpha = alpha * alpha * (3 - 2 * alpha)
            self.holds[name] = ((1 - alpha) * start + alpha * goal, quaternion)
            self.step(self.env.dt)
        self.step(0.25)
        error = (obj.data.root_link_pos_w.torch - goal).norm().item()
        assert error < 0.025, f"Force controller blocked for {name}: position error {error} m"

    def drawer(self, goal, seconds=2.0):
        """Open or close with a bounded joint-force controller."""
        start = self.env.observe()["drawer_q"]
        count = round(seconds / self.env.dt)
        for i in range(count):
            self.drawer_target = start + (goal - start) * (i + 1) / count
            self.step(self.env.dt)
        self.step(0.6)
        assert abs(self.env.observe()["drawer_q"] - goal) < 0.012, self.env.summary()

    def run(self, validate=False):
        """Complete the task and optionally probe physical false-success scenarios."""
        try:
            self.step(4.0)
            self.mark("Initial clutter: success must be false")
            assert not self.env.success_monitor.success
            self.mark("Open the top-left drawer")
            self.drawer(0.245)
            assert not self.env.success_monitor.success
            zero = torch.zeros(1, device=self.env.sim.device)
            tool_q = quat_from_euler_xyz(zero, zero, zero + math.pi / 2)
            rotation = matrix_from_quat(tool_q)[0]
            for index, name in enumerate(TOOL_NAMES):
                self.mark(f"Pick and store tool {index + 1}/5: {name}")
                p = self.env.tools[name].data.root_link_pos_w.torch[0].tolist()
                self.move(name, [p[0], p[1], 0.86], seconds=1.4)
                vertices = self.env.corners_local[name]
                center = (vertices.min(dim=0).values + vertices.max(dim=0).values) / 2
                offset = (rotation @ center).cpu().numpy()
                drawer_q = self.env.observe()["drawer_q"]
                target_center = np.array([-0.025 + index * 0.050 + drawer_q, -0.524, 0.70])
                target = target_center - offset
                entry = target.copy()
                entry[0] = 0.44 - offset[0]
                # Clear the tabletop's front edge before descending into the open drawer.
                self.move(name, entry.tolist(), tool_q, seconds=1.6)
                rotated = self.env.corners_local[name] @ rotation.T
                entry[2] = 0.550 - rotated[:, 2].min().item()
                self.move(name, entry.tolist(), tool_q, seconds=1.4)
                target[2] = entry[2]
                self.move(name, target.tolist(), tool_q, seconds=1.2)
                target[2] = 0.520 - rotated[:, 2].min().item()
                self.move(name, target.tolist(), tool_q, seconds=0.7)
                self.holds.pop(name)
                self.env.tools[name].permanent_wrench_composer.reset()
                self.step(1.2)
                self.mark(f"Released tool {index + 1}/5 in drawer")
                assert len(self.env.success_monitor.placed) == index + 1, self.env.summary()
                assert not self.env.success_monitor.success, "Open or incompletely loaded drawer reported success"
            self.mark("All five inside, drawer still open: success must be false")
            self.before_close = copy.deepcopy(self.env.success_monitor)
            self.mark("Close the loaded drawer")
            self.drawer(0.0, seconds=3.0)
            self.step(1.0)
            self.mark("Cleanup completed: five tools stored and drawer closed")
            assert self.env.success_monitor.success, self.env.summary()
            final = self.env.summary()
            self.capture("completed.png")
            if validate:
                # Reset fixtures belong to validation, outside the completed-task video.
                self.record = False
                negative = self.negative_scenarios()
            else:
                negative = []
            report = {
                "passed": True,
                "mode": "bounded_world_wrenches_and_joint_efforts",
                "physics": "Isaac Sim PhysX",
                "runtime_pose_writes": 0,
                "max_tool_force_norm_N": self.max_force,
                "max_tool_torque_norm_Nm": self.max_torque,
                "video_frames": self.frames,
                "wall_time_s": time.time() - self.started,
                "final": final,
                "events": self.events,
                "negative_scenarios": negative,
            }
            (self.env.output / "demo_validation.json").write_text(json.dumps(report, indent=2) + "\n")
            if validate:
                (self.env.output / "negative_validation.json").write_text(json.dumps(negative, indent=2) + "\n")
            print("DEMO_PASS", json.dumps(report), flush=True)
        finally:
            if self.writer:
                self.writer.stdin.close()
                assert self.writer.wait() == 0, "ffmpeg failed"

    def negative_scenarios(self):
        """Use labeled reset fixtures to test final guards after genuine successful placements.

        These fixture writes are outside the force demonstration. Reuse the completed
        scene and its real history so missing-history rejection cannot mask geometry bugs.
        """
        final_poses = {name: obj.data.root_link_pose_w.torch.clone() for name, obj in self.env.tools.items()}
        history = copy.deepcopy(self.env.success_monitor)
        results = []
        for scenario in ("one_tool_missing", "drawer_left_open", "preloaded_without_history"):
            self.env.reset()
            self.holds.clear()
            for name, obj in self.env.tools.items():
                pose = final_poses[name].clone()
                if scenario == "one_tool_missing" and name == TOOL_NAMES[-1]:
                    pose = self.env.initial_tool_poses[name].clone()
                if scenario == "drawer_left_open":
                    pose[:, 0] += 0.245
                obj.write_root_link_pose_to_sim_index(root_pose=pose)
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.env.sim.device))
            self.drawer_target = 0.245 if scenario == "drawer_left_open" else 0.0
            q = torch.zeros_like(self.env.bench.data.joint_pos.torch)
            q[0, self.env.drawer_id] = self.drawer_target
            self.env.bench.write_joint_position_to_sim_index(position=q)
            self.env.sim.forward()
            if scenario != "preloaded_without_history":
                self.env.success_monitor = copy.deepcopy(history)
                self.env.success_monitor.closed_time = 0.0
                self.env.success_monitor.success = False
            else:
                obs = self.env.observe()
                self.env.success_monitor.reset(obs["corners"], obs["drawer_q"])
            ever_success = False
            for _ in range(240):
                self.step(self.env.dt)
                ever_success |= self.env.success_monitor.success
            assert not ever_success, f"False success in {scenario}"
            results.append({"scenario": scenario, "ever_success": bool(ever_success), "final": self.env.summary()})
            print("NEGATIVE_PASS", json.dumps(results[-1]), flush=True)
        return results
