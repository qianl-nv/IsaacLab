# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose, validate, and demonstrate the robot-free workbench cleanup in PhysX."""

import argparse
import json
import time
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.app import launch_simulation
from isaaclab.sim import SimulationCfg


def main():
    """Launch the requested evaluation stage with a bounded execution budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["scene", "demo", "validate"], default="scene")
    parser.add_argument("--assets", type=Path, default=Path.home() / ".cache/isaaclab_arena/workbench_cleanup")
    parser.add_argument("--output", type=Path, default=Path("artifacts/workbench_cleanup"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cfg = SimulationCfg(
        device=args.device,
        dt=1 / 120,
        physics=PhysxCfg(),
        render_interval=6,
        visualizer_cfgs=[
            KitVisualizerCfg(
                headless=True, window_width=1280, window_height=720, eye=(1.85, -1.75, 1.65), lookat=(0.04, 0.0, 0.70)
            )
        ],
    )
    with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
        import numpy as np
        from environment import WorkbenchCleanupEnv
        from PIL import Image

        from isaaclab.sim import SimulationContext

        sim = SimulationContext(cfg)
        env = WorkbenchCleanupEnv(sim, args.assets, args.output)
        try:
            print("SCENE_READY", env.summary(), flush=True)
            if args.mode == "scene":
                # Four seconds of settling followed by six seconds without control forces.
                for _ in range(480):
                    env.step()
                reference = {name: obj.data.root_link_pos_w.torch.clone() for name, obj in env.tools.items()}
                peak_speed = 0.0
                for _ in range(720):
                    obs, terminated = env.step()
                    peak_speed = max(peak_speed, float(obs["linear_speeds"].max()))
                    assert not terminated, "An untouched cluttered scene must never report success"
                displacement = {
                    name: float((obj.data.root_link_pos_w.torch - reference[name]).norm().item())
                    for name, obj in env.tools.items()
                }
                obs = env.observe()
                assert max(displacement.values()) < 0.005, displacement
                assert peak_speed < 0.05, peak_speed
                assert np.all((obs["world_bottoms"] > 0.66) & (obs["world_bottoms"] < 0.78)), env.summary()
                assert abs(obs["drawer_q"]) < 0.008, env.summary()
                for _ in range(12):
                    sim.render()
                image = sim.visualizers[0].render_rgb_array()
                for _ in range(8):
                    sim.render()
                    image = sim.visualizers[0].render_rgb_array()
                Image.fromarray(image).save(args.output / "scene.png")
                report = {
                    "passed": True,
                    "physics": "Isaac Sim PhysX",
                    "settle_s": 4,
                    "observe_s": 6,
                    "tool_drift_m": displacement,
                    "peak_observed_speed_m_s": peak_speed,
                    **env.summary(),
                    "wall_time_s": time.time() - start,
                }
                (args.output / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
                print("STABILITY_PASS", json.dumps(report), flush=True)
            else:
                from demo import ForceDemo

                ForceDemo(env, record=not args.no_video and args.mode == "demo").run(validate=True)
        finally:
            sim.stop()
            sim.clear_instance()


if __name__ == "__main__":
    main()
