This robot-free evaluation uses **Isaac Sim 6.2 PhysX**, the cached SimReady workbench,
three screwdrivers, two wrenches, a fixed oscilloscope on the left, and a fixed open
socket-set toolbox on the right. The bench includes four articulated drawers and
overhead bins. No cached source USD is modified.

Run from the Isaac Lab checkout:

```bash
# Step 1: settle for 4 s, measure stability for 6 s, and save the scene and preview.
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python \
  scripts/evaluation/workbench_cleanup/run.py --mode scene

# Step 2: force-driven cleanup, video, and physical negative controls.
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python \
  scripts/evaluation/workbench_cleanup/run.py --mode demo

# The same physical validation without recording the trajectory.
OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python \
  scripts/evaluation/workbench_cleanup/run.py --mode validate

# Focused temporal/geometry tests.
uv run --extra isaacsim --extra test python -m pytest \
  scripts/evaluation/workbench_cleanup/test_success.py -q
```

`--assets` overrides `/home/qianl/.cache/isaaclab_arena/workbench_cleanup`.
`--output` overrides `artifacts/workbench_cleanup`; `--device cpu` selects CPU PhysX.
The validated run uses `cuda:0`, a 1/120 s physics timestep, and 1280×720 RTX capture.
The MP4 plays at 20 fps. Isaac Sim must have access to the NVIDIA GPU and its cache.

`WorkbenchCleanupEnv.step(wrenches, drawer_effort)` returns the measured observation
and a success termination boolean. Wrenches contain a world-frame force and torque
for each controlled tool. Drawer force is clipped to ±70 N. The demonstration uses
bounded PD wrenches, including gravity compensation, to lift, transport, lower, and
release one tool at a time. Tools enter through the exposed front of the open drawer
before moving below the tabletop. There are no tool pose or joint position writes
in the demonstration trajectory; writes occur only at reset and in labeled negative
test fixtures after the video ends.

Success requires all of the following in one episode:

- Observe the top-left drawer opening to at least 0.20 m.
- Observe each tool lifting above the tabletop (its lowest hull vertex exceeds 0.705 m).
- Observe five separate arrivals inside the open drawer. Each must settle for 0.20 s
  before another arrival; simultaneous or overlapping arrivals invalidate the episode.
- Keep every complete tool inside the drawer's usable interior, with 2 mm tolerance.
  Bounds use transformed visual convex-hull vertices, including the small wrench's scale.
  They follow the measured drawer translation. Origins alone are insufficient.
- Close the drawer to within 8 mm, with drawer speed below 0.015 m/s and tool linear
  and angular speeds below 0.035 m/s and 0.4 rad/s, continuously for 0.50 s.

Stacked tools are allowed. Reset clears all history, so a preloaded closed drawer is
not success. `success.py` owns this criterion; callers should use the termination
returned by `step` rather than reconstructing it from controller targets.

The source bench's non-watertight SDF meshes are replaced with compound collision
boxes for its frame, tabletop, door, and hollow drawers. Detailed visuals remain
unchanged. The overhead bins are fixed visual scenery. The oscilloscope and toolbox
are fixed assembled distractors with static collision proxies. The toolbox lid is
posed 100° about its authored hinge; reset-transform stacks in nested parts are
rebased before placement. This evaluation does not assess robot reachability,
grasp detection, or a learned policy. Pickup is inferred from measured tool motion.

Measured results, video, scene USD, per-step usage/LOC accounting, and validation logs
are in [the evaluation report](docs/README.md).
`scene.usda` references the local cached assets and therefore needs that cache to open.

The focused tests protect distinct observable contracts: complete sequence, stable
closure, missing/protruding/moving tools, open drawer, preloaded goal state, invalid
pickup/opening order, batch arrivals, episode reset, and valid stacking. Physical
negative controls reuse actual final placements and history so a missing-history
guard cannot mask a geometry or closure failure. They test a missing tool, an open
drawer with all five tools, and a preloaded goal state without task history.
