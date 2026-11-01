# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose cached SimReady USDs without changing the user's asset cache."""

import itertools
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from success import TOOL_NAMES

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def reference(stage: Usd.Stage, path: str, usd: Path, position=(0.0, 0.0, 0.0), yaw=0.0, scale=1.0):
    """Reference an asset with a metre-based placement transform."""
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().GetReferences().AddReference(str(usd))
    # Instance proxies must become editable before authoring physics overrides.
    for prim in Usd.PrimRange(root.GetPrim()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    # ForgeCAD parts author some children in stage coordinates with resetXformStack.
    # Rebase those before placement so the entire assembly follows the wrapper pose.
    cache = UsdGeom.XformCache()
    rebased = []
    for prim in Usd.PrimRange(root.GetPrim()):
        xform = UsdGeom.Xformable(prim)
        if xform and xform.GetResetXformStack():
            local = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            rebased.append((xform, local))
    for xform, local in rebased:
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(False)
        xform.AddTransformOp(opSuffix="cleanupLocal").Set(local)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    root.AddRotateZOp().Set(yaw)
    root.AddScaleOp().Set(Gf.Vec3f(scale))
    return root.GetPrim()


def static_prop(stage: Usd.Stage, root: Usd.Prim) -> None:
    """Keep a detailed prop's appearance while making its assembled pieces static."""
    for prim in list(Usd.PrimRange(root)):
        if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
        for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, PhysxSchema.PhysxCollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)


def box(stage: Usd.Stage, path: str, center, size) -> None:
    """Author an invisible PhysX collision box."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()).CreateContactOffsetAttr(0.001)


def open_case(stage: Usd.Stage, root: Usd.Prim) -> None:
    """Pose the source case lid about its authored hinge before freezing the prop."""
    joint_prim = next(p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() == "case_lid")
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    cache = UsdGeom.XformCache()
    parent_world = cache.GetLocalToWorldTransform(body0)
    pivot = parent_world.Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
    axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[joint.GetAxisAttr().Get()]
    rotation0 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
    axis = parent_world.TransformDir(rotation0.TransformDir(axis)).GetNormalized()
    delta = Gf.Matrix4d().SetTranslate(-pivot) * Gf.Matrix4d().SetRotate(Gf.Rotation(axis, 100.0))
    delta *= Gf.Matrix4d().SetTranslate(pivot)
    local = (
        cache.GetLocalToWorldTransform(body1) * delta * cache.GetLocalToWorldTransform(body1.GetParent()).GetInverse()
    )
    xform = UsdGeom.Xformable(body1)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(local)


def bench_colliders(stage: Usd.Stage, root: Usd.Prim) -> None:
    """Replace defective source SDFs with compound boxes, keeping drawer cavities open."""
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for prim in Usd.PrimRange(root):
        for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, PhysxSchema.PhysxCollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    cache = UsdGeom.XformCache()
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for body in bodies:
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        shapes = []
        if body.GetName() == "table_01_obj_00":
            shapes = [
                ((-0.009, 0, 0.647), (0.52, 1.50, 0.044)),
                ((-0.115, -0.524, 0.33), (0.014, 0.35, 0.57)),
                ((0.04, -0.708, 0.33), (0.38, 0.012, 0.57)),
                ((0.04, -0.340, 0.33), (0.38, 0.012, 0.57)),
            ]
            shapes += [((x, y, 0.32), (0.035, 0.035, 0.64)) for x in (-0.23, 0.22) for y in (-0.73, 0.73)]
        elif body.GetName().startswith("drawers"):
            bounds = bounds_cache.ComputeWorldBound(body).ComputeAlignedRange()
            lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
            low_z, high_z = lo[2] + 0.004, hi[2] - 0.004
            middle_z = (low_z + high_z) / 2
            shapes = [
                ((0.086, -0.524, low_z), (0.300, 0.334, 0.004)),
                ((-0.062, -0.524, middle_z), (0.004, 0.334, high_z - low_z)),
                ((0.234, -0.524, middle_z), (0.004, 0.334, high_z - low_z)),
                ((0.086, -0.689, middle_z), (0.300, 0.004, high_z - low_z)),
                ((0.086, -0.359, middle_z), (0.300, 0.004, high_z - low_z)),
            ]
        elif body.GetName().startswith("door_"):
            shapes = [((0.243, 0.524, 0.382), (0.014, 0.342, 0.423))]
        for index, (center, size) in enumerate(shapes):
            local_center = inverse.Transform(Gf.Vec3d(*center))
            box(stage, str(body.GetPath()) + f"/CollisionProxy_{index}", local_center, size)


def compose(stage: Usd.Stage, asset_root: Path) -> dict:
    """Compose the bench, fixed accessories, and five dynamic tools; return tool geometry."""
    bench = reference(stage, "/World/Workbench", asset_root / "bench.usda")
    # Joint relationships in the source point at collision meshes. Use their owning rigid bodies.
    for prim in Usd.PrimRange(bench):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for relation in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                targets = relation.GetTargets()
                if targets:
                    owner = stage.GetPrimAtPath(targets[0])
                    while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                        owner = owner.GetParent()
                    if owner:
                        relation.SetTargets([owner.GetPath()])
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and "drawers" in prim.GetName():
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.5)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI(prim).CreateSolverPositionIterationCountAttr(16)
    bench_colliders(stage, bench)
    # Add a full support plane aligned to the authored tabletop (the cached clutter support is tiny).
    box(stage, "/World/TableSupport", (-0.009, 0.0, 0.6665), (0.52, 1.50, 0.005))
    scope = reference(stage, "/World/Oscilloscope", asset_root / "oscilloscope.usda", (-0.04, -0.50, 0.669))
    static_prop(stage, scope)
    box(stage, "/World/ScopeCollision", (-0.047, -0.50, 0.78), (0.15, 0.32, 0.22))
    case = reference(stage, "/World/Toolbox", asset_root / "socket_set.usda", (0.02, 0.535, 0.669), 90.0, 0.9)
    open_case(stage, case)
    static_prop(stage, case)
    box(stage, "/World/ToolboxCollision", (0.02, 0.535, 0.682), (0.15, 0.30, 0.025))
    files = [
        "tool_sm_handTool_screwdriver_slotted_55mm_a01_01.usda",
        "tool_sm_handTool_screwdriver_torx_t10_a01_01.usda",
        "tool_sm_handTool_screwdriver_phillips_ph2_a01_01.usda",
        "tool_sm_handTool_wrench_combination_a01_01.usda",
        "tool_sm_handTool_wrench_combination_a01_01.usda",
    ]
    placements = [
        (-0.08, -0.16, 0.680, 20),
        (-0.07, -0.075, 0.685, -15),
        (-0.08, 0.015, 0.685, 12),
        (0.015, 0.09, 0.685, -8),
        (0.025, -0.02, 0.735, 65),
    ]
    geometry = {}
    for name, file, placement in zip(TOOL_NAMES, files, placements):
        scale = 0.75 if name == "small_wrench" else 1.0
        root = reference(stage, f"/World/Tools/{name}", asset_root / file, placement[:3], placement[3], scale)
        bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise ValueError(f"Expected one tool body for {name}, got {len(bodies)}")
        body = bodies[0]
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.12 if "screwdriver" in name else 0.18 * scale**3)
        props = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        props.CreateLinearDampingAttr(0.15)
        props.CreateAngularDampingAttr(0.3)
        props.CreateEnableCCDAttr(False)
        for prim in Usd.PrimRange(body):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        # Bound the complete visual tool in the rigid-link frame. Transform with live body poses later.
        cache = UsdGeom.XformCache()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bounds = bbox.ComputeWorldBound(body).ComputeAlignedRange()
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        corners = [inverse.Transform(Gf.Vec3d(*p)) for p in itertools.product(*zip(bounds.GetMin(), bounds.GetMax()))]
        # Use the actual local bounds, not a world-AABB inverse (which overbounds yawed tools).
        points = []
        for prim in Usd.PrimRange(body):
            if prim.IsA(UsdGeom.Mesh):
                transform = cache.GetLocalToWorldTransform(prim) * inverse
                points.extend(transform.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        points = np.asarray(points if points else corners)
        # Use convex-hull vertices: a rotated bounding box substantially overbounds a
        # screwdriver's thin shaft and can falsely place its empty corners below a table.
        corners = points[ConvexHull(points).vertices]
        body_world = cache.GetLocalToWorldTransform(body)
        body_scale = np.array(
            [body_world.TransformDir(Gf.Vec3d(*(int(i == j) for i in range(3)))).GetLength() for j in range(3)]
        )
        corners *= body_scale
        geometry[name] = {"body_path": str(body.GetPath()), "corners": corners.tolist()}
    return geometry
