# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.cloner import add as clone_add
from isaaclab.utils import configclass, find_unique_string_name

if TYPE_CHECKING:
    from isaaclab.assets import AssetBaseCfg

    from .interactive_scene import InteractiveScene


@configclass
class InteractiveSceneCfg:
    """Configuration for the interactive scene.

    The users can inherit from this class to add entities to their scene. This is then parsed by the
    :class:`InteractiveScene` class to create the scene.

    .. note::
        The adding of entities to the scene is sensitive to the order of the attributes in the configuration.
        Please make sure to add the entities in the order you want them to be added to the scene.
        The recommended order of specification is terrain, physics-related assets (articulations and rigid bodies),
        sensors and non-physics-related assets (lights).

    For example, to add a robot to the scene, the user can create a configuration class as follows:

    .. code-block:: python

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors.ray_caster import GridPatternCfg, RayCasterCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # terrain - flat terrain plane
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
            )

            # articulation - robot 1
            robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
            # articulation - robot 2
            robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
            robot_2.init_state.pos = (0.0, 1.0, 0.6)

            # sensor - ray caster attached to the base of robot 1 that scans the ground
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
            )

            # extras - light
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
            )

    """

    class_type: type[InteractiveScene] | str = "{DIR}.interactive_scene:InteractiveScene"
    """The class to use for the interactive scene.

    Defaults to :class:`isaaclab.scene.InteractiveScene`.
    """

    num_envs: int = MISSING
    """Number of environment instances handled by the scene."""

    env_spacing: float = MISSING
    """Spacing between environments.

    This is the default distance between environment origins in the scene. Used only when the
    number of environments is greater than one.
    """

    lazy_sensor_update: bool = True
    """Whether to update sensors only when they are accessed. Default is True.

    If true, the sensor data is only updated when their attribute ``data`` is accessed. Otherwise, the sensor
    data is updated every time sensors are updated.
    """

    replicate_physics: bool = True
    """Enable/disable replication of physics schemas when using the Cloner APIs. Default is True.

    If True, the simulation will have the same asset instances (USD prims) in all the cloned environments.
    Internally, this ensures optimization in setting up the scene and parsing it via the physics stage parser.

    If False, the simulation allows having separate asset instances (USD prims) in each environment.
    This flexibility comes at a cost of slowdowns in setting up and parsing the scene.

    .. note::
        Optimized parsing of certain prim types (such as deformable objects) is not currently supported
        by the physics engine. In these cases, this flag needs to be set to False.

    .. attention::
        Setting this flag to False is currently not supported on the Newton physics backend:
        Newton discovers the scene through its replication path, which stage parsing cannot
        replace for cloned environments.

    .. note::
        The scene pipes this flag into :attr:`~isaaclab.cloner.CloneCfg.replicate_physics`;
        the policy is applied by :func:`~isaaclab.cloner.replicate`. Direct workflows that
        call :func:`~isaaclab.cloner.replicate` themselves pass ``replicate_physics``
        explicitly.
    """

    filter_collisions: bool = True
    """Enable/disable collision filtering between cloned environments. Default is True.

    If True, collisions will not occur between cloned environments.

    If False, the simulation will generate collisions between environments.

    .. note::
        Collisions can only be filtered automatically in direct workflows when physics replication is enabled.
        If :attr:`replicated_physics` is ``False`` and collision filtering is desired, make sure to call
        ``scene.filter_collisions()``.
    """

    clone_in_fabric: bool = False
    """Deprecated legacy Fabric cloning flag. Default is False.

    Queued replication no longer forwards this flag to the PhysX replicator;
    ``useFabricForReplication`` is always ``False``.
    """

    clone_cfg: CloneCfg = CloneCfg()
    """Clone execution and legal scene-combination configuration."""


def add(
    target: InteractiveSceneCfg,
    source: InteractiveSceneCfg | object | None,
    *,
    asset_skip: Callable[[AssetBaseCfg], bool] | None = None,
    mode: Literal["clone", "merge"] = "clone",
    on_conflict: Literal["error", "replace"] = "error",
    bindings: MutableMapping[str, str] | None = None,
) -> InteractiveSceneCfg:
    """Add a source scene to ``target``.

    In ``"clone"`` mode, ``target`` accumulates heterogeneous clone
    combinations. Both scenes contribute spawned
    :class:`~isaaclab.assets.AssetBaseCfg` fields at literal
    ``{ENV_REGEX_NS}/Leaf`` roots. An asset equal to an existing binding reuses
    it; other assets receive a unique field name and prim path. Sensors,
    terrain importers, rigid-object collections, spawnless assets, and global
    assets do not participate. This is the default for backward compatibility.

    In ``"merge"`` mode, every declared source field is added to the same scene.
    This mode supports component-based scene construction: assets, sensors,
    terrain, collections, global entities, and scene execution settings remain
    available together. Source fields replace matching target fields only when
    ``on_conflict="replace"``. The optional ``bindings`` mapping records each
    source field's final target field name.

    Args:
        target: Scene that accumulates the fold.
        source: Scene or scene fragment to add. ``None`` is ignored in merge mode.
        asset_skip: Optional predicate called with each spawned asset
            configuration in clone mode. An asset is omitted when the predicate
            returns :data:`True`.
        mode: ``"clone"`` for heterogeneous clone composition or ``"merge"``
            for component composition.
        on_conflict: How merge mode handles a non-None target field with the
            same name. Defaults to ``"error"``.
        bindings: Optional mutable mapping updated from source field names to
            their final names in ``target``.

    Returns:
        The mutated ``target`` scene.

    Raises:
        ValueError: If an option is invalid, a merge field conflicts, or clone
            composition receives an unsupported scene.
    """
    if mode == "merge":
        if asset_skip is not None:
            raise ValueError("asset_skip applies only to clone mode.")
        return _merge(target, source, on_conflict=on_conflict, bindings=bindings)
    if mode != "clone":
        raise ValueError(f"Unsupported scene add mode: {mode!r}.")
    if source is None or not isinstance(source, InteractiveSceneCfg):
        raise ValueError("clone mode requires an InteractiveSceneCfg source.")
    if on_conflict != "error":
        raise ValueError("on_conflict applies only to merge mode.")

    if source.clone_cfg.clone_combinations:
        raise ValueError("the added scene must not declare clone combinations; fold it as the base scene instead.")

    target_assets = _scene_assets(target, asset_skip)
    source_assets = _scene_assets(source, asset_skip)
    if not target_assets or not source_assets:
        raise ValueError("both scenes must contain at least one spawned environment asset.")

    # ``target`` accumulates only the participating assets: clear everything else
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    target_names = {name for name, _ in target_assets}
    for name in [
        n for n, v in vars(target).items() if n not in base_fields and v is not None and n not in target_names
    ]:
        setattr(target, name, None)

    # an empty clone configuration is homogeneous: make the base combination explicit
    if not target.clone_cfg.clone_combinations:
        clone_add(target.clone_cfg, InclusionSet(assets=[name for name, _ in target_assets]))

    # a duplicate is an asset equal to an existing binding; each binding is
    # reused at most once per call so distinct twins survive
    available = dict(target_assets)
    added_names: list[str] = []
    used_names = set(dir(target))
    paths = {cfg.prim_path for _, cfg in target_assets}
    for source_name, cfg in source_assets:
        target_name = next((name for name, existing in available.items() if existing == cfg), None)
        if target_name is not None:
            del available[target_name]
        else:
            target_name = find_unique_string_name(source_name, lambda name: name not in used_names)
            if cfg.prim_path in paths:
                cfg = cfg.replace(prim_path=find_unique_string_name(cfg.prim_path, lambda path: path not in paths))
            setattr(target, target_name, cfg)
            used_names.add(target_name)
            paths.add(cfg.prim_path)
        added_names.append(target_name)

    clone_add(target.clone_cfg, InclusionSet(assets=list(dict.fromkeys(added_names))))
    if bindings is not None:
        bindings.update(zip((name for name, _ in source_assets), added_names))
    return target


def _merge(
    target: InteractiveSceneCfg,
    source: object | None,
    *,
    on_conflict: Literal["error", "replace"],
    bindings: MutableMapping[str, str] | None,
) -> InteractiveSceneCfg:
    """Merge one component scene into a common target scene."""
    if source is None:
        return target
    if on_conflict not in ("error", "replace"):
        raise ValueError(f"Unsupported scene field conflict policy: {on_conflict!r}.")

    base_fields = InteractiveSceneCfg.__dataclass_fields__
    for name, value in vars(source).items():
        if name.startswith("_") or value is None:
            continue
        target_value = getattr(target, name, None)
        if (
            target_value is not None
            and target_value is not value
            and name not in base_fields
            and on_conflict == "error"
        ):
            raise ValueError(f"Scene field '{name}' is already defined.")
        setattr(target, name, value)
        if bindings is not None and name not in base_fields:
            bindings[name] = name
    return target


def _scene_assets(
    scene_cfg: InteractiveSceneCfg,
    asset_skip: Callable[[AssetBaseCfg], bool] | None,
) -> list[tuple[str, AssetBaseCfg]]:
    """List the environment-scoped spawned assets of one scene."""
    # Deferred: config construction must stay importable without isaaclab.assets
    # (kitless factories); only scene composition pays for it.
    from isaaclab.assets import AssetBaseCfg  # noqa: PLC0415

    env_root = "{ENV_REGEX_NS}/"
    base_fields, assets = InteractiveSceneCfg.__dataclass_fields__, []
    for name, value in vars(scene_cfg).items():
        # only spawned assets compose; sensors, terrains, and collections are not AssetBaseCfg
        if name in base_fields or not isinstance(value, AssetBaseCfg) or value.spawn is None:
            continue
        if asset_skip is not None and asset_skip(value):
            continue
        path = value.prim_path
        if not (isinstance(path, str) and path.startswith(env_root) and path[len(env_root) :].isidentifier()):
            raise ValueError(
                f"scene composition requires assets at literal '{{ENV_REGEX_NS}}/Leaf' roots; {name!r} uses"
                f" {path!r}. Skip global assets with asset_skip and attach shared world assets to the composed scene."
            )
        assets.append((name, value))
    return assets
