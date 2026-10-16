# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`isaaclab_tasks.utils.parse_cfg.parse_env_cfg`."""

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import materialize_env_cfg, parse_env_cfg, resolve_task_config


class _SourceCfg:
    def __init__(self, value: int):
        self.value = value


class _MaterializedCfg:
    def __init__(self, value: int):
        self.value = value


def _replace_source_cfg(source_cfg: _SourceCfg) -> _MaterializedCfg:
    return _MaterializedCfg(source_cfg.value)


def test_parse_env_cfg_rejects_bare_string_overrides():
    """A bare string is itself a ``Sequence[str]`` of characters; catch it explicitly."""
    with pytest.raises(TypeError, match="bare string"):
        parse_env_cfg("Isaac-Cartpole", overrides="physics=isaacsim_physx")


def test_parse_env_cfg_accepts_list_overrides():
    """A properly wrapped override list should apply without error."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", overrides=["physics=isaacsim_physx"])
    assert env_cfg is not None


def test_parse_env_cfg_preserves_task_device_when_omitted():
    """An omitted device should preserve a task-specific simulation requirement."""
    env_cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow")

    assert env_cfg.sim.device == "cpu"
    env_cfg.validate()


def test_parse_env_cfg_applies_explicit_device_override():
    """An explicit device should continue to override the registered task default."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", device="cpu")

    assert env_cfg.sim.device == "cpu"


def test_parse_env_cfg_materializes_after_explicit_overrides():
    """Materializers observe final parser overrides and run once."""
    task_name = "Isaac-Test-Env-Cfg-Materializer"
    source_cfg = parse_env_cfg("Isaac-Cartpole")
    observed_num_envs = []

    def materializer(cfg):
        observed_num_envs.append(cfg.scene.num_envs)
        cfg.decimation = 17
        return cfg

    gym.register(
        id=task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": source_cfg,
            "env_cfg_materializer_entry_point": materializer,
        },
    )
    try:
        env_cfg = parse_env_cfg(task_name, num_envs=7)
        assert env_cfg.scene.num_envs == 7
        assert env_cfg.decimation == 17
        assert materialize_env_cfg(task_name, env_cfg) is env_cfg
        assert observed_num_envs == [7]
    finally:
        del gym.registry[task_name]


def test_materialize_env_cfg_loads_import_string_and_accepts_replacement():
    """A materializer may be an import string and return a replacement configuration."""
    task_name = "Isaac-Test-Env-Cfg-Import-String-Materializer"
    gym.register(
        id=task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_materializer_entry_point": f"{__name__}:_replace_source_cfg"},
    )
    try:
        source_cfg = _SourceCfg(23)
        materialized_cfg = materialize_env_cfg(task_name, source_cfg)

        assert isinstance(materialized_cfg, _MaterializedCfg)
        assert materialized_cfg.value == 23
        assert materialized_cfg._isaaclab_materialized is True
        assert materialize_env_cfg(task_name, materialized_cfg) is materialized_cfg
    finally:
        del gym.registry[task_name]


def test_resolve_task_config_tracks_explicit_source_overrides_before_materialization():
    """Deferred materialization retains explicit intent, including values equal to defaults."""
    task_name = "Isaac-Test-Env-Cfg-Explicit-Overrides"
    source_cfg = parse_env_cfg("Isaac-Cartpole")
    default_decimation = source_cfg.decimation
    gym.register(
        id=task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": source_cfg},
    )
    try:
        resolved_cfg, _ = resolve_task_config(
            task_name,
            None,
            overrides=[f"env.decimation={default_decimation}"],
            materialize=False,
        )

        assert resolved_cfg.decimation == default_decimation
        assert resolved_cfg.__isaaclab_explicit_overrides__ == ("decimation",)
        assert not hasattr(resolved_cfg, "_isaaclab_materialized")
    finally:
        del gym.registry[task_name]


@pytest.mark.parametrize("missing_module", ["pinocchio", "pink", "qpsolvers", "daqp", "unrelated_dependency"])
def test_task_config_missing_pink_dependency(monkeypatch: pytest.MonkeyPatch, missing_module: str):
    """Missing Pink dependencies should identify the task without hiding unrelated import errors."""
    from isaaclab_tasks.utils import parse_cfg

    task_name = "IsaacContrib-PickPlace-GR1T2-Abs"
    original_error = ModuleNotFoundError(f"No module named '{missing_module}'", name=missing_module)

    def import_missing_dependency(name: str):
        raise original_error

    monkeypatch.setattr(parse_cfg.importlib, "import_module", import_missing_dependency)
    if missing_module == "unrelated_dependency":
        with pytest.raises(ModuleNotFoundError) as exc_info:
            parse_cfg.load_cfg_from_registry(task_name, "env_cfg_entry_point")
        assert exc_info.value is original_error
    else:
        with pytest.raises(ImportError, match=f"{task_name}.*Pink IK requires Linux x86_64 or aarch64") as exc_info:
            parse_cfg.load_cfg_from_registry(task_name, "env_cfg_entry_point")
        assert exc_info.value.__cause__ is original_error


def test_pink_task_checks_lazy_dependencies(monkeypatch: pytest.MonkeyPatch):
    """A real Pink task must report missing dependencies even when its config imports lazily."""
    from isaaclab_tasks.utils import parse_cfg

    task_name = "IsaacContrib-PickPlace-GR1T2-Abs"
    original_find_spec = parse_cfg.importlib.util.find_spec

    def without_pinocchio(name: str, *args, **kwargs):
        return None if name == "pinocchio" else original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(parse_cfg.importlib.util, "find_spec", without_pinocchio)
    with pytest.raises(ImportError, match=f"{task_name}.*Pink IK requires Linux x86_64 or aarch64"):
        parse_cfg.load_cfg_from_registry(task_name, "env_cfg_entry_point")
    assert parse_cfg.load_cfg_from_registry("Isaac-Cartpole", "env_cfg_entry_point") is not None
