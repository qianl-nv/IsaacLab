# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for composing interactive scene configurations."""

from types import SimpleNamespace

import pytest
from isaaclab.scene import InteractiveSceneCfg, add


def test_add_merge_preserves_component_fields_and_settings():
    """Merge mode keeps non-asset entities and applies source scene settings."""
    target = InteractiveSceneCfg(num_envs=8, env_spacing=2.0)
    source = SimpleNamespace(sensor="camera", global_light="dome")
    bindings = {}

    result = add(target, source, mode="merge", bindings=bindings)

    assert result is target
    assert target.sensor == "camera"
    assert target.global_light == "dome"
    assert bindings == {"sensor": "sensor", "global_light": "global_light"}
    assert target.num_envs == 8


def test_add_merge_conflict_policy():
    """Merge mode rejects collisions unless replacement is explicit."""
    target = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
    target.sensor = "first"
    source = SimpleNamespace(sensor="second")

    with pytest.raises(ValueError, match="sensor.*already defined"):
        add(target, source, mode="merge")

    add(target, source, mode="merge", on_conflict="replace")
    assert target.sensor == "second"


def test_add_merge_applies_source_scene_settings():
    """A complete source scene can intentionally override execution settings."""
    target = InteractiveSceneCfg(num_envs=8, env_spacing=2.0, replicate_physics=False)
    source = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)

    add(target, source, mode="merge")

    assert target.num_envs == 4
    assert target.env_spacing == 3.0
    assert target.replicate_physics is True
