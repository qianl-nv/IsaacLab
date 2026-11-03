# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-owned registration callbacks."""

from __future__ import annotations

import sys
from unittest import mock

import gymnasium as gym
import pytest

from isaaclab_tasks.utils import task_registration

pytestmark = pytest.mark.unit

_TASK_NAME = "Isaac-Task-Registration-Test"


@pytest.fixture(autouse=True)
def cleanup_task():
    gym.registry.pop(_TASK_NAME, None)
    yield
    gym.registry.pop(_TASK_NAME, None)


def _register(callback) -> None:
    gym.register(
        id=_TASK_NAME,
        entry_point=lambda **_kwargs: None,
        kwargs={task_registration.TASK_REGISTRATION_CALLBACK_KEY: callback},
    )


def test_runs_task_owned_callback_with_workflow_arguments(monkeypatch):
    observed_argv = None

    def callback():
        nonlocal observed_argv
        observed_argv = sys.argv[:]
        return ["agent.seed=3"]

    _register(callback)
    monkeypatch.setattr(task_registration, "load_external_tasks", mock.Mock())
    original_argv = sys.argv

    result = task_registration.run_task_registration_callback(
        [
            "--task",
            _TASK_NAME,
            "--num_envs",
            "2",
        ]
    )

    assert result == ["agent.seed=3"]
    assert observed_argv[1:] == ["--task", _TASK_NAME, "--num_envs", "2"]
    assert sys.argv is original_argv


def test_explicit_callback_takes_priority(monkeypatch):
    task_callback = mock.Mock()
    explicit_callback = mock.Mock(return_value=[])
    _register(task_callback)
    monkeypatch.setattr(task_registration, "load_external_tasks", mock.Mock())
    resolve_callback = mock.Mock(return_value=explicit_callback)
    monkeypatch.setattr(task_registration, "string_to_callable", resolve_callback)

    result = task_registration.run_task_registration_callback(
        [
            "--task",
            _TASK_NAME,
            "--external_callback",
            "example.register",
        ]
    )

    assert result == []
    resolve_callback.assert_called_once_with("example.register", separator=".")
    explicit_callback.assert_called_once_with()
    task_callback.assert_not_called()


def test_no_registered_callback_is_a_noop(monkeypatch):
    monkeypatch.setattr(task_registration, "load_external_tasks", mock.Mock())

    assert task_registration.run_task_registration_callback(["--task", "missing-task"]) is None


def test_rejects_malformed_task_callback(monkeypatch):
    _register(3)
    monkeypatch.setattr(task_registration, "load_external_tasks", mock.Mock())

    with pytest.raises(AssertionError, match="must be a callable or dotted import path"):
        task_registration.run_task_registration_callback(["--task", _TASK_NAME])
