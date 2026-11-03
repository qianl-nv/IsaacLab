# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned registration callbacks for downstream Isaac Lab environments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import gymnasium as gym

from isaaclab.cli import load_external_tasks
from isaaclab.utils.string import string_to_callable

TASK_REGISTRATION_CALLBACK_KEY = "task_registration_callback"
"""Gym task metadata key containing a callback that completes task registration."""


def run_task_registration_callback(argv: list[str]) -> list[str] | None:
    """Run the explicit or task-owned registration callback selected by command-line arguments.

    Args:
        argv: Command-line arguments excluding the executable name.

    Returns:
        Arguments left for the workflow's configuration system, or None when no callback is registered.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task")
    parser.add_argument("--external_callback")
    args, _ = parser.parse_known_args(argv)

    load_external_tasks()
    callback: str | Callable[..., Any] | None = args.external_callback
    if callback is None and args.task is not None:
        try:
            task_spec = gym.spec(args.task.split(":")[-1])
        except gym.error.Error:
            task_spec = None
        if task_spec is not None:
            callback = task_spec.kwargs.get(TASK_REGISTRATION_CALLBACK_KEY)
    if callback is None:
        return None
    if isinstance(callback, str):
        callback = string_to_callable(callback, separator=".")
    assert callable(callback), (
        f"Gym task metadata {TASK_REGISTRATION_CALLBACK_KEY!r} must be a callable or dotted import path,"
        f" got {callback!r}"
    )

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        return callback()
    finally:
        sys.argv = original_argv
