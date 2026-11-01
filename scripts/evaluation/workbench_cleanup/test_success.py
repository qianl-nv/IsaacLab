# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral traces guard history, full containment, individual placement, and settling."""

import itertools

import numpy as np
import pytest
from success import CleanupSuccess


def fixture_corners():
    """Five separate small test bodies safely inside the drawer, independent of its bounds constants."""
    offsets = np.asarray(list(itertools.product((-0.010, 0.010), (-0.025, 0.025), (-0.010, 0.010))))
    return np.asarray([offsets + [-0.025 + 0.05 * i, -0.524, 0.526] for i in range(5)])


def update(monitor, corners, opening=0.245, speed=None, count=1):
    """Advance a measured-state trace in 50 ms samples."""
    for _ in range(count):
        monitor.update(
            corners,
            corners[:, :, 2].min(axis=1),
            np.zeros(5) if speed is None else speed,
            np.zeros(5),
            opening,
            0.0,
            0.05,
        )


def loaded_history():
    """Exercise five distinct arrivals after a visible lift and opening."""
    inside = fixture_corners()
    current = inside + [0, 0, 0.22]
    monitor = CleanupSuccess()
    monitor.reset(current, 0.0)
    update(monitor, current)
    for i in range(5):
        current[i] = inside[i]
        update(monitor, current, count=6)
        assert len(monitor.placed) == i + 1
        assert not monitor.success
    return monitor, current


def test_success_requires_stable_closure_after_five_individual_placements():
    """Reject premature completion, then accept the complete observable task sequence."""
    monitor, corners = loaded_history()
    update(monitor, corners, opening=0.0, count=8)
    assert not monitor.success
    update(monitor, corners, opening=0.0, count=4)
    assert monitor.success


@pytest.mark.parametrize("fault", ["missing", "protruding", "moving", "open"])
def test_final_guards_with_genuine_placement_history(fault):
    """A valid history cannot mask a missing, protruding, moving tool or an open drawer."""
    monitor, corners = loaded_history()
    speed = np.zeros(5)
    opening = 0.0
    if fault == "missing":
        corners[-1] += [0, 0, 0.22]
    elif fault == "protruding":
        # The tool origin remains inside, but its leading end crosses the wall.
        corners[0, 0, 1] = -0.70
    elif fault == "moving":
        speed[-1] = 0.2
    else:
        opening = 0.245
    update(monitor, corners, opening=opening, speed=speed, count=20)
    assert not monitor.success


def test_preloaded_closed_drawer_does_not_supply_task_history():
    """A static goal arrangement at reset must not substitute for performing the task."""
    corners = fixture_corners()
    monitor = CleanupSuccess()
    monitor.reset(corners, 0.0)
    update(monitor, corners, opening=0.0, count=20)
    assert not monitor.success
    assert not monitor.placed


@pytest.mark.parametrize("arrival", ["simultaneous", "overlapping", "no_lift", "not_open"])
def test_invalid_task_sequences_do_not_succeed(arrival):
    """Batch insertions, overlapping arrivals, and bypassing pickup/opening are rejected."""
    inside = fixture_corners()
    current = inside + [0, 0, 0.22 if arrival != "no_lift" else 0.10]
    monitor = CleanupSuccess()
    monitor.reset(current, 0.0)
    opening = 0.0 if arrival == "not_open" else 0.245
    update(monitor, current, opening)
    if arrival == "simultaneous":
        current = inside.copy()
        update(monitor, current, opening, count=6)
    else:
        for i in range(5):
            current[i] = inside[i]
            update(monitor, current, opening, count=1 if arrival == "overlapping" else 6)
    update(monitor, current, opening=0.0, count=20)
    assert not monitor.success


def test_reset_clears_completed_episode_history():
    """A second episode cannot inherit opening or individual placements from the first."""
    monitor, corners = loaded_history()
    update(monitor, corners, opening=0.0, count=20)
    assert monitor.success
    monitor.reset(corners, 0.0)
    update(monitor, corners, opening=0.0, count=20)
    assert not monitor.success
    assert not monitor.placed


def test_stacked_tools_inside_drawer_are_a_valid_cleanup():
    """The task permits tools resting on other stored tools, without direct floor contact."""
    monitor, corners = loaded_history()
    corners[-1, :, 2] += 0.025
    update(monitor, corners, opening=0.0, count=20)
    assert monitor.success
