# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""History-aware success criterion for opening, individually storing tools, and closing."""

from dataclasses import dataclass, field

import numpy as np

TOOL_NAMES = ("slotted_screwdriver", "torx_screwdriver", "phillips_screwdriver", "combination_wrench", "small_wrench")
# Interior in bench coordinates with the top drawer closed. Bounds exclude its walls.
DRAWER_LOWER = np.array([-0.055, -0.683, 0.515])
DRAWER_UPPER = np.array([0.225, -0.365, 0.591])


@dataclass
class CleanupSuccess:
    """Require observed opening, five separate lifted-and-settled arrivals, and stable closure.

    ``corners`` must describe each complete tool in closed-drawer coordinates, with the
    drawer's measured translation removed. This checks full extent rather than tool origins.
    Call ``reset`` at every episode boundary, including when starting with tools inside.
    """

    opened: bool = False
    invalid_sequence: bool = False
    success: bool = False
    placed: list[int] = field(default_factory=list)
    lifted: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=bool))
    previous_inside: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=bool))
    arrival_time: np.ndarray = field(default_factory=lambda: np.zeros(5))
    eligible: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=bool))
    closed_time: float = 0.0

    def reset(self, corners: np.ndarray, drawer_q: float) -> None:
        """Clear episode history without treating preloaded tools as arrivals."""
        self.opened = False
        self.invalid_sequence = False
        self.success = False
        self.placed.clear()
        self.lifted[:] = False
        self.previous_inside[:] = self.contained(corners)
        self.arrival_time[:] = 0.0
        self.eligible[:] = False
        self.closed_time = 0.0

    @staticmethod
    def contained(corners: np.ndarray) -> np.ndarray:
        """Return full-tool containment in the usable drawer interior, within 2 mm."""
        return np.all((corners >= DRAWER_LOWER - 0.002) & (corners <= DRAWER_UPPER + 0.002), axis=(1, 2))

    def update(
        self,
        corners: np.ndarray,
        world_bottoms: np.ndarray,
        linear_speeds: np.ndarray,
        angular_speeds: np.ndarray,
        drawer_q: float,
        drawer_speed: float,
        dt: float,
    ) -> bool:
        """Update from measured physics state and return the current success termination."""
        inside = self.contained(corners)
        self.opened |= drawer_q >= 0.20
        self.lifted |= (world_bottoms > 0.705) & self.opened
        arrivals = np.flatnonzero(inside & ~self.previous_inside)
        unplaced = [int(i) for i in arrivals if i not in self.placed]
        if len(unplaced) > 1:
            self.invalid_sequence = True
        for index in unplaced:
            if not self.opened or drawer_q < 0.20 or not self.lifted[index]:
                continue
            # A second tool arriving before the first has settled is a batch placement.
            if np.any(self.eligible):
                self.invalid_sequence = True
            self.eligible[index] = True
        settled = inside & (linear_speeds < 0.035) & (angular_speeds < 0.4)
        self.arrival_time = np.where(settled & self.eligible, self.arrival_time + dt, 0.0)
        for index in np.flatnonzero(self.arrival_time >= 0.20):
            if int(index) not in self.placed:
                self.placed.append(int(index))
            self.eligible[index] = False
        valid = (
            self.opened
            and not self.invalid_sequence
            and len(self.placed) == 5
            and bool(np.all(settled))
            and abs(drawer_q) < 0.008
            and abs(drawer_speed) < 0.015
        )
        self.closed_time = self.closed_time + dt if valid else 0.0
        self.success = self.closed_time >= 0.50
        self.previous_inside[:] = inside
        return self.success
