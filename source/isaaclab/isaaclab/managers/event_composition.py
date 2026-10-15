# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for composing event-manager configurations."""

from __future__ import annotations

from collections.abc import Mapping

from .manager_term_cfg import EventTermCfg


def event_add(
    target: dict[str, EventTermCfg],
    source: Mapping[str, EventTermCfg] | object | None,
) -> dict[str, EventTermCfg]:
    """Append event terms from ``source`` to ``target`` in declaration order.

    ``source`` may be a mapping or a configclass instance. Duplicate names are
    rejected because silently replacing an event can also change lifecycle
    ordering.

    Args:
        target: Mutable event mapping that receives the terms.
        source: Event mapping or configclass instance to append.

    Returns:
        The mutated ``target`` mapping.
    """
    if source is None:
        return target

    source_items = source.items() if isinstance(source, Mapping) else vars(source).items()
    for name, term_cfg in source_items:
        if name.startswith("_") or term_cfg is None:
            continue
        if not isinstance(term_cfg, EventTermCfg):
            continue
        if name in target:
            raise ValueError(f"Event term '{name}' is already defined.")
        target[name] = term_cfg
    return target
