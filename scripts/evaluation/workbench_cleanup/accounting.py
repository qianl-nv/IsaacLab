# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Snapshot measured Codex usage without copying conversation content into artifacts."""

import argparse
import collections
import datetime
import json
import re
from pathlib import Path


def summarize(session: Path, start: str, end: str) -> dict:
    """Count actual inference records and tool dispatches in a half-open UTC interval."""
    rows = [json.loads(line) for line in session.read_text().splitlines()]
    selected = [row for row in rows if start <= row["timestamp"] < end]
    responses = {}
    calls = collections.Counter()
    nested = collections.Counter()
    for row in selected:
        payload = row.get("payload", {})
        if row["type"] == "token_usage_record":
            responses[payload["response_id"]] = payload["usage"]
        if row["type"] == "response_item" and payload.get("type") in ("custom_tool_call", "function_call"):
            name = payload["name"]
            calls[name] += 1
            if name.endswith("exec"):
                source = payload.get("input", payload.get("arguments", ""))
                nested.update(re.findall(r"(?<![.\w])tools\.([A-Za-z_][A-Za-z_0-9]*)\s*\(", source))
    usage = collections.Counter()
    for response in responses.values():
        usage.update({key: value for key, value in response.items() if isinstance(value, int)})
    elapsed = (
        datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))
        - datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
    ).total_seconds()
    return {
        "start_utc": start,
        "end_utc": end,
        "wall_time_s": elapsed,
        "inference_calls": len(responses),
        "tokens": dict(usage),
        "orchestration_tool_calls": dict(calls),
        "nested_tool_calls_by_type": dict(nested),
        "accounting_notes": [
            "Inference calls are unique response_id values in token_usage_record events.",
            "Token counts include cached input; total_tokens includes reasoning inside output_tokens.",
            "Nested tool counts are dispatch sites in submitted exec source; this task uses no tool-call loops.",
            "Wall time includes tool execution, model inference, retries, and approval waits.",
            "The snapshot excludes subsequent reporting calls and the final chat response.",
        ],
    }


def main():
    """Write the usage snapshot for one construction step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    end = args.end or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    report = summarize(args.session, args.start, end)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
