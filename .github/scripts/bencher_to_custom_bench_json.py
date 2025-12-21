#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


BENCHER_LINE_RE = re.compile(
    r"^test (.+)\s+\.\.\. bench:\s+([0-9,.]+) (\w+/\w+) \(\+/- ([0-9,.]+)\)$"
)


def choose_unit(ns_per_iter: float) -> Tuple[str, float]:
    if ns_per_iter >= 1_000_000_000:
        return ("s/iter", 1_000_000_000.0)
    if ns_per_iter >= 1_000_000:
        return ("ms/iter", 1_000_000.0)
    if ns_per_iter >= 1_000:
        return ("us/iter", 1_000.0)
    return ("ns/iter", 1.0)


def parse_bencher(path: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            m = BENCHER_LINE_RE.match(line)
            if not m:
                continue
            name = m.group(1).strip()
            ns = float(m.group(2).replace(",", ""))
            ns_range = float(m.group(4).replace(",", ""))

            unit, scale = choose_unit(ns)
            results.append(
                {
                    "name": name,
                    "unit": unit,
                    "value": ns / scale,
                    "range": f"± {ns_range / scale}",
                }
            )
    results.sort(key=lambda r: r["name"])
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert cargo bencher output (ns/iter) to github-action-benchmark custom JSON with adaptive units."
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data = parse_bencher(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

