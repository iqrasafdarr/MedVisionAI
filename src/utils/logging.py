"""Lightweight CSV/JSON experiment logging — deliberately not MLflow/W&B.

Each training script uses this to append per-epoch rows to a CSV log and to
write a final summary.json with headline metrics + the resolved config used
to produce them. This is the single source of truth for anything reported
in the README results tables.
"""

import csv
import json
from pathlib import Path
from typing import Any


def append_epoch_log(log_file: str, row: dict[str, Any]) -> None:
    """Append one epoch's metrics as a row to a CSV log, writing the header
    on first call."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(summary_file: str, summary: dict[str, Any]) -> None:
    """Write the final run summary (config used + best metrics) as JSON.

    This file is what results tables in README.md / docs/experiments.md are
    filled in from — never hand-typed.
    """
    path = Path(summary_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
