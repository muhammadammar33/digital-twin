"""Insert or update a row of the results table in benchmarks.md.

Rows are keyed by their id in the first column (e.g. B0, A1). Existing config / vs cells are
kept; the metric cells and notes are overwritten with the new evaluation.
"""
from __future__ import annotations

import re
from pathlib import Path

HEADER = "| # | config (file + overrides) | vs | macro-F1 | macro-AUC | HYP-F1 | notes |"


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _line(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def write_row(path: Path, row_id: str, macro_f1: str, macro_auc: str, hyp_f1: str, notes: str,
              config: str = "", vs: str = "") -> str:
    text = path.read_text()
    lines = text.splitlines()
    try:
        h = next(i for i, l in enumerate(lines) if l.strip() == HEADER)
    except StopIteration:
        raise ValueError(f"results table header not found in {path}")
    end = h + 2
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    for i in range(h + 2, end):
        cells = _cells(lines[i])
        if cells and cells[0] == row_id:
            keep_notes = cells[6] if len(cells) > 6 else ""
            # keep a short static description that precedes the auto-written part
            static = re.split(r"\s*·\s*auto:", keep_notes)[0]
            cells = cells[:7] + [""] * (7 - len(cells))
            cells[1] = cells[1] or config
            cells[2] = cells[2] or vs
            cells[3], cells[4], cells[5] = macro_f1, macro_auc, hyp_f1
            cells[6] = (f"{static} · auto: {notes}" if static else f"auto: {notes}")
            lines[i] = _line(cells)
            break
    else:
        lines.insert(end, _line([row_id, config, vs, macro_f1, macro_auc, hyp_f1, f"auto: {notes}"]))
    out = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    path.write_text(out)
    return next(l for l in lines if l.startswith(f"| {row_id} |"))


def read_row_values(path: Path, row_id: str) -> dict | None:
    """Point values (number before the CI) of a row's macro-F1 / macro-AUC / HYP-F1 cells."""
    for line in Path(path).read_text().splitlines():
        cells = _cells(line) if line.lstrip().startswith("|") else []
        if cells and cells[0] == row_id and len(cells) >= 6:
            def num(c):
                m = re.match(r"\s*([0-9]*\.[0-9]+)", c)
                return float(m.group(1)) if m else None
            vals = {"macro_f1": num(cells[3]), "macro_auc": num(cells[4]), "hyp_f1": num(cells[5])}
            return vals if any(v is not None for v in vals.values()) else None
    return None
