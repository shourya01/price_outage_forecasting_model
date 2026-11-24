"""Build the main results table from the per-task metrics written by `python -m price.run`.

    python -m price.table --results results

Writes main_table.md, main_table.tex and main_table.csv into the results directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from price.methods import METHODS, TABLE_ORDER

COLUMNS = [("michigan", 24), ("michigan", 48), ("california", 24), ("california", 48)]


def load(results):
    """Read every <results>/*/<method>/metrics.json into {(method, dataset, lookahead): (rmse, mae)}."""
    scores = {}
    for f in Path(results).glob("*/*/metrics.json"):
        m = json.load(open(f))
        scores[(m["method"], m["dataset"], m["lookahead"])] = (m["rmse"], m["mae"])
    return scores


def ranks(scores, col, k):
    """Lowest and second-lowest value, rounded to three decimals, of metric `k` (0 RMSE, 1 MAE) in column `col`."""
    vals = sorted({round(scores[(m, *col)][k], 3) for m in TABLE_ORDER if (m, *col) in scores})
    return (vals[0] if vals else None), (vals[1] if len(vals) > 1 else None)


def main(argv=None):
    """Command-line entry point: build main_table.{md,tex,csv} from the metrics under --results."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    scores = load(a.results)
    if not scores:
        raise SystemExit(f"no metrics.json under {a.results}; run `python -m price.run` first")
    methods = [m for m in TABLE_ORDER if any((m, *c) in scores for c in COLUMNS)]
    best = {(c, k): ranks(scores, c, k) for c in COLUMNS for k in (0, 1)}

    def cell(m, c, k, bold, under):
        """Format one cell, marking the best and the second-best value of its column."""
        if (m, *c) not in scores:
            return "--"
        v = round(scores[(m, *c)][k], 3)
        first, second = best[(c, k)]
        s = f"{v:.3f}"
        return bold(s) if v == first else under(s) if v == second else s

    head = ["Method"] + [f"{d.capitalize()} {la}h {n}" for d, la in COLUMNS for n in ("RMSE", "MAE")]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [
        r"\begin{tabular}{l" + "c" * (len(head) - 1) + "}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Michigan}} & \multicolumn{4}{c}{\textbf{California}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        "Method & " + " & ".join(f"{n} ({la}h)" for _, la in COLUMNS for n in ("RMSE", "MAE")) + r" \\",
        r"\midrule",
    ]
    csv = [",".join(["method"] + [f"{d}_{la}_{n}" for d, la in COLUMNS for n in ("rmse", "mae")])]
    for m in methods:
        label = METHODS[m].label
        md.append(
            "| "
            + " | ".join(
                [label] + [cell(m, c, k, lambda s: f"**{s}**", lambda s: f"_{s}_") for c in COLUMNS for k in (0, 1)]
            )
            + " |"
        )
        tex.append(
            label.replace("_", r"\_")
            + " & "
            + " & ".join(
                cell(m, c, k, lambda s: rf"\textbf{{{s}}}", lambda s: rf"\underline{{{s}}}")
                for c in COLUMNS
                for k in (0, 1)
            )
            + r" \\"
        )
        csv.append(
            ",".join([m] + [f"{scores[(m, *c)][k]:.3f}" if (m, *c) in scores else "" for c in COLUMNS for k in (0, 1)])
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    out = Path(a.results)
    (out / "main_table.md").write_text("\n".join(md) + "\n")
    (out / "main_table.tex").write_text("\n".join(tex) + "\n")
    (out / "main_table.csv").write_text("\n".join(csv) + "\n")
    print("\n".join(md))
    print(
        f"\nPer-county RMSE and MAE on the test targets; best in bold, second best in italics. Written to {out}/main_table.{{md,tex,csv}}"
    )


if __name__ == "__main__":
    main()
