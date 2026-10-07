"""Generate the numerical reference tables of Appendix C as LaTeX fragments.

Run from the Guide folder with the local venv:
    .venv\\Scripts\\python.exe scripts\\make_tables.py
"""
from __future__ import annotations

import math
import pathlib

from scipy.stats import chi2

OUT = pathlib.Path(__file__).resolve().parent.parent / "tables"
OUT.mkdir(exist_ok=True)


def chi2_table() -> str:
    probs = [0.025, 0.95, 0.975, 0.99]
    head = " & ".join([r"$k$"] + [rf"$\chi^2_{{{p:g}}}$" for p in probs])
    rows = []
    dofs = list(range(1, 41)) + [45, 50, 60, 75, 100, 150, 200, 300, 450, 600]
    half = math.ceil(len(dofs) / 2)
    left, right = dofs[:half], dofs[half:]
    for i in range(half):
        cells = []
        for col in (left, right):
            if i < len(col):
                k = col[i]
                cells.append(" & ".join([str(k)] + [f"{chi2.ppf(p, k):.3f}" for p in probs]))
            else:
                cells.append(" & " * len(probs))
        rows.append(cells[0] + r" & \hspace{4pt} & " + cells[1] + r" \\")
    colspec = "@{}r" + "r" * len(probs) + "c" + "r" * (len(probs) + 1) + "@{}"
    return "\n".join([
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        head + " & & " + head + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def anees_table() -> str:
    ns = [1, 2, 3, 6, 9, 15]
    Ns = [1, 10, 25, 50, 100, 200]
    head = r"$n \backslash N$ & " + " & ".join(str(N) for N in Ns) + r" \\"
    rows = []
    for n in ns:
        cells = []
        for N in Ns:
            lo = chi2.ppf(0.025, n * N) / N
            hi = chi2.ppf(0.975, n * N) / N
            cells.append(f"[{lo:.2f}, {hi:.2f}]")
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{tabular}{@{}r" + "c" * len(Ns) + "@{}}",
        r"\toprule",
        head,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def variance_ratio_table() -> str:
    Ms = [50, 100, 200, 500, 1000, 2000]
    rows = []
    for M in Ms:
        lo = chi2.ppf(0.025, M - 1) / (M - 1)
        hi = chi2.ppf(0.975, M - 1) / (M - 1)
        rows.append(f"{M} & {lo:.3f} & {hi:.3f}" + r" \\")
    return "\n".join([
        r"\begin{tabular}{@{}rcc@{}}",
        r"\toprule",
        r"runs $M$ & lower bound & upper bound \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])

def ransac_table() -> str:
    ws = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    ss = [1, 2, 3, 5, 7, 8]
    rows = []
    for s in ss:
        cells = [str(math.ceil(math.log(1 - 0.99) / math.log(1 - w**s))) for w in ws]
        rows.append(f"{s} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{tabular}{@{}r" + "r" * len(ws) + "@{}}",
        r"\toprule",
        r"$s \backslash w$ & " + " & ".join(f"{w:.1f}" for w in ws) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


if __name__ == "__main__":
    for name, fn in [("chi2_table", chi2_table), ("anees_table", anees_table),
                     ("variance_ratio_table", variance_ratio_table), ("ransac_table", ransac_table)]:
        (OUT / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
        print("wrote", name)
