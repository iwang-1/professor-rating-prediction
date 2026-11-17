"""Compare a regenerated results table with the committed one, to a tolerance.

The committed table was produced on macOS arm64, where CI reproduces it byte for byte. On Linux
x86-64 the RandomForest row moves in the fourth decimal (floating-point rounding differs across CPU
architectures and changes which near-tied tree splits win), so there CI checks agreement to the
three decimals the README reports.

Usage: python scripts/compare_results.py COMMITTED.csv REGENERATED.csv [--tol 0.001]
"""

import argparse
import sys

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("committed")
parser.add_argument("regenerated")
parser.add_argument("--tol", type=float, default=1e-3)
args = parser.parse_args()

old = pd.read_csv(args.committed).set_index("model")
new = pd.read_csv(args.regenerated).set_index("model")
if list(old.index) != list(new.index) or list(old.columns) != list(new.columns):
    sys.exit(f"model rows or metric columns changed:\n{old.index.tolist()}\n{new.index.tolist()}")

diff = (old - new).abs()
worst = diff.max().max()
print(diff.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"largest absolute difference: {worst:.4f} (tolerance {args.tol})")
if worst > args.tol:
    sys.exit("regenerated results disagree with the committed table beyond the tolerance")
