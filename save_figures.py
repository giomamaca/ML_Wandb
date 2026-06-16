"""Render the experiment graphs to images/ so the README can embed them.

Pulls the logged runs from W&B and saves: per-experiment train/val curves, the
val-accuracy comparison, and the overfitting-gap plot. Re-run any time to refresh.

    python save_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: render straight to file, no window

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.plots import Plotter  # noqa: E402

RUNS = ["01_mlp_baseline", "02_small_cnn", "03_regularized_cnn"]
CURVE_FILES = {
    "01_mlp_baseline": "images/exp1_mlp.png",
    "02_small_cnn": "images/exp2_smallcnn.png",
    "03_regularized_cnn": "images/exp3_regcnn.png",
}


def main() -> None:
    os.makedirs("images", exist_ok=True)
    p = Plotter(project="fer2013-experiments")
    p.load(RUNS)

    for run_name, path in CURVE_FILES.items():
        p.curves(run_name).savefig(path, dpi=130, bbox_inches="tight")
        print("saved", path)

    p.compare(RUNS, metric="val_acc").savefig("images/comparison.png", dpi=130, bbox_inches="tight")
    print("saved images/comparison.png")
    p.gap(RUNS).savefig("images/gap.png", dpi=130, bbox_inches="tight")
    print("saved images/gap.png")


if __name__ == "__main__":
    main()
