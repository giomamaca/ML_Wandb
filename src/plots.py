from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
_KEYS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "overfit_gap"]


class Plotter:
    def __init__(self, project: str = "fer2013-experiments", entity: Optional[str] = None):
        self.project = project
        self.entity = entity
        self._cache: Dict[str, "object"] = {}   # run_name -> history DataFrame

    def load(self, run_names: Optional[Sequence[str]] = None) -> Dict[str, "object"]:
        import wandb

        api = wandb.Api()
        entity = self.entity or api.default_entity
        runs = api.runs(f"{entity}/{self.project}")

        # A name can appear several times (re-runs / aborted attempts). Keep, per
        # name, the run with the MOST logged epochs, so a half-finished re-run
        # does not shadow the complete one.
        best: Dict[str, tuple] = {}   # name -> (n_rows, df)
        for r in runs:
            if run_names is not None and r.name not in run_names:
                continue
            df = r.history(keys=_KEYS, pandas=True)
            if not len(df):
                continue
            df = df.sort_values("epoch").reset_index(drop=True)
            if r.name not in best or len(df) > best[r.name][0]:
                best[r.name] = (len(df), df)

        out = {name: v[1] for name, v in best.items()}
        self._cache.update(out)
        if not out:
            print("No matching runs found. Available run names:",
                  [r.name for r in api.runs(f"{entity}/{self.project}")])
        return out

    def _get(self, run_name):
        if run_name not in self._cache:
            self.load([run_name])
        return self._cache[run_name]

    def curves(self, run_name: str, history=None):
        """Two panels for one run: loss (train/val) and accuracy (train/val)."""
        df = history if history is not None else self._get(run_name)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(df["epoch"], df["train_loss"], label="train")
        ax1.plot(df["epoch"], df["val_loss"], label="val")
        ax1.set_title(f"{run_name} — loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend()

        ax2.plot(df["epoch"], df["train_acc"], label="train")
        ax2.plot(df["epoch"], df["val_acc"], label="val")
        ax2.set_title(f"{run_name} — accuracy")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend()

        fig.tight_layout()
        return fig

    def gap(self, run_names: Optional[Iterable[str]] = None):
        data = self._select(run_names)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, df in data.items():
            ax.plot(df["epoch"], df["train_acc"] - df["val_acc"], label=name)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("Overfitting gap  (train_acc − val_acc)")
        ax.set_xlabel("epoch"); ax.set_ylabel("gap"); ax.legend()
        fig.tight_layout()
        return fig

    def compare(self, run_names: Optional[Iterable[str]] = None, metric: str = "val_acc"):
        """Overlay one metric (default val_acc) across runs."""
        data = self._select(run_names)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, df in data.items():
            ax.plot(df["epoch"], df[metric], label=name)
        ax.set_title(f"{metric} across architectures")
        ax.set_xlabel("epoch"); ax.set_ylabel(metric); ax.legend()
        fig.tight_layout()
        return fig

    def best_scores(self, run_names: Optional[Sequence[str]] = None, key: str = "best_val_acc"):
        import wandb

        api = wandb.Api()
        entity = self.entity or api.default_entity
        scores: Dict[str, float] = {}
        for r in api.runs(f"{entity}/{self.project}"):
            if run_names is not None and r.name not in run_names:
                continue
            if r.name in scores:
                continue
            val = r.summary.get(key)
            if val is not None:
                scores[r.name] = float(val)

        names = list(scores.keys())[::-1]
        vals = [scores[n] for n in names]
        fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
        bars = ax.barh(names, vals, color="steelblue")
        ax.set_xlabel(key); ax.set_title(f"{key} per run")
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.3f}", va="center")
        fig.tight_layout()
        return fig

    def confusion(self, y_true, y_pred, normalize: bool = True,
                  class_names: Sequence[str] = EMOTIONS, title: str = "Confusion matrix"):
        n = len(class_names)
        cm = np.zeros((n, n), dtype=float)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        if normalize:
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
        thresh = cm.max() * 0.6
        for i in range(n):
            for j in range(n):
                txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        fig.tight_layout()
        return fig

    def _select(self, run_names):
        if run_names is None:
            return dict(self._cache)
        return {n: self._get(n) for n in run_names}
