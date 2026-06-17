"""Generate a Weights & Biases Report (bonus task).

Builds a shareable report in the `fer2013-experiments` project that combines text
with panels comparing every run: validation accuracy/loss, the train-vs-val
overfitting gap, and best accuracy per run.

    pip install wandb-workspaces
    python make_report.py            # uses your default W&B entity
    python make_report.py --entity <your-entity>
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None, help="W&B entity (default: your default)")
    parser.add_argument("--project", default="fer2013-experiments")
    args = parser.parse_args()

    import os
    import wandb
    import wandb_workspaces.reports.v2 as wr

    key = os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key, relogin=True)

    entity = args.entity or wandb.Api().default_entity
    rs = wr.Runset(entity=entity, project=args.project, name="All runs")

    report = wr.Report(
        entity=entity,
        project=args.project,
        title="FER2013 — Iterative Architecture Study",
        description=(
            "MLP -> SmallCNN -> RegularizedCNN -> ResNet / AlexNet / GoogLeNet, "
            "with per-architecture hyperparameter sweeps, an augmentation study, "
            "and deliberate over/underfit demos."
        ),
        width="fluid",
        blocks=[
            wr.MarkdownBlock(
                text=(
                    "# FER2013 facial-expression recognition\n\n"
                    "Goal: not just a high score, but understanding **why** models under/overfit. "
                    "Start small and add capacity one motivated step at a time.\n\n"
                    "**Best model: RegularizedCNN + augmentation, ~0.66 validation accuracy** "
                    "(~human level), with the overfitting gap collapsing from 0.27 to ~0.05."
                )
            ),
            wr.MarkdownBlock(
                text=(
                    "## 1. Architecture comparison\n"
                    "Validation accuracy and loss across the six architectures. RegularizedCNN "
                    "is the strongest plain model; the deeper/classic nets overfit without "
                    "augmentation."
                )
            ),
            wr.PanelGrid(
                runsets=[rs],
                panels=[
                    wr.LinePlot(title="Validation accuracy", x="epoch", y=["val_acc"]),
                    wr.LinePlot(title="Validation loss", x="epoch", y=["val_loss"]),
                ],
            ),
            wr.MarkdownBlock(
                text=(
                    "## 2. Overfitting analysis\n"
                    "`overfit_gap = train_acc - val_acc`. A large, growing gap means overfitting; "
                    "low train accuracy with a small gap means underfitting."
                )
            ),
            wr.PanelGrid(
                runsets=[rs],
                panels=[
                    wr.LinePlot(title="train vs val accuracy", x="epoch", y=["train_acc", "val_acc"]),
                    wr.LinePlot(title="overfitting gap", x="epoch", y=["overfit_gap"]),
                ],
            ),
            wr.MarkdownBlock(text="## 3. Best validation accuracy per run"),
            wr.PanelGrid(
                runsets=[rs],
                panels=[wr.BarPlot(title="best_val_acc", metrics=["best_val_acc"])],
            ),
            wr.MarkdownBlock(
                text=(
                    "## Conclusion\n"
                    "1. Inductive bias (convolution) matters more than raw capacity — the MLP underfits.\n"
                    "2. Regularization (BatchNorm + Dropout + weight decay) and especially "
                    "**augmentation** beat fancier architectures.\n"
                    "3. Learning rate is the most critical hyperparameter; the optimizer choice "
                    "does not fix underfitting."
                )
            ),
        ],
    )
    report.save()
    print("REPORT_URL:", report.url)


if __name__ == "__main__":
    main()
