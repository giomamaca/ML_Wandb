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
        title="FER2013 — იტერაციული არქიტექტურული კვლევა",
        description=(
            "MLP -> SmallCNN -> RegularizedCNN -> ResNet / AlexNet / GoogLeNet, "
            "თითო არქიტექტურაზე ჰიპერპარამეტრების გადარჩევით, augmentation-ით "
            "და underfit/overfit დემონსტრაციით."
        ),
        width="fluid",
        blocks=[
            wr.MarkdownBlock(
                text=(
                    "# FER2013 — სახის ემოციების ამოცნობა\n\n"
                    "მიზანი არ არის მხოლოდ მაღალი შედეგი, არამედ იმის გაგება, **რატომ** ხდება "
                    "under/overfitting. ვიწყებთ პატარა მოდელით და იტერაციულად ვამატებთ ტევადობას.\n\n"
                    "**საუკეთესო მოდელი: RegularizedCNN + augmentation, ვალიდაციის accuracy ~0.66** "
                    "(~ადამიანის დონე), overfit gap 0.27-დან ~0.05-მდე დაეცა."
                )
            ),
            wr.MarkdownBlock(
                text=(
                    "## 1. არქიტექტურების შედარება\n"
                    "ვალიდაციის accuracy და loss ექვს არქიტექტურაზე. RegularizedCNN საუკეთესო "
                    "რეგულარიზებული მოდელია; ღრმა/კლასიკური ქსელები augmentation-ის გარეშე overfit-ენ."
                )
            ),
            wr.PanelGrid(
                runsets=[rs],
                panels=[
                    wr.LinePlot(title="ვალიდაციის accuracy", x="epoch", y=["val_acc"]),
                    wr.LinePlot(title="ვალიდაციის loss", x="epoch", y=["val_loss"]),
                ],
            ),
            wr.MarkdownBlock(
                text=(
                    "## 2. Overfitting-ის ანალიზი\n"
                    "`overfit_gap = train_acc - val_acc`. დიდი, მზარდი gap ნიშნავს overfitting-ს; "
                    "დაბალი train accuracy და პატარა gap ნიშნავს underfitting-ს."
                )
            ),
            wr.PanelGrid(
                runsets=[rs],
                panels=[
                    wr.LinePlot(title="train vs val accuracy", x="epoch", y=["train_acc", "val_acc"]),
                    wr.LinePlot(title="overfit gap", x="epoch", y=["overfit_gap"]),
                ],
            ),
            wr.MarkdownBlock(text="## 3. საუკეთესო ვალიდაციის accuracy თითო run-ზე"),
            wr.PanelGrid(
                runsets=[rs],
                panels=[wr.BarPlot(title="best_val_acc", metrics=["best_val_acc"])],
            ),
            wr.MarkdownBlock(
                text=(
                    "## დასკვნა\n"
                    "1. ინდუქციური bias (კონვოლუცია) უფრო მნიშვნელოვანია, ვიდრე ტევადობა — MLP underfit-ია.\n"
                    "2. რეგულარიზაცია (BatchNorm + Dropout + weight decay) და განსაკუთრებით "
                    "**augmentation** აჯობებს უფრო რთულ არქიტექტურებს.\n"
                    "3. learning rate ყველაზე კრიტიკული ჰიპერპარამეტრია; optimizer-ის არჩევანი "
                    "underfitting-ს ვერ შველის."
                )
            ),
        ],
    )
    report.save()
    print("REPORT_URL:", report.url)


if __name__ == "__main__":
    main()
