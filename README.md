# FER2013 — Facial Expression Recognition: An Iterative Architecture Study

PyTorch + Weights & Biases solution to the Kaggle
[*Challenges in Representation Learning: Facial Expression Recognition Challenge*](https://kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge).

The goal of this project is **not** to chase a single high score. Following the
assignment, it grows a model **one motivated step at a time** and uses the
train/validation curves to diagnose **underfitting** and **overfitting** — the
*why* behind each result matters more than the raw number.

---

## TL;DR

- **6 architectures** forming a clear story: `Linear → MLP → SmallCNN → RegularizedCNN → DeepCNN → ResidualCNN`.
- **Forward/backward sanity checks** (initial-loss, overfit-a-batch, gradient check) before any long run.
- **One W&B run per architecture/config**, organized MLflow-style with `group` (architecture) and `job_type` (underfit/overfit/fit/sweep).
- **Hyperparameter sweeps** (W&B Bayesian search) for the two strongest models.
- **Deliberate ablations** (LR too high, SGD vs Adam) so the analysis points at concrete failure curves.
- A reproducible **Colab notebook** runs the whole study end-to-end.

---

## Dataset

FER2013: 35,887 grayscale **48×48** face images, **7 emotions**
(`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`). We use `fer2013.csv`
(Kaggle dataset `deadskull7/fer2013`) because it keeps the official `Usage` column:

| `Usage`       | Rows   | Our role    |
|---------------|--------|-------------|
| `Training`    | 28,709 | train       |
| `PublicTest`  | 3,589  | validation  |
| `PrivateTest` | 3,589  | test        |

The classes are **strongly imbalanced** — `Disgust` has only ~547 examples vs
~8,989 for `Happy`. This is visible later as the lowest per-class accuracy and is
why we offer inverse-frequency class weighting.

---

## Repository structure

```
ML_Wandb/
├── README.md                     # this file — the iterative story + how to run
├── requirements.txt
├── notebooks/
│   └── FER2013_experiments.ipynb # main Colab notebook (setup → sanity → experiments → sweeps → report)
├── src/
│   ├── data.py                   # FERDataset, splits, augmentation, class weights
│   ├── models.py                 # the 6 architectures + build_model() factory
│   ├── train.py                  # training loop + all W&B logging
│   ├── utils.py                  # sanity checks (forward/backward) + seeding
│   ├── experiments.py            # curated, named runs (the iterations + ablations)
│   └── sweeps.py                 # W&B hyperparameter-sweep runner
├── sweeps/
│   ├── sweep_regularized_cnn.yaml
│   └── sweep_deep_cnn.yaml
├── scripts/
│   └── make_wandb_report.py      # bonus: auto-generate a W&B Report
├── reports/
│   └── REPORT.md                 # written analysis (basis for the W&B report)
└── tests/
    └── smoke_test.py             # fast CPU pipeline check on synthetic data
```

---

## How to run

### Colab (recommended — needs a GPU)
1. Open `notebooks/FER2013_experiments.ipynb` in Colab, set runtime to **GPU**.
2. Add **Colab Secrets** (🔑): `WANDB_KEY`, and optionally `KAGGLE_JSON` (paste
   the contents of your `kaggle.json`) and `GITHUB_TOKEN` (only if the repo is private).
3. Run all cells. The notebook clones this repo, installs deps, downloads the data,
   logs in to W&B, runs the sanity checks, the iterations, the ablations, the
   sweeps, and finally generates the W&B report.

> **Secrets are never hard-coded.** They are read from Colab Secrets / env vars.

### Local
```bash
pip install -r requirements.txt
python tests/smoke_test.py                 # ~30s CPU check, no data/W&B needed
# real training (after downloading fer2013.csv):
python -c "from src.experiments import run_experiment; run_experiment('iter4_deep_cnn', 'fer2013.csv')"
```

---

## W&B logging structure (MLflow-style)

| MLflow concept     | Here                                                            |
|--------------------|----------------------------------------------------------------|
| Experiment         | W&B **project** `fer2013-experiments`                           |
| Run                | one `wandb.init` per architecture/config                       |
| Nested grouping    | `group` = architecture family, `job_type` = phase              |
| Params             | `config` (lr, optimizer, dropout, scheduler, augment, …)       |
| Metrics (per step) | `train_loss/acc`, `val_loss/acc`, **`overfit_gap`**, `lr`      |
| Metrics (summary)  | `best_val_acc`, `best_epoch`, `test_acc_best`, `acc_<emotion>` |
| Artifacts / media  | best-checkpoint artifact, confusion matrix, sample predictions |

`overfit_gap = train_acc − val_acc` is logged every epoch precisely so the
under/over-fitting behaviour is a first-class, comparable metric across runs.

---

## The iterative story (and the reasoning behind each step)

Each iteration changes **one thing** and we predict + then verify its effect.

### Iteration 0 — `LinearClassifier` (softmax regression)
- **What:** a single `Linear(2304 → 7)` over raw pixels.
- **Why:** establishes the floor — can we beat the 1/7 ≈ 14% chance baseline at all?
- **Expected:** clear **underfit**; train ≈ val and both low (~30–35%). A linear model
  can't capture the non-linear structure of faces.

### Iteration 1 — `MLP`
- **What:** flatten → 512 → 256 → 7 with ReLU.
- **Why:** add non-linear capacity. Does raw capacity alone help?
- **Expected:** better than linear but **still underfits a CNN** — flattening throws
  away 2D spatial structure, so it must relearn locality from scratch. Adding dropout
  (`iter1_mlp_dropout`) does **not** help, demonstrating that regularizing an already
  underfitting model is the wrong move.

### Iteration 2 — `SmallCNN` (no regularization)
- **What:** 3 conv blocks (32→64→128) + a large FC head, **no** BatchNorm/Dropout.
- **Why:** introduce the convolutional inductive bias (locality + weight sharing).
- **Expected:** big jump in train accuracy, but **strong overfitting** — `train_acc`
  races ahead of `val_acc`, so `overfit_gap` grows. The fat FC head memorizes the
  training set. This is our canonical overfitting run.

### Iteration 3 — `RegularizedCNN` (BatchNorm + Dropout, then + augmentation)
- **What:** double conv per block + **BatchNorm** (stable/faster optimization) +
  **Dropout** (regularization). Two runs: without and with **augmentation**
  (random flips + translations).
- **Why:** directly attack the overfitting from iter 2.
- **Expected:** the train/val **gap shrinks**; validation accuracy rises.
  Augmentation shrinks it further by enlarging the effective dataset. A
  `ReduceLROnPlateau` scheduler drops the LR when val accuracy stalls.

### Iteration 4 — `DeepCNN` (VGG-style + global average pooling) — best baseline
- **What:** 4 conv stages (64→128→256→512) and a **GAP** head instead of a giant FC
  layer, with `AdamW`, weight decay, **cosine** LR schedule and **label smoothing**.
- **Why:** more representational depth, far fewer head parameters (GAP ⇒ less memorization),
  and modern training tricks for better generalization.
- **Expected:** the best validation/test accuracy of the baselines, with a small gap.

### Iteration 5 — `ResidualCNN`
- **What:** a small ResNet (stem + 3 residual stages + GAP).
- **Why:** skip connections let us go deeper **without** the optimization degradation
  that plagues plain deep stacks.
- **Expected:** comparable-to-best accuracy, stable training at depth.

### Ablations (failure/analysis runs)
- `ablation_lr_too_high` (lr = 0.5): the optimizer **diverges/stalls** — loss stays high. Shows what a bad LR looks like on the curves.
- `ablation_sgd_vs_adam`: SGD+momentum vs Adam — different convergence speed for the same model.

---

## Hyperparameter tuning

Broad search is delegated to **W&B Sweeps** (Bayesian + Hyperband early-termination)
for the two strongest architectures:

- `sweeps/sweep_regularized_cnn.yaml` — lr, optimizer, weight_decay, dropout, batch_size, label_smoothing.
- `sweeps/sweep_deep_cnn.yaml` — lr, weight_decay, dropout, batch_size, label_smoothing.

```python
from src.sweeps import run_sweep
run_sweep(CSV, sweep_yaml="sweeps/sweep_regularized_cnn.yaml", count=20)
```
W&B then produces parameter-importance and parallel-coordinates plots automatically.

---

## Sanity checks (the forward/backward tests)

Run from `src/utils.py` before trusting any curve (see notebook §3):

| Check                  | Passing criterion                          | A failure means…                       |
|------------------------|--------------------------------------------|----------------------------------------|
| `check_initial_loss`   | random-init loss ≈ `ln(7) = 1.946`         | bad init / wrong #classes / wiring bug |
| `overfit_small_batch`  | reaches ~100% on a fixed 64-image batch    | broken graph / loss / label pipeline   |
| `check_gradients`      | every param gets a finite, non-zero grad   | dead path (0) or exploding/NaN grads   |

---

## Results

The numbers below are **typical FER2013 ranges** for each design point plus the
**qualitative behaviour** to expect; fill in the *Your val/test* columns from your
W&B runs. (Human accuracy on FER2013 is ~65 ± 5%; strong single CNNs reach ~68–72%.)

| Iter | Model            | Behaviour            | Typical val acc | Your val | Your test |
|------|------------------|----------------------|-----------------|----------|-----------|
| 0    | LinearClassifier | underfit (floor)     | ~0.31           |          |           |
| 1    | MLP              | underfit             | ~0.36           |          |           |
| 2    | SmallCNN         | **overfit**          | ~0.55 (big gap) |          |           |
| 3    | RegularizedCNN   | fit (gap shrinks)    | ~0.62           |          |           |
| 4    | DeepCNN          | **best baseline**    | ~0.66–0.70      |          |           |
| 5    | ResidualCNN      | fit (deep, stable)   | ~0.66–0.70      |          |           |

See `reports/REPORT.md` for the full written analysis and `FER2013_experiments.ipynb`
for the runnable study. The interactive comparison (curves, confusion matrices,
sample predictions, sweeps) lives on the W&B project dashboard.

---

## Notes
- **Reproducibility:** `src.utils.set_seed(42)` seeds Python/NumPy/PyTorch.
- **Mixed precision** (`amp=True`) and `pin_memory` activate only on GPU.
- **Security:** never commit `kaggle.json`, API keys, or tokens — they're git-ignored
  and read from Colab Secrets at runtime.
