# ML practice: drills + segmentation project

Two tracks that build the same skills:

- **`seg/`**: the project track. Build a segmentation pipeline stage by stage, ending with a U-Net trained on Oxford-IIIT Pet.
- **`drills/`**: the interview track. Classic "implement it from scratch" questions, done against a timer.

Every function is a stub with a docstring giving the shapes, the spec and sometimes a hint. Tests check your code against NumPy, PyTorch, scikit-learn or torchvision references. **No solutions are included.** If you're stuck, ask for a *hint*, not the code.

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
cd ml-practice
uv sync                                                   # creates .venv from pyproject.toml + uv.lock
uv run pytest tests/drills/test_d01_softmax_xent.py -q    # run one file
uv run pytest -q -x                                       # everything; stop at first failure
uv run pytest tests/drills/test_d05_kmeans.py -q -k blobs # one test by name
uv run python train.py --toy
```

Or activate the environment once with `source .venv/bin/activate`, and then drop the `uv run` prefix. In VS Code, choose `.venv` as the interpreter (Cmd-Shift-P, then "Python: Select Interpreter"). Add packages with `uv add <pkg>` (or `uv add --dev <pkg>` for tools), which updates `pyproject.toml` and the lockfile together.

Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt pytest`.

All 103 tests fail at the start (`NotImplementedError: TODO`), and that's expected.

## Project track: one stage per day

| Day | Stage | You write | Check |
|---|---|---|---|
| 1 | `seg/s1_images.py` | HWC/CHW, normalise, one-hot, joint flip/crop, mask overlay | `pytest tests/project/test_s1_images.py` |
| 2 | `seg/s2_sklearn_baseline.py` | per-pixel features, random-forest pipeline, grouped CV | `test_s2_sklearn_baseline.py` |
| 3 | `seg/s3_torch_basics.py` | tensor conversion, dtypes, training loop from memory | `test_s3_torch_basics.py` |
| 4 | `seg/s4_dataset.py` | `JointTransform`, `SegDataset` | `test_s4_dataset.py` |
| 5 | `seg/s5_unet.py` | `DoubleConv`, `UNet` (no reference!) | `test_s5_unet.py` |
| 6 | `seg/s6_losses_metrics.py`, `seg/s7_train.py`, `seg/pets.py` | Dice, CE+Dice, mIoU, train/eval loops, plots, Pets loading | `test_s6_*`, `test_s7_*` |
| 7 | — | `python train.py --toy`, then `--overfit`, then real training. Break things on purpose and note what each bug looks like. | plots in `runs/` |

`seg/toy.py` (given) generates a small synthetic dataset, so everything can be tested without downloads. `train.py` (given) is a complete training script. Once your stages pass, read it line by line until you could write it yourself.

```bash
python train.py --toy --epochs 6            # ~15 s on CPU; mIoU should reach ~0.9
python train.py --overfit                   # real data: loss should approach 0
python train.py --epochs 20 --limit 1000    # quick real run; drop --limit for the full set
```

## Interview track: the drills

Rules: set a **timer**, use **no internet and no autocomplete**, and **talk out loud** about shapes and choices. Run the tests only when you think you're done. If you go over time or have to look something up, log it in the cheatsheet's "things I got wrong" table and redo the drill 3–4 days later from a blank file (`git stash` or `git checkout drills/<file>`).

| # | Drill | Target | Done | Redo |
|---|---|---|---|---|
| 01 | softmax, log-softmax, cross-entropy + gradient | 15 min | | |
| 02 | linear / ridge regression: closed form + GD | 20 min | | |
| 03 | logistic regression: stable sigmoid, BCE, gradient | 25 min | | |
| 04 | kNN: vectorised distances, voting | 15 min | | |
| 05 | k-means with k-means++ | 30 min | | |
| 06 | PCA via SVD | 15 min | | |
| 07 | decision-tree split search (Gini / entropy) | 25 min | | |
| 08 | **MLP with hand-written backprop** | 40 min | | |
| 09 | dropout, BatchNorm forward/backward, running stats | 35 min | | |
| 10 | conv2d naive + vectorised, max-pool | 40 min | | |
| 11 | confusion matrix, IoU, Dice | 20 min | | |
| 12 | box IoU + NMS | 25 min | | |
| 13 | attention (NumPy) + multi-head attention (PyTorch) | 35 min | | |
| 14 | precision/recall/F1, ROC-AUC with ties | 25 min | | |

Suggested order alongside the project: 01, 11 (Day 1); 02, 07 (Day 2); 03, 04 (Day 3); 05, 06 (Day 4); 10 (Day 5); 08 (Day 6); 09, 12 (Day 7). Then 13 and 14, and after that start redoing whatever you logged.

## When you're stuck (in this order)

1. Re-read the docstring and the failing test. The test shows the exact expected behaviour.
2. Print `.shape` / `.dtype`, or put `breakpoint()` in your function and inspect the arrays.
3. Try a tiny input (N=2, D=3) and work it out by hand.
4. Read the docs for the one function you're unsure about.
5. After about 20 minutes, ask Claude for **a hint**, and say "no code".

## Using git (recommended)

```bash
git init && git add . && git commit -m "stubs"
```

Commit each finished stage. To redo a drill from scratch: `git show HEAD~N:drills/d08_mlp_backprop.py > /tmp/mine.py` keeps your version, and `git checkout <first-commit> -- drills/d08_mlp_backprop.py` restores the stub.
