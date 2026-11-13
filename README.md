# ML practice: drills + segmentation project

Three tracks that build the same skills:

- **`seg/`**: the project track. Build a segmentation pipeline stage by stage, ending with a U-Net trained on Oxford-IIIT Pet.
- **`drills/`**: the interview track. Classic "implement it from scratch" questions, done against a timer.
- **`numpy_drills/`**: the indexing warm-up. 25 short exercises on slicing, views, masks, fancy indexing, scatter, sliding windows and patches, ordered from easy to hard.

Every function is a stub with a docstring giving the shapes, the spec and sometimes a hint. Tests check your code against NumPy, SciPy, PyTorch, scikit-learn or torchvision references. **No solutions are included.** If you're stuck, ask for a *hint*, not the code.

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

All 289 tests fail at the start (`NotImplementedError: TODO`), and that's expected.

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

## Indexing track: `numpy_drills/`

Short NumPy exercises, from easy to hard. Most answers are one to three lines; the work is picking the right indexing tool. The drill rules apply (timer, no internet, no autocomplete), plus two that the tests enforce:

- **No Python loops.** Every test file ends with `test_no_python_loops`, which reads your function's source and fails on `for`, `while`, comprehensions and disguised loops such as `np.vectorize`. Some drills also ban the one function that would make the exercise trivial (`np.flip` in 03, `np.bincount` in 18, `sliding_window_view` in 23, ...).
- **Views when asked.** Wherever a docstring says "view", the test checks `np.shares_memory`.

```bash
uv run pytest tests/numpy_drills -q                         # all 186 tests
uv run pytest tests/numpy_drills/test_n05_view_quiz.py -q   # one drill
```

Drill 05 has no code: replace each `None` with a True/False prediction for 40 expressions, then let the tests tell you which ones you got wrong.

| # | Drill | Covers | Target | Done | Redo |
|---|---|---|---|---|---|
| 01 | 1D slicing | `x[-k:]` and `x[k:-k]` when k = 0, negative steps | 5 min | | |
| 02 | 2D slicing | int vs slice (keepdims), corners via steps, tiles | 5 min | | |
| 03 | steps and slice assignment | flips, downsampling, interleave, checkerboard | 8 min | | |
| 04 | Ellipsis | any number of leading axes, centre crop | 6 min | | |
| 05 | view or copy? (quiz) | `shares_memory`, and whether a write reaches x | 8 min | | |
| 06 | views, copies, in-place | `x = x / s` vs `x /= s`, functions that return None | 8 min | | |
| 07 | boolean masks | combining conditions, masked copies, ignore_index mean | 8 min | | |
| 08 | masks along one axis | drop NaN rows, select columns, NaN fill without warnings | 10 min | | |
| 09 | masks to positions | `nonzero` / `argwhere`, first/last True per row, bbox | 10 min | | |
| 10 | None and broadcasting | outer product, per-row/per-channel ops, causal mask, pairwise diffs | 8 min | | |
| 11 | integer-array indexing | joint shuffle, one-hot, palette lookup, inverse permutation | 8 min | | |
| 12 | paired indices `x[rows, cols]` | per-row pick/set, pixel gather, row swap | 10 min | | |
| 13 | sub-grids with `np.ix_` | submatrix, subgraph, boolean grids | 10 min | | |
| 14 | slices mixed with index arrays | where the broadcast axis ends up | 12 min | | |
| 15 | `take_along_axis` | sort by keys, value at argmax, per-pixel target logits | 10 min | | |
| 16 | broadcast masks, ragged data | sequence mask, pad/unpad, radial distance | 12 min | | |
| 17 | per-row top-k | `argpartition`, top-k mask, top-k accuracy | 12 min | | |
| 18 | `np.add.at` scatter | why `out[idx] += 1` loses counts; bincount, segment mean | 10 min | | |
| 19 | scatter in 2D | heatmap splat, embedding gradient, scatter-max | 12 min | | |
| 20 | sliding windows (1D) | index arithmetic vs `sliding_window_view`, frames, local maxima | 10 min | | |
| 21 | resampling | nearest resize with index grids, bilinear sampling | 15 min | | |
| 22 | non-overlapping patches | reshape + transpose: average pooling, patchify, ViT tokens | 15 min | | |
| 23 | `as_strided` by hand | 1D/2D windows on non-contiguous input | 15 min | | |
| 24 | overlapping patches (im2col) | `unfold` with stride and padding, patches around points | 20 min | | |
| 25 | putting patches back | `fold` with `np.add.at`, max-pool indices, unpooling | 25 min | | |

Do two or three a day as a warm-up before a drill. 20, 22 and 24 are good preparation for drill 10 (conv2d), and 10 and 16 for drill 13 (attention).

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

Commit each finished stage. To redo a drill from scratch: `git show HEAD~N:drills/d08_mlp_backprop.py > /tmp/mine.py` keeps your version, and `git checkout <first-commit> -- drills/d08_mlp_backprop.py` restores the stub. NumPy drill stubs came in a later commit: `git log --diff-filter=A --format=%h -- numpy_drills/` prints its hash.
