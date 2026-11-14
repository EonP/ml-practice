# ml-practice

A personal practice repo for learning ML by writing it from scratch. Every exercise is a stub function with a docstring (shapes, spec, sometimes a hint), and pytest checks it against a NumPy/SciPy/PyTorch/scikit-learn reference. README.md has the full plan, the progress tables and the rules for each track.

## How to help

- **Don't write solutions.** The point is that the user writes the code. Never fill in a stub in `seg/`, `drills/` or `numpy_drills/`, and don't paste a solution in chat, unless the user explicitly asks for the code.
- **Give hints when asked, starting small:** point at the failing test, a shape mismatch, or the one function worth looking up. Only get more specific if the user asks again.
- **When asked to check an answer:** run its tests, explain any failures, and comment on correctness, vectorisation and numerical stability. Don't rewrite it.
- Given code, not exercises: `seg/toy.py`, `train.py`, everything under `tests/`.

## Layout

- `seg/s1_*.py` … `seg/s7_*.py`, `seg/pets.py`: the project track, building up to a U-Net trained on Oxford-IIIT Pet. Tests are in `tests/project/`.
- `drills/d01_*.py` … `d14_*.py`: timed interview drills. Tests are in `tests/drills/`.
- `numpy_drills/n01_*.py` … `n25_*.py`: short indexing exercises. Tests are in `tests/numpy_drills/`. Every test file ends with `test_no_python_loops`, which uses the `no_loops` fixture in `tests/numpy_drills/conftest.py` to reject loops, comprehensions, `np.vectorize` and similar. Drill 05 is a True/False quiz, not code.
- `notebooks/`: scratch space. Git's nbstripout filter strips notebook outputs on commit.

## Commands

The environment is managed with uv (Python 3.12). Prefix commands with `uv run`.

```bash
uv run pytest tests/drills/test_d01_softmax_xent.py -q   # one file
uv run pytest tests/numpy_drills -q                      # one track
uv run pytest -q -k <name>                               # one test by name
uv run python train.py --toy --epochs 6                  # ~15 s smoke run on the synthetic data
```

## Git workflow

- **`main`** holds the questions only, with every function an unanswered stub. The **`stubs`** tag marks that snapshot.
- **`answers`** holds the user's work. Commit answers here, never on `main`.
- Restore a stub to redo it: `git checkout stubs -- <file>`.
- Adding new questions:
  1. Commit them on `main`.
  2. Move the tag: `git tag -f stubs main`.
  3. Bring them into the user's branch: `git switch answers && git merge main`.
  4. Push: `git push origin main answers && git push -f origin stubs`.
- Ask before pushing. If a push is blocked, give the user the command to run with the `!` prefix.

## Adding exercises

Match the existing files:

- **Stub:** a module docstring with a target time, and functions whose docstrings give shapes, spec and hints. Each body is `raise NotImplementedError("TODO: <name>")`.
- **Tests:** compare against a library reference (torch, scipy, sklearn) or a plain-loop reference. Cover edge cases such as k = 0, size-1 axes and non-contiguous input. Check views with `np.shares_memory`.
- **Verify before committing:** fill the stubs temporarily with a reference solution and confirm every test passes. Break the solution in a few ways and confirm a test catches each one. Then restore the stubs, so that on `main` every test fails with `NotImplementedError`. Solutions never get committed.
- **Update README.md:** the track table and the test counts ("All N tests fail at the start", and the per-track count).
