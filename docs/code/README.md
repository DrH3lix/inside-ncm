# Code for "Inside a Next-Concept Model"

Scripts for every experiment on the results page, by DJ Oetken and Claude (Anthropic), October 2026.
NCM is from Stanford's Hazy Research and Scaling Intelligence labs.
They run against a clone of [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (tested at commit `c702742`):
put these files in a `probe/` folder at the root of the repo and run them from the repo root.
Everything ran on one 16 GB consumer GPU (PyTorch 2.11, CUDA 12.8).

## Setup

Checkpoints and data mirror the folder layout of the Hugging Face repos:

| What | From | Where the scripts look |
|---|---|---|
| NCM TinyStories, Sudoku easy/hard | `hazyresearch/ncm-checkpoints` | `artifacts/tinystories`, `artifacts/sudoku/{easy,hard}` |
| Control model | `roneneldan/TinyStories-33M` | `artifacts/tinystories33m` |
| Sudoku puzzles + solutions | `hazyresearch/ncm-datasets` (`sudoku/`) | `data/sudoku/{easy,hard}/{puzzles,solutions}.pickle` |

To keep them somewhere else, set `NCM_ARTIFACTS`, `NCM_DATA_ROOT` and `NCM_PROBE_OUT` (outputs; default `probe_out/`).
See `paths.py`. All released checkpoints load with `torch.load(..., weights_only=True)`.

## The experiments, in page order

| Script | What it does | Command |
|---|---|---|
| `drafts.py` | Decodes NCM's draft after each level (and has `level_snapshots`, which the other scripts use) | `python probe/drafts.py --n 4 --seed 7` |
| `jlens.py` | Fits the Jacobian lens for NCM and the control, then measures alignment against a rotated null, Jacobian size and rank, and readouts | `python probe/jlens.py fit --model control`, `... fit --model ncm`, then `python probe/jlens.py analyze` |
| `swap.py` | Dog-to-cat push along the lens direction, with a random-direction control and the "words kept" measure | `python probe/swap.py ncm`, `python probe/swap.py control` |
| `negation.py` | Do dropped menu words get pushed below neutral words? | `python probe/negation.py --n 64` |
| `sudoku.py` | Sudoku: accuracy, drafts by level, menus, erasure | `python probe/sudoku.py --task hard --n 1000` |
| `sudoku_followup.py` | Sudoku: erasure from every level, the pencil-mark (rules) test, error ranks (run `sudoku.py` first) | `python probe/sudoku_followup.py --task hard` |
| `sudoku_swap.py` | Sudoku: swap levels 1..k for another puzzle's codes and let the model finish | `python probe/sudoku_swap.py --task hard --n 1000` |
| `stories_swap.py` | The same swap on TinyStories: same random draws, levels 1..k from another story | `python probe/stories_swap.py --n 128 --seed 21` |

Images: `make_tweet_images.py` draws the charts from `results_page/*.json`; `anim_swap_data.py` regenerates the
cat-swap story level by level (seed 11, level 5, strength 4), and `make_animations.py` turns it and the fish story
into the animations (needs ffmpeg for the MP4s). Outputs go to `tweet_images/` next to `probe/`.

`jlens.py analyze` reads the fits from `fit`; `swap.py` and `negation.py` reuse them. For the fresh-seed
replication we set `JLENS_SEED_SHIFT` (new seeds) and `JLENS_OUT` (a separate output folder) and reran everything.

## The record

`HYPOTHESES.md` is the lab notebook: each set of predictions was written to it before the data, then scored
afterward, wrong calls and corrections included. "Argon" in it is Gemini; DJ kept Gemini's predictions sealed
until the results were in.

<sub>MIT License. Copyright (c) 2026 DJ Oetken.</sub>
