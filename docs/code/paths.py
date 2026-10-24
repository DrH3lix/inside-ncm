"""Where the probe scripts find checkpoints and data, and where they write. Override with environment variables.

  NCM_ARTIFACTS  checkpoints: <ARTIFACTS>/tinystories, <ARTIFACTS>/tinystories33m (roneneldan/TinyStories-33M),
                 <ARTIFACTS>/sudoku/easy, <ARTIFACTS>/sudoku/hard  (each NCM folder: generator.pt, tokenizer.pt, config.yaml)
  NCM_DATA_ROOT  Sudoku data: <DATA>/sudoku/{easy,hard}/{puzzles,solutions}.pickle
  NCM_PROBE_OUT  outputs: lens fits in <OUT>/jlens, Sudoku results in <OUT>/sudoku
"""
import os

ARTIFACTS = os.environ.get("NCM_ARTIFACTS", "artifacts")
DATA = os.environ.get("NCM_DATA_ROOT", "data")
WORK = os.environ.get("NCM_PROBE_OUT", "probe_out")
