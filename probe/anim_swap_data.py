"""Level-by-level drafts of one story with and without the cat-minus-dog push at level 5 (for the thread animation).

Re-runs swap.py's NCM setup exactly (seed 11, 64 stories, same level-1 codes and sampling seed, push at L5 with
strength 4) and decodes the draft after every level for the "Lucy walks her dog Max" story, baseline and pushed.
Writes <NCM_PROBE_OUT>/anim/cat_swap_drafts.json

  python probe/anim_swap_data.py
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from drafts import level_snapshots, text  # noqa: E402
from jlens import DEV, OUT, NCM, unit  # noqa: E402
from paths import WORK  # noqa: E402
from swap import Push  # noqa: E402

SEED, N, LEVEL, STRENGTH = 11, 64, 5, 4.0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    m = NCM()
    Ms = torch.load(f"{OUT}/ncm_M.pt", weights_only=True)["M"]
    pt = m.tk.pre_tokenizer
    cat, dog = pt.encode(" cat")[0], pt.encode(" dog")[0]
    torch.manual_seed(SEED)
    l0 = torch.multinomial(m.l0_p, N * m.levels[0], replacement=True).view(N, m.levels[0]).to(DEV)

    def gen(push):
        h = m.net.transformer.ln_f.register_forward_pre_hook(push) if push else None
        try:
            torch.manual_seed(SEED + 1)
            with torch.no_grad():
                return m.net.generate(l0, torch.full((N, 1), m.eos, device=DEV), temperature=1.0, top_k=0, top_p=1.0)
        finally:
            if h:
                h.remove()

    j = LEVEL - 2
    W = m.W @ Ms[j, 12].to(DEV)
    d = unit(unit(W[cat]) - unit(W[dog]))
    base, pushed = gen(None), gen(Push(j, d, STRENGTH, m.P))
    finals = m.text(base)
    i = next(k for k, t in enumerate(finals) if "Lucy" in t and "dog, Max" in t)
    print("story", i, repr(finals[i][:120]))

    def per_level(idx):
        full, _ = level_snapshots(m.vq, idx[i:i + 1])
        with torch.no_grad():
            return [text(m.tk, m.dec(z), m.eos)[0] for z in full]

    out = dict(story=i, seed=SEED, level=LEVEL, strength=STRENGTH, levels=list(m.levels),
               base=per_level(base), pushed=per_level(pushed))
    print("pushed final:", repr(out["pushed"][-1][:120]))
    os.makedirs(os.path.join(WORK, "anim"), exist_ok=True)
    json.dump(out, open(os.path.join(WORK, "anim", "cat_swap_drafts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
