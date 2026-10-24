"""TinyStories code swap: the Sudoku swap test on NCM's unconditional story model (DJ + Claude, 2026-10-21).

Story A = what a seed produces. Swap run: the SAME seed (same random draws), but levels 1..k are forced to story B's
codes (B = the previous story in the batch). The sampler is still called at forced levels so the random stream stays
aligned; re-injecting A's own codes must reproduce A exactly. Controls: A's own codes, uniformly random codes.
Measures over content words (alphabetic, >= 3 letters, not stopwords):
  LEAK   share of B's words that aren't in A which appear in the final story
  KEEP   share of A's words that aren't in B which survive
  chance the same shares for an unrelated story C
Predictions are in HYPOTHESES.md (written first).

  python probe/stories_swap.py --n 128
"""
import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from jlens import DEV, NCM  # noqa: E402
from negation import STOP  # noqa: E402
from paths import WORK  # noqa: E402

KS = [1, 2, 3, 5, 7, 9, 11]


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w not in STOP}


def share(part, of):
    return len(part & of) / len(of) if of else float("nan")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default=os.path.join(WORK, "stories_swap"))
    ap.add_argument("--ks", default=",".join(map(str, KS)), help="levels to swap through, e.g. 1,3,6")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    os.makedirs(a.out, exist_ok=True)

    m = NCM()
    levels = m.levels
    starts = [sum(levels[:i]) for i in range(len(levels))]
    V = [int(e.weight.shape[0]) for e in m.vq.embeddings]
    orig = m.net._sample_level
    forced = {"codes": None, "k": 0, "i": 0}

    def sample(logits, *args):                    # generator call i produces codebook level i+1
        i = forced["i"]
        forced["i"] += 1
        out = orig(logits, *args)                 # always draw, so the random stream matches the unswapped run
        lvl = i + 1
        if lvl < forced["k"]:
            return forced["codes"][:, starts[lvl]:starts[lvl] + levels[lvl]].to(out.device)
        return out
    m.net._sample_level = sample

    def run(codes=None, k=0):
        torch.manual_seed(a.seed)
        l0 = torch.multinomial(m.l0_p, a.n * levels[0], replacement=True).view(a.n, levels[0]).to(DEV)
        if codes is not None and k >= 1:
            l0 = codes[:, :levels[0]].to(DEV)
        forced.update(codes=codes, k=k if codes is not None else 0, i=0)
        torch.manual_seed(a.seed + 1)
        with torch.no_grad():
            idx = m.net.generate(l0, torch.full((a.n, 1), m.eos, device=DEV), temperature=1.0, top_k=0, top_p=1.0)
        return idx.cpu(), m.text(idx)

    base_idx, A = run()
    again_idx, again = run()
    print(f"{a.n} stories. Same seed twice reproduces: {sum(x == y for x, y in zip(A, again)) / a.n:.0%} of texts")
    perm = torch.roll(torch.arange(a.n), 1)
    B = [A[i] for i in perm.tolist()]
    Cn = [A[i] for i in torch.roll(torch.arange(a.n), 2).tolist()]
    WA, WB, WC = [words(t) for t in A], [words(t) for t in B], [words(t) for t in Cn]
    uB = [b - x for b, x in zip(WB, WA)]
    uA = [x - b for b, x in zip(WB, WA)]
    mean = lambda v: sum(v) / len(v)
    chance_leak = mean([share(c, u) for c, u in zip(WC, uB)])
    chance_keep = mean([share(c, u) for c, u in zip(WC, uA)])
    print(f"chance: an unrelated story contains {chance_leak:.1%} of B's distinctive words and {chance_keep:.1%} of A's")

    g = torch.Generator().manual_seed(a.seed)
    rand = torch.cat([torch.randint(0, V[i], (a.n, n), generator=g) for i, n in enumerate(levels)], 1)

    def score(T):
        WT = [words(t) for t in T]
        leak = [share(w, u) for w, u in zip(WT, uB)]
        keep = [share(w, u) for w, u in zip(WT, uA)]
        return dict(leak=mean(leak), keep=mean(keep), follows_b=mean([float(l > k) for l, k in zip(leak, keep)]),
                    same_as_a=mean([float(t == x) for t, x in zip(T, A)]))

    rows, examples = [], {}
    print("\n  k  codes | own codes: = A   | DONOR: leak  keep  follows B | RANDOM: leak  keep")
    for k in ks:
        own = score(run(base_idx, k)[1])
        _, Td = run(base_idx[perm], k)
        don = score(Td)
        rnd = score(run(rand, k)[1])
        rows.append(dict(k=k, codes=sum(levels[:k]), own=own, donor=don, random=rnd))
        examples[k] = dict(A=A[0], B=B[0], swapped=Td[0])
        print(f" {k:>2} {sum(levels[:k]):>5}  |      {own['same_as_a']:5.0%}        |  {don['leak']:6.1%} {don['keep']:6.1%}   {don['follows_b']:5.0%}"
              f"    |  {rnd['leak']:6.1%} {rnd['keep']:6.1%}", flush=True)
    for k in [x for x in (1, 3, 7) if x in examples]:
        e = examples[k]
        print(f"\nEXAMPLE k={k}\n  A: {e['A'][:230]!r}\n  B: {e['B'][:230]!r}\n  swapped: {e['swapped'][:230]!r}")
    json.dump(dict(n=a.n, seed=a.seed, chance_leak=chance_leak, chance_keep=chance_keep, rows=rows, examples=examples),
              open(os.path.join(a.out, "stories_swap.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
