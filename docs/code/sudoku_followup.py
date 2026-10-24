"""Sudoku follow-ups, POST HOC (written after reading sudoku.py's output; see HYPOTHESES.md).

The pre-registered reference rule (widest menu) picked L1, where ~6 of 9 digits have p >= 0.1 - fog, not a menu.
So: (A) run the erasure comparison from EVERY reference level instead of choosing one; (B) a pencil-mark test that
needs no reference level: by level, the centred logit of the true digit vs wrong-but-LEGAL digits (not ruled out
by the clues) vs ILLEGAL digits (ruled out by a clue in the cell's row/column/box). The decoder never sees the
puzzle, so any gap between legal-wrong and illegal has to travel through the codes; (C) for wrong cells, where the
true digit ranks.

  python probe/sudoku_followup.py --task hard
"""
import argparse
import json
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sudoku import legal_marks, SEP, DIG  # noqa: E402
from paths import DATA, WORK  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hard")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(WORK, "sudoku"))
    a = ap.parse_args()
    d = torch.load(os.path.join(a.out, f"sudoku_{a.task}_logits.pt"), weights_only=True)
    logits, S, Pz = d["logits"].float(), d["sols"], d["puz"]
    L, N = logits.shape[:2]
    puzzles = pickle.load(open(os.path.join(DATA, "sudoku", a.task, "puzzles.pickle"), "rb"))[:N]
    cell = S != SEP
    blank = cell & (Pz == 0)
    final = logits[-1].argmax(-1)
    D = logits[..., DIG]
    P = D.softmax(-1)
    C = D - D.mean(-1, keepdim=True)
    tgt = S.clamp(1, 9) - 1
    out = {}

    # (A) erasure from every reference level. Blank cells whose final digit is a digit.
    cb = blank & (final >= 1) & (final <= 9)
    fin = final[cb] - 1
    Cb, Pb = C[:, cb], P[:, cb]
    ch = torch.zeros(Pb.shape[1:], dtype=torch.bool).scatter_(1, fin[:, None], True)
    print(f"{a.task}: {N} puzzles, {int(cb.sum())} blank cells\n")
    print("(A) erasure from every reference level k. REJECTED = p>=%.2f at k, not chosen; NEUTRAL = p<0.01 at k, not chosen." % a.tau)
    print("    ref  menu  n_rej   n_neu   rej@k  neu@k  ->  rej@L12 neu@L12 | paired final gap (rej-neu)  share rej<neu | change k->L12 rej  neu")
    sweep = []
    for k in range(L - 1):
        rej = (Pb[k] >= a.tau) & ~ch
        neu = (Pb[k] < 0.01) & ~ch
        if rej.sum() == 0 or neu.sum() == 0:
            continue
        both = rej.any(-1) & neu.any(-1)
        f = Cb[-1][both]
        gap = (f * rej[both]).sum(-1) / rej[both].sum(-1) - (f * neu[both]).sum(-1) / neu[both].sum(-1)
        r = dict(ref=k + 1, menu=((Pb[k] >= a.tau).sum(-1).float().mean().item()), n_rej=int(rej.sum()), n_neu=int(neu.sum()),
                 rej_k=Cb[k][rej].mean().item(), neu_k=Cb[k][neu].mean().item(),
                 rej_f=Cb[-1][rej].mean().item(), neu_f=Cb[-1][neu].mean().item(),
                 gap=gap.mean().item(), se=(gap.std() / len(gap) ** 0.5).item(), cells=len(gap),
                 below=(gap < 0).float().mean().item())
        r["d_rej"], r["d_neu"] = r["rej_f"] - r["rej_k"], r["neu_f"] - r["neu_k"]
        sweep.append(r)
        print(f"    L{k + 1:<3} {r['menu']:4.2f} {r['n_rej']:7d} {r['n_neu']:7d}  {r['rej_k']:+6.2f} {r['neu_k']:+6.2f}  ->  "
              f"{r['rej_f']:+6.2f}  {r['neu_f']:+6.2f} | {r['gap']:+6.2f} ± {r['se']:.2f} ({r['cells']:5d} cells)   {r['below']:5.0%}   |"
              f"   {r['d_rej']:+6.2f} {r['d_neu']:+6.2f}")
    out["sweep"] = sweep

    # (B) pencil marks. true / legal-wrong / illegal, by level, blank cells.
    legal = torch.stack([legal_marks(p[1:90]) for p in puzzles])     # N x 89 x 9
    lb = legal[blank]
    tb = torch.zeros_like(lb).scatter_(1, tgt[blank][:, None], True)
    Cl = C[:, blank]
    groups = {"true": tb, "legal-wrong": lb & ~tb, "illegal": ~lb}
    print("\n(B) pencil marks: mean centred logit by level, blank cells (no reference level needed)")
    print("    " + " " * 12 + "".join(f"  L{k + 1:<4}" for k in range(L)))
    pm = {}
    for g, m in groups.items():
        traj = [Cl[k][m].mean().item() for k in range(L)]
        pm[g] = traj
        print(f"    {g:<12}" + "".join(f" {v:+6.2f}" for v in traj))
    gapLI = [pm["legal-wrong"][k] - pm["illegal"][k] for k in range(L)]
    print(f"    {'legal-illeg':<12}" + "".join(f" {v:+6.2f}" for v in gapLI))
    # paired per cell at a few levels
    for k in (0, 5, 8, 10, 11):
        lw, il = groups["legal-wrong"], groups["illegal"]
        both = lw.any(-1) & il.any(-1)
        f = Cl[k][both]
        g = (f * lw[both]).sum(-1) / lw[both].sum(-1) - (f * il[both]).sum(-1) / il[both].sum(-1)
        print(f"    paired per cell at L{k + 1}: legal-wrong minus illegal {g.mean():+.2f} ± {g.std() / len(g) ** 0.5:.2f} ({len(g)} cells)")
    # probability mass on illegal digits, by level
    Pl = P[:, blank]
    mass = [(Pl[k] * (~lb)).sum(-1).mean().item() for k in range(L)]
    base = (~lb).float().mean().item()
    print("    probability mass on ILLEGAL digits by level (uniform would be %.1f%%): " % (100 * base)
          + " ".join(f"L{k + 1}:{100 * m:.1f}%" for k, m in enumerate(mass)))
    # among WRONG candidates only: are they legal more often than chance?
    menu = P[:, blank] >= a.tau
    wrong_menu_legal = []
    base_w = ((lb & ~tb).sum(-1).float() / 8).mean().item()
    for k in range(L):
        wm = menu[k] & ~tb
        wrong_menu_legal.append(((wm & lb).sum() / wm.sum().clamp_min(1)).item())
    print(f"    share of WRONG menu digits (p>=%.2f) that are legal, by level (chance {100 * base_w:.1f}%%): " % a.tau
          + " ".join(f"L{k + 1}:{100 * v:.0f}%" for k, v in enumerate(wrong_menu_legal)))
    out["pencil"] = dict(traj=pm, gap=gapLI, illegal_mass=mass, illegal_base=base, wrong_menu_legal=wrong_menu_legal, wrong_base=base_w)

    # (C) wrong cells: rank of the true digit
    wrong = blank & (final != S)
    if wrong.any():
        print(f"\n(C) {int(wrong.sum())} wrong blank cells: rank of the TRUE digit among 9 (1 = top), and is it legal?")
        for k in (8, 9, 10, 11):
            rk = (D[k][wrong] > D[k][wrong].gather(-1, tgt[wrong][:, None])).sum(-1) + 1
            hist = torch.bincount(rk, minlength=10)[1:].tolist()
            print(f"    L{k + 1}: " + " ".join(f"r{i + 1}:{h}" for i, h in enumerate(hist)) + f"   (runner-up share {hist[1] / sum(hist):.0%})")
        wd = final[wrong].clamp(1, 9) - 1
        wl = legal[wrong].gather(-1, wd[:, None]).squeeze(-1)
        print(f"    the WRONG digit written is a legal pencil mark in {wl.float().mean():.0%} of wrong cells "
              f"(chance among wrong digits {(legal[wrong].sum(-1).float() - 1).div(8).mean():.0%})")
        out["wrong_legal"] = wl.float().mean().item()
    json.dump(out, open(os.path.join(a.out, f"sudoku_{a.task}_followup.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
