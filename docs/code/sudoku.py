"""Sudoku: does NCM hold a menu of candidate digits per cell, and actively push the wrong ones down? (DJ + Claude.)

Same question as negation.py, but with ground truth. The puzzle (91 tokens) is the prefix; the model writes the
89-token solution (81 digits + 8 '|') coarse-to-fine. After each level, decode the cumulative codes and read each
cell's logits over the 9 digit tokens, CENTRED over those 9. Predictions are in HYPOTHESES.md (written first).

  python probe/sudoku.py --task hard --n 1000
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from drafts import level_snapshots, load  # noqa: E402
from paths import ARTIFACTS, DATA, WORK  # noqa: E402

SEP, DIG = 10, slice(1, 10)


def legal_marks(puz):
    """For each of the 89 output positions: bool[9] of digits not ruled out by the clues in its row/col/box."""
    grid = [[0] * 9 for _ in range(9)]
    for p, t in enumerate(puz):
        if p % 10 != 9:
            grid[p // 10][p % 10] = t
    out = torch.zeros(89, 9, dtype=torch.bool)
    for p in range(89):
        if p % 10 == 9:
            continue
        r, c = p // 10, p % 10
        seen = set(grid[r]) | {grid[i][c] for i in range(9)}
        br, bc = r // 3 * 3, c // 3 * 3
        seen |= {grid[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)}
        for d in range(1, 10):
            out[p, d - 1] = d not in seen
    return out


def valid(sol, puz):
    """Is a predicted 89-token grid a complete valid Sudoku that agrees with the puzzle's clues?"""
    if any(sol[p] != SEP for p in range(9, 89, 10)):
        return False
    g = [[sol[r * 10 + c] for c in range(9)] for r in range(9)]
    full = set(range(1, 10))
    rows = all(set(r) == full for r in g)
    cols = all({g[r][c] for r in range(9)} == full for c in range(9))
    boxes = all({g[br + i][bc + j] for i in range(3) for j in range(3)} == full for br in (0, 3, 6) for bc in (0, 3, 6))
    clues = all(puz[p] in (0, sol[p]) for p in range(89) if p % 10 != 9)
    return rows and cols and boxes and clues


def grid_txt(ids):
    ch = "_123456789|B"
    return "".join(ch[i] for i in ids).replace("|", "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hard")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=125)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(WORK, "sudoku"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ck = os.path.join(ARTIFACTS, "sudoku", a.task)
    cfg, tk, net, _ = load(ck, os.path.join(ck, "config.yaml"))
    vq, dec = tk.vqvae.vq, tk.vqvae.decoder
    dd = os.path.join(DATA, "sudoku", a.task)
    puzzles = pickle.load(open(os.path.join(dd, "puzzles.pickle"), "rb"))[:a.n]
    sols = pickle.load(open(os.path.join(dd, "solutions.pickle"), "rb"))[:a.n]
    assert len(puzzles[0]) == 91 and len(sols[0]) == 89, (len(puzzles[0]), len(sols[0]))

    t0 = time.time()
    lv = []                                                          # per batch: levels x b x 89 x 12
    for i in range(0, a.n, a.bs):
        x = torch.tensor(puzzles[i:i + a.bs], device="cuda")
        with torch.no_grad():
            idx = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
            full, _ = level_snapshots(vq, idx)
            lv.append(torch.stack([dec(z) for z in full]).float().cpu())
    logits = torch.cat(lv, 1)                                        # L x N x 89 x 12
    L, N = logits.shape[:2]
    print(f"{a.task}: {N} puzzles, {L} levels, generation+decode {time.time() - t0:.1f}s")

    S = torch.tensor(sols)                                           # N x 89
    Pz = torch.tensor([p[1:90] for p in puzzles])                    # N x 89, aligned to output positions
    cell = S != SEP
    clue = cell & (Pz != 0)
    blank = cell & (Pz == 0)
    final = logits[-1].argmax(-1)
    exact = (final == S).all(-1)
    val = torch.tensor([valid(final[k].tolist(), puzzles[k][1:90]) for k in range(N)])
    print(f"\nS1  exact match {exact.float().mean():.1%}   valid+clue-consistent {val.float().mean():.1%}   "
          f"blank-cell acc {(final == S)[blank].float().mean():.2%}   clue-cell acc {(final == S)[clue].float().mean():.2%}")

    # S2: drafts. Cell accuracy = argmax over the whole 12-token vocab, by level.
    print("\nS2  draft accuracy by level (argmax of the decoded draft):")
    print("     level  codes   sep    clue   blank  | digits-only argmax: clue  blank")
    am = logits.argmax(-1)
    dam = logits[..., DIG].argmax(-1) + 1
    s2 = []
    for k in range(L):
        row = dict(level=k + 1, codes=int(vq.levels[k]), sep=(am[k] == S)[~cell].float().mean().item(),
                   clue=(am[k] == S)[clue].float().mean().item(), blank=(am[k] == S)[blank].float().mean().item(),
                   clue_d=(dam[k] == S)[clue].float().mean().item(), blank_d=(dam[k] == S)[blank].float().mean().item())
        s2.append(row)
        print(f"     L{k + 1:<4} {row['codes']:>5}  {row['sep']:6.1%} {row['clue']:6.1%} {row['blank']:6.1%}  |"
              f"                    {row['clue_d']:6.1%} {row['blank_d']:6.1%}")

    # digit distributions per level
    P = logits[..., DIG].softmax(-1)                                 # L x N x 89 x 9
    C = logits[..., DIG] - logits[..., DIG].mean(-1, keepdim=True)   # centred over the 9 digits
    menu = P >= a.tau
    msize = [menu[k][blank].sum(-1).float().mean().item() for k in range(L)]
    ref = max(range(L), key=lambda k: msize[k])
    print("\nS3  mean menu size (digits with p >= %.2f) per BLANK cell by level: " % a.tau
          + " ".join(f"L{k + 1}:{m:.2f}" for k, m in enumerate(msize)))
    print(f"    REFERENCE level (widest menu, rule fixed in advance) = L{ref + 1}")
    tgt = (S.clamp(1, 9) - 1)                                        # true digit index 0..8 (only valid on cells)
    on_menu = menu[ref].gather(-1, tgt[..., None]).squeeze(-1)
    print(f"    true digit on the L{ref + 1} menu: blank {on_menu[blank].float().mean():.1%}   clue {on_menu[clue].float().mean():.1%}")
    legal = torch.stack([legal_marks(p[1:90]) for p in puzzles])     # N x 89 x 9
    mb, lb = menu[ref][blank], legal[blank]
    frac_legal_menu = (mb & lb).sum() / mb.sum()
    base = lb.float().mean()
    print(f"    menu digits that are legal pencil marks {frac_legal_menu:.1%}  vs base rate {base:.1%}  "
          f"-> enrichment {frac_legal_menu / base:.2f}x   (legal marks per blank cell {lb.sum(-1).float().mean():.2f})")
    enrich = []
    for k in range(L):
        mk = menu[k][blank]
        enrich.append(((mk & lb).sum() / mk.sum().clamp_min(1) / base).item())
    print("    enrichment by level: " + " ".join(f"L{k + 1}:{e:.2f}" for k, e in enumerate(enrich)))

    # S4: erasure. Blank cells only. CHOSEN = final digit; REJECTED = p>=tau at ref, not chosen; NEUTRAL = p<0.01 at ref.
    fin = final.clamp(1, 9) - 1
    cb = blank & (final >= 1) & (final <= 9)
    Cb = C[:, cb]                                                    # L x M x 9
    Pr = P[ref][cb]                                                  # M x 9
    ch = torch.zeros_like(Pr, dtype=torch.bool).scatter_(1, fin[cb][:, None], True)
    grp = {"chosen": ch, "rejected": (Pr >= a.tau) & ~ch, "neutral": (Pr < 0.01) & ~ch}
    print(f"\nS4  erasure, {int(cb.sum())} blank cells: rejected {int(grp['rejected'].sum())}, neutral {int(grp['neutral'].sum())}")
    print("    mean ABSOLUTE centred logit by level (L1..L12):")
    s4 = {}
    for g, m in grp.items():
        traj = torch.stack([Cb[k][m].mean() for k in range(L)])
        se = torch.stack([Cb[k][m].std() / m.sum().sqrt() for k in range(L)])
        s4[g] = dict(abs=traj.tolist(), se=se.tolist(), change=(traj - traj[ref]).tolist(), n=int(m.sum()))
        print(f"    {g:<9}" + "".join(f"{v:+7.2f}" for v in traj.tolist()))
    print("    change vs reference L%d:" % (ref + 1))
    for g in grp:
        print(f"    {g:<9}" + "".join(f"{v:+7.2f}" for v in s4[g]["change"]))
    gap = s4["rejected"]["abs"][-1] - s4["neutral"]["abs"][-1]
    # paired version: per cell, mean rejected minus mean neutral at final level (cells that have both)
    both = grp["rejected"].any(-1) & grp["neutral"].any(-1)
    fz = Cb[-1][both]
    rj = (fz * grp["rejected"][both]).sum(-1) / grp["rejected"][both].sum(-1)
    nt = (fz * grp["neutral"][both]).sum(-1) / grp["neutral"][both].sum(-1)
    d = rj - nt
    print(f"    final-level gap rejected - neutral: {gap:+.2f} (pooled);  paired per cell {d.mean():+.2f} ± {d.std() / len(d) ** 0.5:.2f}"
          f" over {len(d)} cells; share of cells where rejected ends below neutral {(d < 0).float().mean():.0%}")
    rawd = logits[..., DIG][:, cb]                                   # uncentred check
    rr = torch.stack([rawd[k][grp["rejected"]].mean() - rawd[k][grp["neutral"]].mean() for k in range(L)])
    print("    uncentred check, rejected - neutral raw logit by level: " + " ".join(f"{v:+.2f}" for v in rr.tolist()))
    # per-level increments
    print("    per-level increment (rejected minus neutral): " + " ".join(
        f"L{k + 1}:{(s4['rejected']['abs'][k] - s4['rejected']['abs'][k - 1]) - (s4['neutral']['abs'][k] - s4['neutral']['abs'][k - 1]):+.2f}"
        for k in range(1, L)))

    # S5: failures
    wrong = blank & (final != S)
    s5 = None
    if wrong.any():
        wm = on_menu[wrong].float().mean().item()
        bad_rows = (~exact).sum().item()
        print(f"\nS5  {bad_rows} wrong grids, {int(wrong.sum())} wrong blank cells; true digit on the L{ref + 1} menu for {wm:.1%} of them")
        # when do wrong cells go wrong? first level from which the cell's digit-argmax equals the final (wrong) digit and stays
        commit = []
        for k in range(L):
            commit.append(((dam[k:] == final[None]).all(0) & wrong).sum().item() / wrong.sum().item())
        print("    wrong cells already locked on their final wrong digit, from level: " + " ".join(f"L{k + 1}:{c:.0%}" for k, c in enumerate(commit)))
        lk = [((dam[k:] == final[None]).all(0) & blank & (final == S)).sum().item() / (blank & (final == S)).sum().item() for k in range(L)]
        print("    right cells locked on their final digit, from level:           " + " ".join(f"L{k + 1}:{c:.0%}" for k, c in enumerate(lk)))
        s5 = dict(wrong_grids=bad_rows, wrong_cells=int(wrong.sum()), true_on_menu=wm, commit_wrong=commit, commit_right=lk)

    # an example draft, level by level
    ex = int((~exact).nonzero()[0]) if (~exact).any() else 0
    print(f"\nEXAMPLE puzzle {ex} ({'wrong' if not exact[ex] else 'right'}):\npuzzle:\n{grid_txt(puzzles[ex][1:90])}")
    for k in (0, 2, 4, 6, 8, 11):
        print(f"\ndraft after L{k + 1}:\n{grid_txt(am[k, ex].tolist())}")
    print(f"\nsolution:\n{grid_txt(sols[ex])}")

    json.dump(dict(task=a.task, n=N, tau=a.tau, ref=ref + 1, exact=exact.float().mean().item(), valid=val.float().mean().item(),
                   s2=s2, menu_size=msize, true_on_menu_blank=on_menu[blank].float().mean().item(), enrichment=enrich,
                   s4=s4, final_gap=gap, paired_gap=d.mean().item(), paired_se=(d.std() / len(d) ** 0.5).item(),
                   share_below=(d < 0).float().mean().item(), raw_gap=rr.tolist(), s5=s5),
              open(os.path.join(a.out, f"sudoku_{a.task}.json"), "w"), indent=1)
    torch.save(dict(logits=logits.half(), sols=S, puz=Pz), os.path.join(a.out, f"sudoku_{a.task}_logits.pt"))


if __name__ == "__main__":
    main()
