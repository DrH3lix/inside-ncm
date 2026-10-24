"""Do NCM's finer levels actively ERASE the menu options the story didn't use? (Gemini's prediction; DJ + Claude.)

For each story: decode the draft after every level. The level-3 "menu" = content words the L3 draft gives
p >= TAU somewhere. CHOSEN = appears in the final text; REJECTED = never does. At the positions where a word was
drafted, track its CENTRED logit (its logit minus the mean logit over the vocabulary at that position, so
whole-distribution drift cancels), relative to its L3 value, level by level. Active erasure = a rejected word's
centred logit going DOWN; passive fading = it stays roughly flat while chosen words climb past it.
Neutral baseline: random content words that were never on the menu, at the same positions.

  python probe/negation.py --n 64
"""
import argparse
import os
import random
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from drafts import level_snapshots  # noqa: E402
from jlens import DEV, NCM  # noqa: E402

STOP = set("""the a an and or but so then than that this these those there their they them he she him her his hers it its
was were is are be been being have has had do does did not no yes to of in on at by for with from up down out over
under into onto about as if when while after before because very too also just one day once upon time said says say
again all any some many much more most other another such own same can could will would shall should may might must
we us our you your i me my mine what who whom which where why how""".split())


def content_ids(tk, V):
    """Token ids whose decoded text is an alphabetic word of length >= 3 and not a stopword."""
    keep = []
    for i in range(V):
        w = tk.pre_tokenizer.decode([i]).strip()
        if re.fullmatch(r"[A-Za-z]{3,}", w or "") and w.lower() not in STOP:
            keep.append(i)
    return torch.tensor(keep, device=DEV)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.10)
    a = ap.parse_args()

    m = NCM()
    vq, dec = m.vq, m.dec
    V = m.W.shape[0]
    content = content_ids(m.tk, V)
    is_content = torch.zeros(V, dtype=torch.bool, device=DEV)
    is_content[content] = True
    word = lambda i: m.tk.pre_tokenizer.decode([i]).strip()

    def stories(n, seed):
        idx = m.sample(n, seed)
        full, _ = level_snapshots(vq, idx)
        with torch.no_grad():
            logits = torch.stack([dec(z) for z in full])          # levels x n x 128 x V
        return idx, logits

    idx, logits = stories(a.n, a.seed)
    nlev = logits.shape[0]
    L3 = 2                                                        # codebook index of story level L3
    probs3 = logits[L3].softmax(-1)
    final_ids = logits[-1].argmax(-1)                             # n x 128
    cent = logits.sub_(logits.mean(-1, keepdim=True))             # centred logits, in place (~3 GB tensor)

    groups = {"chosen": [], "rejected": [], "neutral": []}
    rng = random.Random(0)
    examples = []
    for s in range(a.n):
        present = set(final_ids[s].tolist())
        p = probs3[s]                                             # 128 x V
        maxp, _ = p.max(0)
        menu = ((maxp >= a.tau) & is_content).nonzero().flatten().tolist()
        if not menu:
            continue
        for w in menu:
            pos = (p[:, w] >= a.tau).nonzero().flatten()
            traj = cent[:, s, pos, w].mean(-1)                    # levels
            traj = (traj - traj[L3]).tolist()
            grp = "chosen" if w in present else "rejected"
            groups[grp].append(traj)
            if grp == "rejected" and len(examples) < 12:
                examples.append((s, word(w), traj))
            # neutral: a random content word never on this story's menu, at the same positions
            for _ in range(20):
                nw = content[rng.randrange(len(content))].item()
                if maxp[nw] < 0.01 and nw not in present:
                    t2 = cent[:, s, pos, nw].mean(-1)
                    groups["neutral"].append((t2 - t2[L3]).tolist())
                    break

    print(f"{a.n} stories; menu words with p >= {a.tau} in the L3 draft: chosen {len(groups['chosen'])}, "
          f"rejected {len(groups['rejected'])}; neutral samples {len(groups['neutral'])}\n")
    print("Mean change in centred logit vs L3, at the positions where the word was drafted (negative = pushed down):")
    hdr = "level      " + "".join(f"  L{k + 1:<5}" for k in range(L3, nlev))
    print(hdr)
    stats = {}
    for g, rows in groups.items():
        if not rows:
            continue
        t = torch.tensor(rows)
        mean, se = t.mean(0), t.std(0) / max(len(rows), 1) ** 0.5
        stats[g] = (mean, se)
        print(f"{g:<10} " + "".join(f" {mean[k]:+6.2f} " for k in range(L3, nlev)))
        print(f"{'  ±se':<10} " + "".join(f"  {se[k]:5.2f} " for k in range(L3, nlev)))
    print("\nPer-level INCREMENT (what each level does on its own), rejected vs chosen vs neutral:")
    for g in ("rejected", "chosen", "neutral"):
        if g in stats:
            mean = stats[g][0]
            inc = [mean[k] - mean[k - 1] for k in range(L3 + 1, nlev)]
            print(f"{g:<10} " + "".join(f"  L{k + 1}:{v:+5.2f}" for k, v in zip(range(L3 + 1, nlev), inc)))
    if "rejected" in groups and groups["rejected"]:
        t = torch.tensor(groups["rejected"])
        frac_down = (t[:, L3 + 2] < -0.5).float().mean().item()       # by L5
        print(f"\nShare of rejected words pushed down by >0.5 logit by L5: {frac_down:.0%}  "
              f"(neutral: {(torch.tensor(groups['neutral'])[:, L3 + 2] < -0.5).float().mean().item():.0%})")
    print("\nExamples (rejected word: centred-logit change by level L4..L12):")
    for s, w, traj in examples:
        print(f"  story {s:>2} {w!r:>12}: " + " ".join(f"{v:+.1f}" for v in traj[L3 + 1:]))

    # case study: the fish story from the J-lens readout (seed 999, story 0)
    idx2, lg2 = stories(8, 999)
    c2 = lg2 - lg2.mean(-1, keepdim=True)
    txt = m.text(idx2[0:1])[0]
    print("\nCASE STUDY - the Fin story:", txt[:140].replace("\n", " "))
    print("max centred logit over positions, by level L3..L12 (higher = more present in the draft):")
    for w in [" Fin", " Finny", " Bubbles", " shark", " dolphin", " crab", " duck", " swan", " pond", " beach"]:
        ids = m.tk.pre_tokenizer.encode(w)
        if len(ids) != 1:
            print(f"  {w!r:>10}: (multi-token, skipped)")
            continue
        traj = c2[L3:, 0, :, ids[0]].max(-1).values.tolist()
        print(f"  {w!r:>10}: " + " ".join(f"{v:5.1f}" for v in traj) + ("   <- in final story" if has_word(txt, w) else ""))


def has_word(text, w):
    return re.search(r"\b" + re.escape(w.strip()) + r"\b", text) is not None


if __name__ == "__main__":
    main()
