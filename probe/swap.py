"""Concept swap (dog -> cat) through the J-lens, in NCM and in the word-by-word control (DJ + Claude, 2026-10-20).

NCM: during generation, at one level, add  strength * |h| * unit(d_cat - d_dog)  to the last generator layer (input
to ln_f) at every code position of that level, where d_w = unit(W_dec[w] @ M[level, 12]) is word w's J-lens
direction. Later levels generate normally. Every condition uses the same level-1 codes and the same sampling seed,
so baseline and swapped runs only differ because of the push.
Control: same push at the " dog" token of a prompt, at one layer, then sample a continuation with the same seed.
Third condition everywhere: a random unit direction with the same strength.

  python probe/swap.py ncm  --n 64 --strengths 1,2,4
  python probe/swap.py control --strengths 1,2,4
"""
import argparse
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from jlens import DEV, OUT, NCM, Control, unit  # noqa: E402


def has(text, w):
    t = " " + text.lower().replace("\n", " ") + " "
    return any(f" {w}{p}" in t for p in (" ", ".", ",", "!", "?", "'", '"'))


def share(texts, w):
    return sum(has(t, w) for t in texts) / max(len(texts), 1)


ANIMAL = {"cat", "cats", "dog", "dogs", "kitten", "kittens", "puppy", "puppies", "kitty"}


def kept(base, other):
    """Mean share of each baseline story's words (minus cat/dog words) that survive in the edited story (Jaccard)."""
    out = []
    for a, b in zip(base, other):
        A = set(re.findall(r"[a-z]+", a.lower())) - ANIMAL
        B = set(re.findall(r"[a-z]+", b.lower())) - ANIMAL
        out.append(len(A & B) / max(len(A | B), 1))
    return sum(out) / max(len(out), 1)


# ----------------------------------------------------------------------------------------------- NCM
class Push:
    """ln_f pre-hook that fires on the j-th call (= generator level j) and pushes the code positions."""

    def __init__(self, j, direction, strength, P):
        self.j, self.d, self.s, self.P, self.calls = j, direction, strength, P, 0

    def __call__(self, mod, args):
        x = args[0]
        hit = self.calls == self.j
        self.calls += 1
        if not hit or self.d is None:
            return None
        h = x[:, self.P:, :]
        scale = h.norm(dim=-1, keepdim=True).mean()
        x = x.clone()
        x[:, self.P:, :] = h + self.s * scale * self.d
        return (x,)


def run_ncm(a):
    m = NCM()
    Ms = torch.load(f"{OUT}/ncm_M.pt", weights_only=True)["M"]
    pt = m.tk.pre_tokenizer
    cat, dog = pt.encode(" cat")[0], pt.encode(" dog")[0]
    g = torch.Generator().manual_seed(0)
    rand = unit(torch.randn(768, generator=g)).to(DEV)
    torch.manual_seed(a.seed)
    l0 = torch.multinomial(m.l0_p, a.n * m.levels[0], replacement=True).view(a.n, m.levels[0]).to(DEV)

    def gen(push):
        h = m.net.transformer.ln_f.register_forward_pre_hook(push) if push else None
        try:
            torch.manual_seed(a.seed + 1)
            with torch.no_grad():
                idx = m.net.generate(l0, torch.full((a.n, 1), m.eos, device=DEV), temperature=1.0, top_k=0, top_p=1.0)
            return m.text(idx)
        finally:
            if h:
                h.remove()

    base = gen(None)
    print(f"NCM baseline ({a.n} stories): cat {share(base, 'cat'):.0%}  dog {share(base, 'dog'):.0%}", flush=True)
    rows = []
    for level in a.levels:
        j = level - 2                                    # generator level index (codebook level j+1 = story level L{j+2})
        W = m.W @ Ms[j, 12].to(DEV)                      # 8192 x 768 lens directions at the last layer
        d = unit(unit(W[cat]) - unit(W[dog]))
        for s in a.strengths:
            sw = gen(Push(j, d, s, m.P))
            rn = gen(Push(j, rand, s, m.P))
            changed = sum(x != y for x, y in zip(sw, base)) / a.n
            rows.append((level, s, share(sw, "cat"), share(sw, "dog"), share(rn, "cat"), share(rn, "dog"), changed))
            print(f"  L{level:<2} strength {s:>3}:  SWAP cat {share(sw, 'cat'):>4.0%} dog {share(sw, 'dog'):>4.0%}"
                  f"   | RANDOM cat {share(rn, 'cat'):>4.0%} dog {share(rn, 'dog'):>4.0%}   | stories changed {changed:.0%}"
                  f"   | words kept: swap {kept(base, sw):.0%} random {kept(base, rn):.0%}", flush=True)
            if a.show and s == a.strengths[-1]:
                for i in range(a.n):
                    if has(base[i], "dog") and not has(sw[i], "dog"):
                        print(f"     before: {base[i][:220]!r}\n     after : {sw[i][:220]!r}")
                        break
    return rows


# ----------------------------------------------------------------------------------------------- control
PROMPTS = [
    "Once upon a time, there was a little dog",
    "One day, Lily found a small dog",
    "Tom had a big brown dog",
    "There was a happy dog",
    "Once upon a time, a little girl had a dog",
    "Ben saw a dog",
    "Sara wanted to play with her dog",
    "Once upon a time, there was a lazy dog",
]


def run_control(a):
    c = Control()
    Ms = torch.load(f"{OUT}/control_M.pt", weights_only=True)["M"]
    cat, dog = c.tok.encode(" cat")[0], c.tok.encode(" dog")[0]
    g = torch.Generator().manual_seed(0)
    rand = unit(torch.randn(768, generator=g)).to(DEV)
    per = max(1, a.n // len(PROMPTS))

    def gen(layer, direction, strength):
        texts = []
        for p in PROMPTS:
            ids = torch.tensor([c.tok.encode(p)] * per, device=DEV)
            pos = ids.shape[1] - 1                     # the " dog" token ends every prompt
            hook = None
            if direction is not None:
                blk = c.m.transformer.h[layer] if layer < len(c.m.transformer.h) else c.m.transformer.ln_f

                def pre(mod, args, kwargs=None):
                    x = args[0] if args else kwargs["hidden_states"]
                    if x.shape[1] <= pos:              # later cached steps: only new tokens, leave them
                        return None
                    x = x.clone()
                    x[:, pos, :] = x[:, pos, :] + strength * x[:, pos, :].norm(dim=-1, keepdim=True) * direction
                    if kwargs is None:                 # plain pre-hook (ln_f)
                        return (x,) + tuple(args[1:])
                    if args:
                        return (x,) + tuple(args[1:]), kwargs
                    return args, dict(kwargs, hidden_states=x)
                hook = blk.register_forward_pre_hook(pre, with_kwargs=True) if blk is not c.m.transformer.ln_f \
                    else blk.register_forward_pre_hook(lambda mod, args: pre(mod, args))
            try:
                torch.manual_seed(a.seed)
                out = c.m.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, top_k=0, top_p=1.0,
                                   temperature=1.0, max_new_tokens=80, pad_token_id=c.eos)
            finally:
                if hook:
                    hook.remove()
            texts += c.tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        return texts

    base = gen(0, None, 0)
    print(f"CONTROL baseline ({len(base)} continuations after '...dog'): cat {share(base, 'cat'):.0%}  dog {share(base, 'dog'):.0%}", flush=True)
    rows = []
    for layer in a.layers:
        W = c.W @ Ms[layer].to(DEV)
        d = unit(unit(W[cat]) - unit(W[dog]))
        for s in a.strengths:
            sw, rn = gen(layer, d, s), gen(layer, rand, s)
            rows.append((layer, s, share(sw, "cat"), share(sw, "dog"), share(rn, "cat"), share(rn, "dog")))
            print(f"  layer {layer} strength {s:>3}:  SWAP cat {share(sw, 'cat'):>4.0%} dog {share(sw, 'dog'):>4.0%}"
                  f"   | RANDOM cat {share(rn, 'cat'):>4.0%} dog {share(rn, 'dog'):>4.0%}", flush=True)
            if a.show and s == a.strengths[-1]:
                print(f"     e.g. after swap: {PROMPTS[0]}{sw[0][:160]!r}")
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["ncm", "control"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--strengths", default="1,2,4")
    ap.add_argument("--levels", default="3,5,7,9")
    ap.add_argument("--layers", default="1,2,3")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    a.strengths = [float(x) for x in a.strengths.split(",")]
    a.levels = [int(x) for x in a.levels.split(",")]
    a.layers = [int(x) for x in a.layers.split(",")]
    run_ncm(a) if a.model == "ncm" else run_control(a)
