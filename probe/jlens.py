"""Jacobian lens (J-lens) for two models trained on the same TinyStories data (DJ + Claude, 2026-10-20).

  control : roneneldan/TinyStories-33M  - an ordinary left-to-right GPT-Neo (4 layers, d=768)
  ncm     : Hazy Research NCM generator  - coarse-to-fine, 12 levels of codes, 12 layers, d=768, then a VQ-VAE decoder

Same definition for both, after Anthropic's J-lens ("average causal effect of changes in the residual stream on the
model's eventual outputs"):  for residual point l,  M_l = E_positions[ d( sum_t H_t ) / d h_l ]  where H is the final
normalised hidden state that feeds the word-output matrix W, summed over every position of the eventual text.  The lens
is J_l = W @ M_l  (one direction per word).  For NCM the path is: generator residual (at a code position of level k)
-> code head -> soft code (straight-through: the decoder still sees the real sampled story) -> upsample -> VQ-VAE
decoder -> H.  Later levels are held fixed (direct effect of this level's codes on the text).

J-space occupancy: how much of a centred residual a sparse (K=25) non-negative mix of lens directions explains,
against three baselines: the same residuals randomly rotated (the main null: same lengths and spectrum, no
alignment with any word), 25 random words' lens directions, and the plain unembedding rows ("logit lens").

  python probe/jlens.py fit     --model control|ncm [--batches 4 --bs 16]
  python probe/jlens.py analyze
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from paths import ARTIFACTS, WORK  # noqa: E402
OUT = os.environ.get("JLENS_OUT", os.path.join(WORK, "jlens"))   # replication runs write elsewhere
SHIFT = int(os.environ.get("JLENS_SEED_SHIFT", "0"))      # fresh seeds for a replication
CONTROL = os.path.join(ARTIFACTS, "tinystories33m")
NCM_DIR = os.path.join(ARTIFACTS, "tinystories")
DEV = "cuda"


# ----------------------------------------------------------------------------------------------- control model
class Control:
    name = "control"

    def __init__(self):
        from transformers import AutoTokenizer, GPTNeoForCausalLM
        self.tok = AutoTokenizer.from_pretrained(CONTROL)
        self.m = GPTNeoForCausalLM.from_pretrained(CONTROL, dtype=torch.float32).to(DEV).eval()
        self.W = self.m.lm_head.weight.detach()            # V x d
        self.n_points = len(self.m.transformer.h) + 1      # input to each block + input to ln_f
        self.eos = self.tok.eos_token_id

    def sample(self, n, seed):
        torch.manual_seed(seed)
        out = self.m.generate(torch.full((n, 1), self.eos, device=DEV), do_sample=True, top_k=0, top_p=1.0,
                              temperature=1.0, max_new_tokens=128, pad_token_id=self.eos)
        return out[:, 1:129]

    def mask(self, ids):
        """1 for positions up to and including the first EOS of each story."""
        is_eos = (ids == self.eos).int()
        after = (is_eos.cumsum(1) - is_eos) > 0
        return (~after).float()

    def points(self, ids):
        """Forward with capture -> (residuals [list of B,T,d], H [B,T,d], position mask [B,T])."""
        res, H = [], []

        def pre(mod, args, kwargs):
            res.append(args[0] if args else kwargs["hidden_states"])
        hooks = [b.register_forward_pre_hook(pre, with_kwargs=True) for b in self.m.transformer.h]
        hooks.append(self.m.transformer.ln_f.register_forward_pre_hook(lambda mod, args: res.append(args[0])))
        hooks.append(self.m.transformer.ln_f.register_forward_hook(lambda mod, args, out: H.append(out)))
        try:
            self.m(ids)
        finally:
            for h in hooks:
                h.remove()
        return res, H[0], self.mask(ids)

    def text(self, ids):
        return [self.tok.decode(r, skip_special_tokens=True) for r in ids.tolist()]

    def word(self, i):
        return self.tok.decode([i])


# ----------------------------------------------------------------------------------------------- NCM
class NCM:
    name = "ncm"

    def __init__(self):
        from drafts import load
        self.cfg, self.tk, self.net, self.l0_p = load(NCM_DIR, os.path.join(NCM_DIR, "config.yaml"))
        self.vq = self.tk.vqvae.vq
        self.dec = self.tk.vqvae.decoder
        self.W = self.dec.output_layer.weight.detach()     # 8192 x 384
        self.n_points = len(self.net.transformer.blocks) + 1
        self.eos = self.tk.pre_tokenizer.eos_token_id
        self.levels = list(self.vq.levels)
        self.P = self.net.prefix_len

    @torch.no_grad()
    def sample(self, n, seed):
        torch.manual_seed(seed)
        l0 = torch.multinomial(self.l0_p, n * self.levels[0], replacement=True).view(n, self.levels[0]).to(DEV)
        return self.net.generate(l0, torch.full((n, 1), self.eos, device=DEV), temperature=1.0, top_k=0, top_p=1.0)

    def contrib(self, k, emb):
        """Level k's contribution to the full-length reconstruction (differentiable)."""
        z = self.vq._up_to_full(emb, k, self.levels[-1]) if k != len(self.levels) - 1 else emb
        return self.vq._apply_quant_resi(z, k)

    def H_of(self, z_full):
        d = self.dec
        z = z_full
        if d.up is not None:
            z = d.up(z.transpose(1, 2)).transpose(1, 2)
        if d.wpe is not None:
            z = z + d.wpe(torch.arange(z.shape[1], device=z.device))
        x = d.middle_block(z)
        for b in d.extra_blocks:
            x = b(x)
        return d.ln_f(x)

    def level_points(self, idx, j):
        """Generator level j (= codebook level j+1): residuals at its code positions, and H of the final text."""
        net, vq, P = self.net, self.vq, self.P
        B = idx.shape[0]
        offs = [0]
        for n in self.levels:
            offs.append(offs[-1] + n)
        with torch.no_grad():
            z_hat = torch.zeros(B, self.levels[-1], vq.embed_dim, device=DEV)
            for k in range(j + 1):
                z_hat = z_hat + self.contrib(k, vq.embeddings[k](idx[:, offs[k]:offs[k + 1]]))
            x_in = F.interpolate(z_hat.transpose(-1, -2), size=(self.levels[j + 1],), mode="area").transpose(-1, -2)
            z_all = torch.zeros(B, self.levels[-1], vq.embed_dim, device=DEV)
            for k in range(len(self.levels)):
                if k != j + 1:
                    z_all = z_all + self.contrib(k, vq.embeddings[k](idx[:, offs[k]:offs[k + 1]]))
            hard_emb = vq.embeddings[j + 1](idx[:, offs[j + 1]:offs[j + 2]])
            prefix = torch.full((B, 1), self.eos, device=DEV)
            z_prefix = self.tk.vqvae.encoder.wte(prefix)[:, :P, :]
        prefix_emb = net.transformer.wte(z_prefix)
        level_pos = net.transformer.wle(net.lvl_1L) if net.config.use_level_encoding else 0
        acc = sum(self.levels[1:j + 1])
        x = net.transformer.wte(x_in) + level_pos[:, acc:acc + self.levels[j + 1], :]
        res = []
        hooks = [b.register_forward_pre_hook(lambda mod, args, kwargs: res.append(args[0]), with_kwargs=True)
                 for b in net.transformer.blocks]
        hooks.append(net.transformer.ln_f.register_forward_pre_hook(lambda mod, args: res.append(args[0])))
        try:
            logits = net._decode_level_logits(x, prefix_emb, j, None, hidx=j)       # B x L x 4096
        finally:
            for h in hooks:
                h.remove()
        soft = F.softmax(logits, dim=-1) @ vq.embeddings[j + 1].weight
        emb = hard_emb + (soft - soft.detach())                                    # straight-through
        H = self.H_of(z_all + self.contrib(j + 1, emb))
        # differentiate w.r.t. the full residuals (slices made after the forward are not on the graph);
        # rmask drops the prefix position from every average
        rmask = torch.ones(B, res[0].shape[1], device=DEV)
        rmask[:, :P] = 0
        return res, H, torch.ones(B, self.levels[-1], device=DEV), rmask

    def text(self, idx):
        from drafts import level_snapshots, text
        full, _ = level_snapshots(self.vq, idx)
        with torch.no_grad():
            return text(self.tk, self.dec(full[-1]), self.eos)

    def word(self, i):
        return self.tk.pre_tokenizer.decode([i])


# ----------------------------------------------------------------------------------------------- fitting
def accumulate(M, mu, cnt, res, H, hmask, chunk, rmask=None):
    """Add sum over positions of d(sum_t H_t,k)/d res_l for every output dim k, plus residual means.
    hmask: which output (text) positions count; rmask: which residual positions count (default: all)."""
    d_out = H.shape[-1]
    w = hmask.unsqueeze(-1)                                   # only real text positions count as outputs
    Hm = H * w
    rm = torch.ones(res[0].shape[:2], device=DEV) if rmask is None else rmask
    rw = rm.unsqueeze(-1)
    for k0 in range(0, d_out, chunk):
        ks = list(range(k0, min(k0 + chunk, d_out)))
        cot = torch.zeros(len(ks), *H.shape, device=DEV)
        for i, k in enumerate(ks):
            cot[i, ..., k] = 1.0
        g = torch.autograd.grad(Hm, res, grad_outputs=cot, is_grads_batched=True, retain_graph=True)
        for l, gl in enumerate(g):
            M[l][ks] += (gl * rw).sum(dim=(1, 2)).detach()
    for l, r in enumerate(res):
        mu[l] += (r.detach() * rw).sum(dim=(0, 1))
    return cnt + rm.sum().item()


def fit(model, batches, bs, chunk, seed0=100):
    seed0 += SHIFT
    os.makedirs(OUT, exist_ok=True)
    d_res = 768
    if model.name == "control":
        d_out = model.W.shape[1]
        M = [torch.zeros(d_out, d_res, device=DEV) for _ in range(model.n_points)]
        mu = [torch.zeros(d_res, device=DEV) for _ in range(model.n_points)]
        cnt = 0
        for b in range(batches):
            ids = model.sample(bs, seed0 + b)
            res, H, m = model.points(ids)
            cnt = accumulate(M, mu, cnt, res, H, m, chunk, rmask=m)
            print(f"  batch {b + 1}/{batches}  positions so far {cnt}", flush=True)
        M = torch.stack([x / cnt for x in M]).cpu()
        mu = torch.stack([x / cnt for x in mu]).cpu()
    else:
        d_out = model.W.shape[1]
        nlev = len(model.levels) - 1
        M = torch.zeros(nlev, model.n_points, d_out, d_res)
        mu = torch.zeros(nlev, model.n_points, d_res)
        cnts = [0] * nlev
        for b in range(batches):
            idx = model.sample(bs, seed0 + b)
            for j in range(nlev):
                Mj = [torch.zeros(d_out, d_res, device=DEV) for _ in range(model.n_points)]
                muj = [torch.zeros(d_res, device=DEV) for _ in range(model.n_points)]
                res, H, m, rm = model.level_points(idx, j)
                c = accumulate(Mj, muj, 0, res, H, m, chunk, rmask=rm)
                M[j] += torch.stack(Mj).cpu()
                mu[j] += torch.stack(muj).cpu()
                cnts[j] += c
                del res, H
            print(f"  batch {b + 1}/{batches}  done (11 levels)", flush=True)
        for j in range(nlev):
            M[j] /= cnts[j]
            mu[j] /= cnts[j]
    torch.save({"M": M, "mu": mu}, f"{OUT}/{model.name}_M.pt")
    print("saved", f"{OUT}/{model.name}_M.pt", tuple(M.shape))


# ----------------------------------------------------------------------------------------------- analysis
def nnls(A, h, iters=150):
    """Batched non-negative least squares by projected gradient: min ||h - c A||, c >= 0.  A: N x k x d, h: N x d."""
    G = A @ A.transpose(1, 2)
    b = (A @ h.unsqueeze(-1)).squeeze(-1)
    L = G.abs().sum(-1).max(-1).values.clamp(min=1e-9).unsqueeze(-1)    # >= largest eigenvalue: safe step
    c = torch.zeros_like(b)
    for _ in range(iters):
        c = (c - (torch.einsum("nij,nj->ni", G, c) - b) / L).clamp(min=0)
    return c


def sparse_fit(h, D, K=25, cand_idx=None):
    """Greedy non-negative matching pursuit: explain h (N x d) with <=K rows of D (V x d, unit rows), refit by NNLS
    each step. cand_idx (N x C) restricts each row to its own candidate set (used for the random-words baseline).
    Returns R^2 per row, always in [0, 1]."""
    N = h.shape[0]
    r = h.clone()
    sel = torch.zeros(N, 0, dtype=torch.long, device=h.device)
    for step in range(K):
        if cand_idx is None:
            s = r @ D.t()
            if sel.numel():
                s.scatter_(1, sel, -float("inf"))
            pick = s.argmax(1, keepdim=True)
        else:
            s = torch.einsum("nd,ncd->nc", r, D[cand_idx])
            if sel.numel():
                s = s.masked_fill((cand_idx.unsqueeze(-1) == sel.unsqueeze(1)).any(-1), -float("inf"))
            pick = cand_idx.gather(1, s.argmax(1, keepdim=True))
        sel = torch.cat([sel, pick], 1)
        A = D[sel]
        coef = nnls(A, h, iters=300 if step == K - 1 else 100)
        r = h - torch.einsum("nk,nkd->nd", coef, A)
    return (1 - (r ** 2).sum(1) / (h ** 2).sum(1).clamp(min=1e-9)).clamp(0, 1)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)


@torch.no_grad()
def occupancy(model, Ms, mus, res_list, K=25, n_rand=25, seed=0):
    """Per residual point: mean R^2 of sparse fits with J-lens dict vs logit-lens dict vs random-K words."""
    W = model.W
    out = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for l, (Ml, mul, R) in enumerate(zip(Ms, mus, res_list)):
        h = (R - mul.to(DEV))
        Dj = unit((W @ Ml.to(DEV)))                                  # V x d  J-lens directions
        Dlog = unit(W @ torch.eye(W.shape[1], 768, device=DEV)) if W.shape[1] == 768 else None
        r2_j = sparse_fit(h, Dj, K).mean().item()
        r2_log = sparse_fit(h, Dlog, K).mean().item() if Dlog is not None else float("nan")
        rand = torch.randint(0, Dj.shape[0], (h.shape[0], n_rand), generator=g).to(DEV)
        r2_rand = sparse_fit(h, Dj, K, cand_idx=rand).mean().item()
        # rotation null: same vectors, same lengths and spectrum, randomly rotated -> no alignment with any word
        Q, _ = torch.linalg.qr(torch.randn(h.shape[1], h.shape[1], generator=g).to(DEV))
        r2_rot = sparse_fit(h @ Q, Dj, K).mean().item()
        out.append({"point": l, "jlens": r2_j, "rotated_null": r2_rot, "logit_lens": r2_log, "random": r2_rand})
    return out


@torch.no_grad()
def readout(model, Ml, mul, h, top=6):
    s = (h - mul.to(DEV)) @ (model.W @ Ml.to(DEV)).t()
    return [[model.word(i) for i in row] for row in s.topk(top, dim=-1).indices.tolist()]


def analyze(args):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rep = {}
    # ---- control
    ctl = Control()
    d = torch.load(f"{OUT}/control_M.pt", weights_only=True)
    ids = ctl.sample(8, 999 + SHIFT)
    res, H, m = ctl.points(ids)
    with torch.no_grad():
        flat = [r.detach()[m.bool()] for r in res]
    sel = torch.randperm(flat[0].shape[0], generator=torch.Generator().manual_seed(1 + SHIFT))[:512]
    occ = occupancy(ctl, d["M"], d["mu"], [f[sel.to(DEV)] for f in flat])
    rep["control_occupancy"] = occ
    print("\nCONTROL (word-by-word) J-space occupancy, mean R^2 with K=25:")
    for o in occ:
        print(f"  point {o['point']} (after {o['point']} blocks): J-lens {o['jlens']:.3f} | rotated null {o['rotated_null']:.3f} | logit lens {o['logit_lens']:.3f} | 25 random words {o['random']:.3f}", flush=True)
    story = ids[0:1]
    toks = [ctl.word(t) for t in story[0].tolist()]
    print("\nCONTROL readout, story:", ctl.text(story)[0][:200].replace("\n", " "))
    res1, _, _ = ctl.points(story)
    rep["control_readout"] = []
    for pos in [p for p in range(4, min(len(toks), 60), 8)]:
        row = {"pos": pos, "token": toks[pos], "layers": []}
        for l in range(ctl.n_points):
            row["layers"].append(readout(ctl, d["M"][l], d["mu"][l], res1[l].detach()[0, pos:pos + 1])[0])
        rep["control_readout"].append(row)
        print(f"  at {pos:>3} {toks[pos]!r:>12}: " + " | ".join(f"L{l}:" + ",".join(w.strip() for w in ws[:4]) for l, ws in enumerate(row["layers"])))
    del ctl
    torch.cuda.empty_cache()

    # ---- ncm
    ncm = NCM()
    d = torch.load(f"{OUT}/ncm_M.pt", weights_only=True)
    idx = ncm.sample(8, 999 + SHIFT)
    rep["ncm_occupancy"] = {}
    print("\nNCM (coarse-to-fine) J-space occupancy per level, mean R^2 with K=25 (J-lens / rotated null / 25 random words):")
    for j in range(len(ncm.levels) - 1):
        res, H, m, _ = ncm.level_points(idx, j)
        flat = [r.detach()[:, ncm.P:].reshape(-1, r.shape[-1]) for r in res]
        n = flat[0].shape[0]
        sel = torch.randperm(n, generator=torch.Generator().manual_seed(1 + SHIFT))[:min(n, 256)].to(DEV)
        occ = occupancy(ncm, d["M"][j], d["mu"][j], [f[sel] for f in flat])
        rep["ncm_occupancy"][f"L{j + 2}"] = occ
        print(f"  level L{j + 2:<2} ({ncm.levels[j + 1]:>3} codes): " + "  ".join(f"p{o['point']}:{o['jlens']:.2f}/{o['rotated_null']:.2f}/{o['random']:.2f}" for o in occ[::2]), flush=True)
    story = idx[0:1]
    print("\nNCM readout, story:", ncm.text(story)[0][:200].replace("\n", " "))
    rep["ncm_readout"] = {}
    for j in [0, 1, 2, 4, 7, 10]:
        res, _, _, _ = ncm.level_points(story, j)
        rows = []
        for l in [0, 3, 6, 9, 12]:
            words = readout(ncm, d["M"][j][l], d["mu"][j][l], res[l].detach()[0, ncm.P:])
            rows.append({"point": l, "per_code": words})
        rep["ncm_readout"][f"L{j + 2}"] = rows
        print(f"\n  level L{j + 2} ({ncm.levels[j + 1]} codes) - top words per code position, by depth:")
        for row in rows:
            cells = [",".join(w.strip() for w in ws[:3]) for ws in row["per_code"][:8]]
            print(f"    p{row['point']:<2}: " + " | ".join(cells))
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
    print("\nwrote", f"{OUT}/report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "analyze"])
    ap.add_argument("--model", choices=["control", "ncm"])
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(Control() if a.model == "control" else NCM(), a.batches, a.bs, a.chunk)
    else:
        analyze(a)
