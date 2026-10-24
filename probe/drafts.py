"""Read an NCM model's drafts level by level (DJ + Claude, 2026-10-20).

NCM writes a story coarse-to-fine: level 1 is ONE code, then 8, 16, ... 128. This script generates stories and,
after each level, decodes what the model "has so far" two ways:
  short  - the draft squeezed to that level's length (level 2 = an 8-token version of the story)
  full   - the full-length 128-token reconstruction from levels 1..k (a blurry whole story that sharpens)
The tokenizer was trained with nested dropout of fine levels, so decoding a partial stack is in-distribution.

Note: quant.idx_to_zhats adds in place (z_hat.add_) and appends the same tensor every level, so every entry of its
z_hats list is the final reconstruction - this script builds its own per-level snapshots. (An earlier note here also
said decode_multiscales had no return; that was wrong - the file just ends without a trailing newline.)

  python probe/drafts.py --ckpt artifacts/tinystories --n 4 --seed 7
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ARTIFACTS  # noqa: E402


def load(ckpt_dir, config_path):
    cfg = OmegaConf.load(config_path)
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ckpt_dir, cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
    sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=True)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    l0_p = None
    spec = cfg.generation.get("level0_unigram")
    if spec:
        path, ref = str(spec).split(":", 1)
        obj = sd if path == cfg.checkpoints.generator else torch.load(os.path.join(ckpt_dir, path), map_location="cpu", weights_only=True)
        for k in ref.split("."):
            obj = obj[k]
        l0_p = obj.float()
    return cfg, tk, net, l0_p


@torch.no_grad()
def level_snapshots(vq, idx):
    """Cumulative full-length reconstruction after each level, as separate tensors (clone, not in-place aliases)."""
    z_hat = torch.zeros(idx.shape[0], vq.levels[-1], vq.embed_dim, device=idx.device)
    full, short, acc = [], [], 0
    for k, n in enumerate(vq.levels):
        emb = vq.embeddings[k](idx[:, acc:acc + n])
        z = vq._up_to_full(emb, k, vq.levels[-1]) if k != len(vq.levels) - 1 else emb
        z_hat = z_hat + vq._apply_quant_resi(z, k)
        full.append(z_hat.clone())
        short.append(F.interpolate(z_hat.transpose(-1, -2), size=(n,), mode="area").transpose(-1, -2)
                     if k != len(vq.levels) - 1 else z_hat.clone())
        acc += n
    return full, short


def text(tk, logits, eos):
    out = []
    for ids in logits.argmax(-1).tolist():
        if eos in ids:
            ids = ids[:ids.index(eos)]
        out.append(tk.pre_tokenizer.decode([t for t in ids if t != tk.pre_tokenizer.pad_token_id], skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ARTIFACTS, "tinystories"))
    ap.add_argument("--config", default=None, help="defaults to <ckpt>/config.yaml (shipped with the weights)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temp", type=float, default=None)
    a = ap.parse_args()

    cfg, tk, net, l0_p = load(a.ckpt, a.config or os.path.join(a.ckpt, "config.yaml"))
    vq = tk.vqvae.vq
    eos = tk.pre_tokenizer.eos_token_id
    print("levels:", list(vq.levels))
    torch.manual_seed(a.seed)
    l0 = torch.multinomial(l0_p, a.n * vq.levels[0], replacement=True).view(a.n, vq.levels[0]).cuda() if l0_p is not None else None
    G = cfg.generation
    temp = a.temp if a.temp is not None else float(G.get("temperature", 1.0))
    idx = net.generate(l0, torch.full((a.n, 1), eos, device="cuda"), temperature=temp,
                       top_k=int(G.get("top_k", 0)), top_p=float(G.get("top_p", 1.0)))
    full, short = level_snapshots(vq, idx)
    dec = tk.vqvae.decoder
    full_txt = [text(tk, dec(z), eos) for z in full]
    short_txt = [text(tk, dec(z), eos) for z in short]
    for s in range(a.n):
        print("\n" + "=" * 100 + f"\nSTORY {s + 1}  (level-1 code: {idx[s, 0].item()})")
        for k, n in enumerate(vq.levels):
            print(f"\n  L{k + 1:<2} ({n:>3} codes)  short: {short_txt[k][s][:160]!r}")
            print(f"                   full : {full_txt[k][s][:300]!r}")


if __name__ == "__main__":
    main()
