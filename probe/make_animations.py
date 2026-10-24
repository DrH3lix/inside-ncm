"""Thread animations from real model output, with changed words highlighted. Writes MP4 (for X) and GIF copies.

  2_drafts.{mp4,gif}    the fish story, level by level; words that changed since the previous level highlighted
  4b_cat_swap.{mp4,gif} the Lucy/Max story, no edit vs a cat-minus-dog push at level 5 (same seed), side by side;
                        words that differ between the two highlighted
Inputs: results_page/drafts.json, <NCM_PROBE_OUT>/anim/cat_swap_drafts.json (probe/anim_swap_data.py).
Needs ffmpeg on PATH for the MP4s.
"""
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import WORK  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tweet_images")
os.makedirs(OUT, exist_ok=True)
FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
F = lambda name, size: ImageFont.truetype(os.path.join(FONT, name), size)
f_title, f_sub, f_lbl, f_foot = F("DejaVuSans-Bold.ttf", 40), F("DejaVuSans.ttf", 25), F("DejaVuSans-Bold.ttf", 26), F("DejaVuSans.ttf", 16)
INK, MUTED, GRID, BG, BLUE, RED = "#1d1f23", "#6b7079", "#e6e4de", "#fbfaf7", "#2f6fde", "#c2412d"
HI_NEW, HI_OLD = "#d6e4ff", "#f6dcd6"
W, H = 1600, 900
FOOT = "Inside a Next-Concept Model  ·  DJ Oetken with Claude (Anthropic); blind predictions by Gemini (Google)  ·  NCM: Stanford Hazy Research + Scaling Intelligence"
FPS = 10


def words(t):
    return " ".join(t.split()).split(" ")


def marks(a, b):
    """Token flags for a and b: True where the token is part of a difference between them."""
    fa, fb = [False] * len(a), [False] * len(b)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            for i in range(i1, i2):
                fa[i] = True
            for j in range(j1, j2):
                fb[j] = True
    return fa, fb


def flow(d, toks, flags, x, y, maxw, font, hi, max_lines, line_h):
    """Draw words left to right with wrapping; flagged words get a highlight box. Stops (with …) at max_lines."""
    space = font.getlength(" ")
    cx, line = x, 0
    for t, f in zip(toks, flags):
        w = font.getlength(t)
        if cx + w > x + maxw and cx > x:
            line += 1
            cx = x
            if line >= max_lines:
                d.text((cx, y + line * line_h - line_h), "", font=font)
                break
        yy = y + line * line_h
        if f:
            d.rounded_rectangle((cx - 4, yy + 4, cx + w + 4, yy + line_h - 6), radius=6, fill=hi)
        d.text((cx, yy), t, font=font, fill=INK)
        cx += w + space


def header(d, title, sub):
    d.text((80, 50), title, font=f_title, fill=INK)
    d.text((80, 112), sub, font=f_sub, fill=MUTED)
    d.text((80, H - 45), FOOT, font=f_foot, fill=MUTED)


def bar(d, k, y, push=None):
    x0, x1 = 80, W - 80
    d.rectangle((x0, y, x1, y + 8), fill=GRID)
    d.rectangle((x0, y, x0 + (x1 - x0) * k / 12, y + 8), fill=BLUE)
    if push:
        px = x0 + (x1 - x0) * push / 12
        d.rectangle((px - 2, y - 10, px + 2, y + 18), fill=RED)
        d.text((px + 8, y + 12), f"push at level {push}", font=f_foot, fill=RED)


def write(frames, durs, name):
    tmp = tempfile.mkdtemp(dir=WORK)
    try:
        n = 0
        for im, ms in zip(frames, durs):
            for _ in range(max(1, round(ms / 1000 * FPS))):
                im.save(os.path.join(tmp, f"f{n:05d}.png"))
                n += 1
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                        os.path.join(OUT, name + ".mp4")], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    frames[0].save(os.path.join(OUT, name + ".gif"), save_all=True, append_images=frames[1:], duration=durs, loop=0)
    print("wrote", name + ".mp4 / .gif")


# ---------------------------------------------------------------- fish story, changes highlighted
drafts = sorted(json.load(open(os.path.join(HERE, "results_page", "drafts.json"), encoding="utf-8")), key=lambda r: r["level_n"])
f_body = F("DejaVuSerif.ttf", 38)
frames, durs, prev = [], [], None
for r in drafts:
    toks = words(r["full"])
    flags = marks(prev, toks)[1] if prev is not None else [False] * len(toks)
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    header(d, "NCM writes the whole story at once, then sharpens it",
           f"Draft after level {r['level_n']} of 12  ({r['codes']} codes).  Highlighted: words that changed since the last level.")
    bar(d, r["level_n"], 160)
    flow(d, toks, flags, 80, 205, W - 160, f_body, HI_NEW, 9, 60)
    frames.append(im)
    durs.append(4000 if r["level_n"] == 12 else 1500)
    prev = toks
write(frames, durs, "2_drafts")

# ---------------------------------------------------------------- cat swap, side by side
cs = json.load(open(os.path.join(WORK, "anim", "cat_swap_drafts.json"), encoding="utf-8"))
f_col = F("DejaVuSerif.ttf", 42)
colw, lx, rx = 690, 80, 830
NW = 24                                                   # opening words only, so the key changes stand out
frames, durs = [], []
for k, (a, b) in enumerate(zip(cs["base"], cs["pushed"]), 1):
    ta, tb = words(a), words(b)
    fa, fb = marks(ta, tb)                                # diff the whole drafts, then show the opening only
    ta, tb, fa, fb = ta[:NW], tb[:NW], fa[:NW], fb[:NW]
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    same = "identical so far" if not any(fb) else "highlighted: where the two drafts differ (opening lines shown)"
    header(d, "Same story, same random seed. One push at level 5.",
           f"Draft after level {k} of 12  ·  {same}")
    bar(d, k, 160, push=cs["level"])
    d.text((lx, 205), "No edit", font=f_lbl, fill=MUTED)
    d.text((rx, 205), "Pushed along \"cat − dog\" at level 5", font=f_lbl, fill=BLUE)
    d.line((rx - 30, 205, rx - 30, H - 80), fill=GRID, width=2)
    flow(d, ta + ["…"], fa + [False], lx, 255, colw, f_col, HI_OLD, 8, 66)
    flow(d, tb + ["…"], fb + [False], rx, 255, colw, f_col, HI_NEW, 8, 66)
    frames.append(im)
    durs.append(4500 if k == 12 else 2200 if k in (4, 5) else 1400)
write(frames, durs, "4b_cat_swap")
