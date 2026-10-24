"""Images for the public thread, drawn from the results page's own data files (no number typed by hand).

Writes 16:9 PNGs (1600x900) and two animated GIFs to ../tweet_images/:
  1_jspace.png         does NCM have a J-space?
  2_drafts.gif         one story, written level by level (animation)
  3_menu.png           the Fin story: a menu of candidates, dropped ones pushed down
  4_cat_dog.png        cat-minus-dog push, by level edited
  5_swap.png           swapping early levels: Sudoku recovers, stories follow the swap
  5b_sudoku.gif        one hard puzzle, level by level (animation)
  6_scorecard.png      every prediction scored, both models
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results_page")
OUT = os.path.join(os.path.dirname(HERE), "tweet_images")
os.makedirs(OUT, exist_ok=True)
load = lambda n: json.load(open(os.path.join(SRC, n + ".json"), encoding="utf-8"))

INK, MUTED, GRID, BG = "#1d1f23", "#6b7079", "#e6e4de", "#fbfaf7"
BLUE, ORANGE, GREEN, RED, GREY = "#2f6fde", "#d9822b", "#2e9d6b", "#c2412d", "#b4b8c1"
FOOT = "Inside a Next-Concept Model  ·  DJ Oetken with Claude (Anthropic); blind predictions by Gemini (Google)  ·  NCM: Stanford Hazy Research + Scaling Intelligence"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG})


def frame(title, sub):
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.text(0.05, 0.93, title, fontsize=31, color=INK, weight="bold", va="top")
    fig.text(0.05, 0.855, sub, fontsize=17, color=MUTED, va="top")
    fig.text(0.05, 0.03, FOOT, fontsize=11.5, color=MUTED)
    return fig


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=100)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. J-space
al = load("alignment")
ncm = [r for r in al if r["model"] == "ncm" and r["depth"] in (0, 12)]
lv = lambda r: int(r["level"][1:])
content = [r["ratio"] for r in ncm if 3 <= lv(r) <= 6]
fine = [r["ratio"] for r in ncm if 7 <= lv(r) <= 12]
ctl = max(r["ratio"] for r in al if r["model"] == "control")
bars = [("NCM, content levels\n(levels 3–6)", sum(content) / len(content), BLUE),
        ("Ordinary word-by-word model\n(same stories, best layer)", ctl, GREY)]
fig = frame("Does NCM have a J-space? Yes, at its content levels.",
            "How much more hidden states line up with word directions than randomly rotated copies of themselves (1× = chance).\n"
            "Jacobian lens, first and last generator layers; replicated on fresh seeds.")
ax = fig.add_axes([0.07, 0.13, 0.88, 0.62])
xs = range(len(bars))
ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.56)
for x, (lab, v, _) in zip(xs, bars):
    ax.text(x, v + 0.08, f"{v:.1f}×", ha="center", va="bottom", fontsize=30, color=INK, weight="bold")
ax.axhline(1, color=MUTED, ls="--", lw=1.5)
ax.text(0.5, 1.05, "chance", color=MUTED, fontsize=15, ha="center", va="bottom")
ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=17, color=INK)
ax.set_ylim(0, 4.6)
ax.set_yticks([0, 1, 2, 3, 4], ["0", "1×", "2×", "3×", "4×"])
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
save(fig, "1_jspace.png")

# ---------------------------------------------------------------- 2. drafts animation (superseded by make_animations.py)
FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
f_title = ImageFont.truetype(os.path.join(FONT, "DejaVuSans-Bold.ttf"), 40)
f_body = ImageFont.truetype(os.path.join(FONT, "DejaVuSerif.ttf"), 40)
f_small = ImageFont.truetype(os.path.join(FONT, "DejaVuSans.ttf"), 24)
f_foot = ImageFont.truetype(os.path.join(FONT, "DejaVuSans.ttf"), 16)
f_mono = ImageFont.truetype(os.path.join(FONT, "DejaVuSansMono-Bold.ttf"), 34)
W, H = 1600, 900
drafts = sorted(load("drafts"), key=lambda r: r["level_n"])
frames, durs = [], []
for r in drafts:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((80, 60), "NCM writes the whole story at once, then sharpens it", font=f_title, fill=INK)
    d.text((80, 125), f"Draft after level {r['level_n']} of 12  ({r['codes']} codes)", font=f_small, fill=BLUE)
    # progress bar
    d.rectangle((80, 170, W - 80, 178), fill=GRID)
    d.rectangle((80, 170, 80 + (W - 160) * r["level_n"] / 12, 178), fill=BLUE)
    text = " ".join(r["full"].split())[:430] + "…"
    y = 220
    for line in textwrap.wrap(text, 52)[:9]:
        d.text((80, y), line, font=f_body, fill=INK)
        y += 62
    d.text((80, H - 45), FOOT, font=f_foot, fill=MUTED)
    frames.append(im)
    durs.append(3500 if r["level_n"] == 12 else 1300)
frames[0].save(os.path.join(OUT, "2_drafts_plain.gif"), save_all=True, append_images=frames[1:], duration=durs, loop=0)
print("wrote 2_drafts_plain.gif (the highlighted version is made by make_animations.py)")

# ---------------------------------------------------------------- 3. menu (Fin story)
fc = load("fin_case")
fig = frame("Its early levels hold a menu, not a sentence.",
            "The fish story: strongest relative logit for each candidate word, level by level. Solid = in the final story.\n"
            "Fin beats Finny and Bubbles for the name; the dropped animals (shark, swan, dolphin) fall as crab and duck lock in.")
ax = fig.add_axes([0.07, 0.13, 0.78, 0.62])
words = list(dict.fromkeys(r["word"] for r in fc))
pal = iter([BLUE, ORANGE, GREEN, "#8a5cd6", "#d1495b", "#1f9bb3"])
ends = []
for w in words:
    pts = sorted([r for r in fc if r["word"] == w], key=lambda r: r["level_n"])
    kept = pts[0]["in_final"]
    c = next(pal) if kept else GREY
    ax.plot([p["level_n"] for p in pts], [p["value"] for p in pts], color=c, lw=3 if kept else 2, ls="-" if kept else "--",
            marker="o", ms=5)
    ends.append([pts[-1]["value"], w.strip(), c if kept else MUTED, "bold" if kept else "normal"])
ends.sort(key=lambda e: e[0])
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 1.15)
for y, w, c, wt in ends:
    ax.text(12.15, y, w, color=c, fontsize=15, va="center", weight=wt)
ax.set_xticks(range(3, 13), [f"L{i}" for i in range(3, 13)])
ax.set_xlabel("level")
ax.set_ylabel("relative logit")
ax.grid(color=GRID)
ax.set_axisbelow(True)
save(fig, "3_menu.png")

# ---------------------------------------------------------------- 4. cat / dog
sw = [r for r in load("swap") if r["model"] == "ncm"]
base = next(r for r in sw if r["place"] == "baseline")
s2 = sorted([r for r in sw if r["place"] != "baseline" and r["strength"] == 2], key=lambda r: r["place_n"])
ex = next(r for r in load("swap_examples") if r["id"] == "ncm-L5")
fig = frame("You can steer it: push \"cat minus dog\".",
            "Stories that mention a cat after a push at one level (strength 2). Grey: a random push of the same size. 64 stories each.")
ax = fig.add_axes([0.07, 0.15, 0.45, 0.6])
xs = range(len(s2))
ax.bar([x - 0.2 for x in xs], [r["cat_swap"] for r in s2], 0.4, color=BLUE, label="cat − dog push")
ax.bar([x + 0.2 for x in xs], [r["cat_random"] for r in s2], 0.4, color=GREY, label="random push")
for x, r in zip(xs, s2):
    ax.text(x - 0.2, r["cat_swap"] + 0.02, f"{r['cat_swap']:.0%}", ha="center", fontsize=15, color=INK)
ax.axhline(base["cat_swap"], color=MUTED, ls="--", lw=1.5, label=f"no edit ({base['cat_swap']:.0%})")
ax.set_xticks(list(xs), [f"level {r['place'][1:]}" for r in s2])
ax.set_ylim(0, 1.12)
ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.legend(frameon=False, loc="upper right", fontsize=14)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
clip = lambda t: " ".join(t.split())[:150] + "…"
fig.text(0.57, 0.72, "Edit at level 5 (strength 4), same random seed:", fontsize=17, color=INK, weight="bold")
fig.text(0.57, 0.66, "Before", fontsize=14, color=MUTED)
fig.text(0.57, 0.645, "\n".join(textwrap.wrap(clip(ex["before"]), 46)), fontsize=16, color=INK, va="top", family="serif")
fig.text(0.57, 0.40, "After", fontsize=14, color=BLUE)
fig.text(0.57, 0.385, "\n".join(textwrap.wrap(clip(ex["after"]), 46)), fontsize=16, color=INK, va="top", family="serif")
save(fig, "4_cat_dog.png")

# ---------------------------------------------------------------- 5. swap: stories vs Sudoku
st = sorted([r for r in load("stories_swap") if r["seed"] == 21], key=lambda r: r["k"])
sd = sorted([r for r in load("sudoku_swap") if r["task"] == "hard" and r["cond"] == "donor"], key=lambda r: r["k"])
fig = frame("Swap in early levels: Sudoku recovers, stories don't.",
            "How much of the donor ends up in the final output. Stories have no prompt, so the early codes become the story;\n"
            "Sudoku sees its puzzle at every level and is trained to fix early mistakes, so it recovers.")
ax = fig.add_axes([0.07, 0.13, 0.80, 0.62])
ax.plot([r["k"] for r in st], [r["leak"] for r in st], color=BLUE, lw=4, marker="o", ms=8)
ax.plot([r["k"] for r in sd], [r["leak"] for r in sd], color=GREEN, lw=4, marker="o", ms=8)
ax.text(st[-1]["k"] + 0.2, st[-1]["leak"], "stories\n(donor's words)", color=BLUE, fontsize=16, va="center", weight="bold")
ax.text(sd[-1]["k"] + 0.2, sd[-1]["leak"], "Sudoku\n(donor's digits)", color=GREEN, fontsize=16, va="center", weight="bold")
k6 = next(r for r in st if r["k"] == 6)
s6 = next(r for r in sd if r["k"] == 6)
ax.annotate(f"{k6['leak']:.0%}", (6, k6["leak"]), xytext=(6, k6["leak"] + 0.1), ha="center", fontsize=20, color=BLUE, weight="bold")
ax.annotate(f"{s6['leak']:.1%}", (6, s6["leak"]), xytext=(6.25, s6["leak"] + 0.035), ha="left", fontsize=20, color=GREEN, weight="bold")
ax.axhline(st[0]["chance"], color=MUTED, ls="--", lw=1.2)
ax.text(1.0, st[0]["chance"] + 0.015, "chance (stories)", color=MUTED, fontsize=13)
ax.set_xlim(0.6, 12.4)
ax.set_xticks(range(1, 12), [f"L1–{k}" if k > 1 else "L1" for k in range(1, 12)], fontsize=13)
ax.set_xlabel("levels swapped")
ax.set_ylim(0, 1.05)
ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.grid(color=GRID)
ax.set_axisbelow(True)
save(fig, "5_swap.png")

# ---------------------------------------------------------------- 5b. Sudoku animation
grid = sorted(load("sudoku_grid"), key=lambda r: r["level_n"])
frames, durs = [], []
cell = 70
for r in grid:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((80, 60), "Sudoku: the whole grid sharpens at once", font=f_title, fill=INK)
    sub = "The puzzle (clues in bold)" if r["level_n"] == 0 else f"Decoded after level {r['level_n']} of 12  ·  {round(r['acc'] * 100)}% of cells right  ·  red = not right yet"
    d.text((80, 125), sub, font=f_small, fill=BLUE)
    x0, y0 = 80, 190
    for i, ch in enumerate(r["digits"]):
        row, col = divmod(i, 9)
        x, y = x0 + col * cell, y0 + row * cell
        bad = r["right"][i] == "0"
        d.rectangle((x, y, x + cell, y + cell), outline=GRID, fill="#f6dcd6" if bad else BG)
        t = "·" if ch == "_" else ch
        col_ = RED if bad else (INK if r["clue"][i] == "1" else "#3d4350")
        d.text((x + cell / 2, y + cell / 2), t, font=f_mono, fill=col_ if ch != "_" else GREY, anchor="mm")
    for k in range(0, 10, 3):
        d.line((x0, y0 + k * cell, x0 + 9 * cell, y0 + k * cell), fill=INK, width=3)
        d.line((x0 + k * cell, y0, x0 + k * cell, y0 + 9 * cell), fill=INK, width=3)
    note = ["Hard puzzle. NCM writes all 81 cells at once over 12 levels;",
            "the decoder never sees the puzzle, so even the clues",
            "have to come through the codes. The middle drafts",
            "don't look like pencil marks: the rules show up",
            "at the last level."]
    for j, line in enumerate(note):
        d.text((800, 260 + j * 44), line, font=f_small, fill=MUTED)
    d.text((80, H - 45), FOOT, font=f_foot, fill=MUTED)
    frames.append(im)
    durs.append(3500 if r["level_n"] in (0, 12) else 1300)
frames[0].save(os.path.join(OUT, "5b_sudoku.gif"), save_all=True, append_images=frames[1:], duration=durs, loop=0)
print("wrote 5b_sudoku.gif")

# ---------------------------------------------------------------- 6. scorecard
rows = load("scorecard") + load("sudoku_score") + load("sudoku_swap_score") + load("stories_swap_score")
cats = [("supported", "right", GREEN), ("partly", "partly", ORANGE), ("not supported", "wrong", RED), ("uninformative", "uninformative", GREY)]
fig = frame("Every prediction written down first, then scored.",
            "Claude and Gemini, across the J-space, editing, erasure, Sudoku and swap tests. Misses stay on the page.\n"
            "(Claude made more predictions: it also predicted the Sudoku tests alone.)")
ax = fig.add_axes([0.16, 0.2, 0.66, 0.5])
for yi, who in enumerate(["Gemini", "Claude"]):
    left = 0
    mine = [r for r in rows if r["who"] == who]
    for key, lab, c in cats:
        n = sum(r["verdict"] == key for r in mine)
        if n:
            ax.barh(yi, n, left=left, color=c, height=0.55)
            if key == "uninformative":
                ax.text(left + n + 0.3, yi, f"{n} {lab}", ha="left", va="center", color=MUTED, fontsize=15)
            else:
                ax.text(left + n / 2, yi, f"{n} {lab}", ha="center", va="center", color="white", fontsize=17, weight="bold")
            left += n
ax.set_yticks([0, 1], ["Gemini", "Claude"], fontsize=22, color=INK)
ax.set_xlabel("predictions")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
save(fig, "6_scorecard.png")
