# J-lens on NCM vs a word-by-word control — predictions written BEFORE the data

Written 2026-10-20 ~01:00 by Claude, before running the full fits, before reading Argon's reply. Scored afterward,
whichever way it goes.

Setup: Jacobian lens (average effect of each layer's residual on the eventual output text, mapped into words), same
definition for both models, both trained on TinyStories. Control = TinyStories-33M (GPT-Neo, 4 layers). NCM =
Hazy Research's 12-level coarse-to-fine generator (12 layers) + VQ-VAE decoder. "J-space occupancy" = how much of a
residual a sparse (K=25) non-negative mix of lens directions explains, vs 25 random words and vs the plain unembedding.

## Predictions

1. **NCM has a J-space-like structure (J-lens beats random words clearly at every level).** Confidence: high.
   Anything that ends in text through a linear readout will have word-aligned directions; the question is where.

2. **In NCM, LEVEL matters more than LAYER.** Coarse levels (L2–L4) will read out content words (cast, setting,
   key objects: rat, forest, fish, acorn), because one coarse code shapes many text positions at once. Fine levels
   (L9–L12) will read out function words and punctuation (the, was, ".", ","). Confidence: medium-high.

3. **In the control, LAYER is what matters.** Early layers read out the current token / local syntax; the last
   layers read out next-token guesses. Concept words appear mostly at their own positions. Confidence: medium-high.

4. **At coarse levels, NCM's J-space and its written sketch will mostly agree** (the J-lens at L3 shows the same
   nouns the L3 draft decodes to) — "inside-out": little gap between what it holds and what it writes. In the
   control, the J-lens will hold things it hasn't written yet (planning ahead), so the gap is bigger.
   Confidence: medium. This is the interesting one.

5. **Occupancy is higher at NCM's coarse levels than anywhere in the control** — a coarse code position is close to
   a small bag of concepts by construction. Fine levels drop back toward the control's numbers. Confidence:
   low-medium (risky; the code space is 384-d and the generator is 768-d, so it could easily go the other way).

6. **A concept swap at a coarse level changes the whole story** (swap fish→rat at L3 and the animal changes
   everywhere downstream); the same swap in the control at one position mostly changes the local continuation.
   Confidence: medium. (Causal test, run last.)

## Known limits of this lens
- NCM path is the DIRECT effect of a level's codes on the text; later levels are held fixed, so effects that travel
  through later levels' generation are not counted. Likely under-reads coarse levels.
- Readout is through each model's own vocabulary (control: GPT-2 BPE 50k; NCM: 8k BPE), so words won't line up 1:1.
- Small sample (64 stories per model). Directional evidence, not a paper.

---
## RESULTS (2026-10-20 ~01:45) — self-scored before reading Argon's prediction

Fits: control 8,108 positions; NCM 64 stories x 11 levels. Occupancy = mean R^2 of a 25-direction non-negative fit,
compared with a ROTATED NULL (same vectors randomly rotated: same lengths/spectrum, no alignment with any word).
Raw R^2 is not comparable across models (50k-word vs 8k-word dictionaries); J-lens / null ratio is.

| model / place | J-lens R^2 | rotated null | ratio |
|---|---|---|---|
| control, input (L0) | 0.24 | 0.19 | 1.2x |
| control, mid (L1-L3) | 0.26-0.28 | 0.22-0.24 | 1.2x |
| control, last layer | 0.41 | 0.28 | 1.5x |
| NCM L2 (8 codes), all depths | 0.30-0.41 | 0.03-0.05 | 8-10x (but readout = junk subwords; random words also fit 0.13-0.29 -> low-rank, not word-like) |
| NCM L3-L6, input | 0.28-0.33 | 0.07-0.08 | ~4x |
| NCM L3-L6, middle layers | 0.10-0.19 | 0.07-0.09 | 1.3-2x |
| NCM L3-L6, last layer | 0.23-0.26 | 0.06-0.08 | ~3.5x |
| NCM L7-L12 | 0.05-0.20 | 0.04-0.07 | 1.3-3x (last layer best) |

Readouts: control L0 = current word's neighbourhood (story -> tale, Stories); last layer = next / future words
("make" -> noises; opening quote -> bang, which appears several words later; "playing" -> cops, pirates, knights).
NCM L3 for a fish story: Fin, Finny, Bubbles | duck, swan, crocodile | Ouch, ow | reef, sea, shore | dolphin, shark,
crab | One, day, Hi. L6: reef, jellyfish, crab, Bubbles, pond, ducks. L9: punctuation and glue (., ",", was, were,
His, He, Something). L2: junk fragments (pted, yach, elevator). L12: noisy (names, animals, fragments).

Scores:
1. NCM has a J-space-like structure — SUPPORTED. Beats the null at every level, strongly at coarse levels.
2. Level matters more than layer; coarse = content, fine = glue — PARTLY. L3-L6 content yes, L9 glue yes, alignment
   falls with level yes. But L2 (coarsest) is NOT content-readable, L12 is noise not glue, and depth matters a lot
   too: a U-shape (strong at input + last layer, dip in the middle) at every level.
3. Control: early = current word, late = next/future words — SUPPORTED.
4. NCM coarse J-space ~ its written draft (inside-out, small gap) — NOT SUPPORTED as stated (not measured
   quantitatively; qualitatively the lens holds a MENU of candidates the story doesn't use: Fin/Finny/Bubbles,
   duck/swan/crocodile, dolphin/shark/crab). The control's planning-ahead half is supported.
5. NCM coarse more J-space-shaped than anywhere in the control — SUPPORTED on the null-normalised measure (3.5-4x
   vs <=1.5x), NOT on raw R^2 (control last layer 0.41 > NCM L3 0.30). Call it partly.
6. Concept swap — NOT RUN YET.

Surprise: in the control, the plain unembedding ("logit lens") explains MORE residual variance than the J-lens
(0.34-0.43 vs 0.24-0.41; 0.88 at the embedding layer because input/output embeddings are tied). The residual mostly
carries "what word is here"; the J-lens picks out the smaller part that moves the output — consistent with
Anthropic's claim that J-space is a small slice of the activation.

---
## ARGON's prediction (Google model; saw only DJ's question, nothing else) — scored against the same data

Extra measurement for Argon's "spectrum collapse": Jacobian Frobenius norm and effective rank (exp entropy of
squared singular values, of 384) per level, at depth p0 / p6 / p12:
  L2 norm 20.0, rank 1.1/3.4/2.6 | L3 9.4, 29/37/26 | L5 5.4, 63/63/67 | L7 0.59, 60/85/40 | L9 0.55, 6/19/7 |
  L12 0.064, 39/50/33.  Control: norms 545->66, eff. rank 163->760 (of 768) with depth.

- J-space exists in NCM: YES (same as mine).
- Early layers duplicate word clouds across adjacent (upsampled) positions: YES, precisely (L3 p0 comes in pairs:
  Fin/Fin, swim/swim, Ouch/Ouch, duck/duck). I missed this.
- Middle layers: rank spike + symmetry breaking: YES (mid depth has the highest effective rank at all 11 levels;
  pairs diverge by the last layer). "Middle = J-space sweet spot": NO (word alignment dips mid-stack).
- Coarse levels: low rank per slot, global coupling, nouns + discourse anchors: MOSTLY (L2 is ~1 direction with
  the largest norm; L3 holds nouns + One/day/Hi) but L2 itself reads as junk fragments.
- Mid levels = syntactic scaffolding/article collisions: NO (L4-L6 still mostly content).
- Fine levels: tiny Jacobian norms + spectrum collapse: YES on norms (~300x drop L2->L12); rank collapse at L9-L11
  but not L12.
- Late-layer denoising and "negation vectors": UNTESTED.

Verdict: Argon called NCM's dynamics better (upsampling duplicates, mid-layer rank spike, fine-level norm
collapse); Claude called the control model and where content sits (L3-L6, glue at L9). Neither predicted L2's
single junk direction or the "menu of alternatives" at L3. Joint finding: the mid-stack is where computation is
highest-rank and least word-like.

---
## CONCEPT SWAP (dog -> cat) — predictions recorded before running (2026-10-20 ~02:15)

Design: push hidden states along unit(J-lens cat dir - J-lens dog dir) at the last generator layer (input to ln_f),
all code positions of one level; regenerate the remaining levels with the SAME seed as an unswapped baseline. Third
condition: random unit direction, same strength. Measure the share of stories whose final text contains " cat" and
" dog". Control model: same push at the " dog" token of a prompt (layers 1-3), continuation sampled with the same seed.

- Gemini/Argon: swap at L3 or L5 rewrites the downstream narrative cleanly (environment adapts too); at L7 or L9 the
  Jacobian is ~9x weaker -> no change, or a clumsy local splice where the context doesn't update.
- Claude: (6) a coarse-level swap changes the whole story; in the control, a swap at one position mostly changes the
  local continuation. Additional: random-direction pushes of the same size will change stories (sampling diverges)
  but will NOT raise " cat"; only the J-lens swap will.
- Success criterion (both): " cat" share rises >= 20 points above BOTH baseline and random control, " dog" falls.

### SWAP RESULTS (64 stories / 64 continuations; same seeds across conditions)
NCM baseline: cat 12%, dog 17%. Random pushes (all levels/strengths): cat 12-16%, dog 9-17% (no concept effect).
| level | strength 1 | 2 | 4 | what the story looks like |
|---|---|---|---|---|
| L3 | cat 72% dog 11% | 98% / 23% | 100% / 28% | whole story rewritten ("big dog and small cat playing in the park" -> "big cat and small cat went on a trip"); grammar degrades at 4; dog not removed |
| L5 | cat 94% dog 6% | 98% / 5% | 98% / 3% | CLEANEST: world rewritten coherently - "girl Lucy walks her dog Max" -> "a little cat named Mimi went for a walk with her cat named Luna" (names become cat names; even the girl becomes a cat) |
| L7 | 84% / 16% | 84% / 17% | 81% / 12% | SPLICE: "walk with her cat, Max" - name, girl, plot unchanged |
| L9 | 78% / 14% | 80% / 12% | 81% / 12% | SPLICE + glitches ("The sun was singing singing", "went to to") |
Control baseline (8 prompts ending "...dog", 64 continuations): cat 9%, dog 55%. Random pushes remove "dog" at
strength >= 2 (the token gets scrambled) but never add cat (3-16%).
| layer | strength 1 | 2 | 4 |
|---|---|---|---|
| 1 | cat 44% dog 2% | 33% / 2% | 42% / 9% |  (early pushes give cat-ASSOCIATED animals: "Mouse")
| 2 | 39% / 3% | 42% / 9% | 72% / 6% |
| 3 | 92% / 3% | 100% / 19% | 100% / 12% |  ("a little dog' cat named Kitty..." - the continuation is a cat story, but the written "dog" can't be unwritten)

Scores:
- Gemini/Argon "L3/L5 rewrite the narrative cleanly": YES (L5 especially; L3 rewrites but gets noisy). "L7/L9: no
  change or a clumsy local splice where context doesn't update": SPLICE YES (cat named Max, girl Lucy unchanged);
  "no change" NO - the push still flips ~80% (strength is scaled to each level's own residual norm).
- Claude "coarse swap changes the whole story": YES. "Random pushes won't add cat": YES. "Control swap stays
  local": NO - a push at the single 'dog' position at the last layer turns the whole continuation into a cat
  story (through attention), it just can't change the word already written.
- Success criterion (cat +20 pts over baseline AND random, dog falls): L5 fully; L3/L7/L9 and control partly.
HEADLINE: in NCM, the LEVEL you edit sets the SCOPE of the edit - coarse = the whole world adapts (names, cast),
fine = one word swapped in place. In the word-by-word model, the edit only reaches the future; the past is fixed.

---
## NEGATION / ERASURE — predictions before running (2026-10-20 ~02:40)
Test: per story, the L3 draft's "menu" = content words with p >= 0.10 somewhere in the decoded L3 draft; CHOSEN = in
the final text, REJECTED = not. At the positions where each was drafted, track its logit minus the mean logit over
the vocabulary (centred, so global drift cancels), level by level L3 -> L12. Neutral baseline = random content
words never on the menu, same positions.
- Gemini/Argon: rejected menu items get ACTIVELY pushed down (negative update) at L4-L5 ("erasing the unchosen
  branches"); a layer's update has negative alignment with rejected words' directions.
- Claude: mostly PASSIVE - rejected words fall behind because chosen words climb, while their own centred logit
  drops only a little; clear active drops only where two options compete for the same slot (Fin vs Finny).

### ERASURE RESULTS (64 stories: 665 chosen, 583 rejected menu words, 1248 neutral samples)
Mean centred-logit change vs L3 at drafted positions (se 0.07-0.37):
  chosen   L4 +1.22  L5 +2.14  L6 +2.97  L7 +3.24  L8 +2.62  L10 +2.16  L12 +0.60
  rejected L4 -0.29  L5 -0.78  L6 -0.87  L7 -1.23  L8 -1.97  L10 -2.63  L12 -4.18
  neutral  flat (+0.01..+0.08)
- Active erasure is real: rejected words fall below the neutral baseline from L4 on, steadily, through every level.
- At L5 ~3/4 of the chosen-vs-rejected gap is chosen words rising, ~1/4 rejected falling; the push-down keeps
  growing and dominates later (partly positional sharpening: chosen words also drop at L8/L12 at their non-final slots).
- Fin story case study (max centred logit): rejected NAMES stall (Finny 12.4->~16 flat, Bubbles ~17 flat: passive);
  rejected ANIMALS are pushed down at commitment, L5->L6 (shark 19.2->15.5, swan 15.1->11.9, dolphin 16.7->12.9 by L7)
  while crab and duck lock in.
Scores: Gemini "active push-down of rejected items at L4-L5": SUPPORTED (starts at L4; continues through all levels).
Claude "mostly passive; active only for same-slot competitors like Fin/Finny": MOSTLY WRONG (systematic suppression;
and Fin/Finny was the passive case, the competing animals the active one).

---
## REPLICATION (2026-10-20 ~02:30, fresh seeds; both lenses refit on new stories) — probe/build_replication.py
Same measures, same code, original -> replication:
- control ratio last layer 1.47 -> 1.48; middle 1.19 -> 1.21
- NCM L3-L6 first+last layer ratio mean 3.63 -> 3.74 (range 2.91-4.10 -> 2.93-4.22); middle 1.63 -> 1.65; L7-L12 2.05 -> 2.03
- L2 effective rank 1.07 -> 1.07; Jacobian size L2/L12 312x -> 359x
- swap L5 (s=2): cat 98% -> 100%, dog 5% -> 5%; random cat 16% -> 9%; control layer 3 (s=1) cat 92% -> 98% (random 9% -> 8%)
- erasure: kept L5 +2.14 -> +2.07; dropped L5 -0.78 -> -0.73; dropped L12 -4.18 -> -4.26; neutral +0.03 -> -0.05
CORRECTIONS from the replication:
1. "Middle layer has the highest effective rank at all 11 levels" was overstated: 10/11 originally (L5 is flat:
   63/63/67), 9/11 on fresh seeds (L5 and L8).
2. "L5 is the only edit that removes the dog" does NOT replicate: on fresh seeds L7 removed it too (5% vs 12%
   baseline). What replicates: L3 adds a cat but keeps (even raises) the dog; L5 removes it.
NEW MEASURE (scope): share of each unedited story's words (cat/dog words excluded) surviving the edit, strength 2:
   L3 26%/26% (random 34%/33%), L5 43%/44% (50%/50%), L7 69%/72% (73%/75%), L9 89%/90% (91%/92%).
   => The LEVEL sets how much of the story changes for ANY push (random pushes change about as much); the J-lens
   DIRECTION sets what it changes to (only it adds cats). This replaces "level sets the scope of the concept edit".
Also corrected: decode_multiscales DOES return (earlier claim wrong; file lacks a trailing newline). The real
repo bug is idx_to_zhats in-place aliasing (reproduced). See probe/github_issue_draft.txt.

---
## SUDOKU — predictions written BEFORE any Sudoku data (2026-10-21 09:15 EDT, Claude alone; no Gemini this round)
Setup: NCM sudoku-hard / sudoku-easy checkpoints. Puzzle (91 tokens) = prefix; the model writes the 89-token solution
(81 digits + 8 '|') coarse-to-fine over 12 levels [1,2,4,6,9,13,18,26,37,50,67,89]. The VQ-VAE decoder never sees the
puzzle: every cell, clues included, must be carried by the codes. Drafts = decode the cumulative codes after each level.
Erasure measure: at each cell, logits over the 9 digit tokens, CENTRED over those 9 (sum is zero, so a winner rising
forces the rest down on average - which is why the key comparison is rejected vs neutral, both non-winners).
REFERENCE level (fixed now, by rule): the level where the mean number of digits with p >= 0.1 per blank cell is largest
(the widest menu). REJECTED = digits with p >= 0.1 at the reference level that the final grid doesn't use there;
NEUTRAL = digits with p < 0.01 at reference, not used. CHOSEN = the final digit.
S1 accuracy (exact-match grids, greedy): easy >= 85%; hard 30-80%. Cell accuracy on hard >= 90%.
S2 drafts: separators '|' are right from L1. Digit accuracy stays near chance (~11-25%) through L3 and passes 50% only
   around L7-L9. Clue cells are NOT resolved much earlier than blank cells (gap < 10 points at every level), because
   the decoder can't see the puzzle.
S3 menu: at the reference level, menus have 2-4 digits per blank cell, and the true digit is on the menu >= 80% of
   the time. Menu digits are enriched for the cell's legal pencil marks (digits not ruled out by the clues in its
   row/column/box), but only modestly: ratio 1.0-1.5x the base rate.
S4 erasure (updated on the stories result, where Gemini was right and I wasn't): rejected digits fall faster than
   neutral ones after the reference level, AND end the run with a LOWER absolute centred logit than neutral digits
   (they start higher and finish lower = active suppression, not regression to the mean). Gap at the final level >= 1 logit.
S5 failures (hard): in grids the model gets wrong, the true digit was on the menu at the reference level for most
   (>= 60%) of the wrong cells: errors are mostly a wrong pick between candidates, not a digit never considered.

### SUDOKU RESULTS (2026-10-21 ~09:40; 1000 hard + 1000 easy puzzles, greedy) — probe/sudoku.py, logs in I:\ncm\sudoku\
- Exact match: hard 84.2%, easy 99.6% (valid+clue-consistent = exact in both). Blank-cell acc hard 97.0%, easy 99.95%.
- Drafts (hard; easy within ~3 pts): separators right from L1. Digit accuracy L1 14% -> L3 17% -> L6 27% -> L8 45% ->
  L9 61% -> L11 80% -> L12 97%. Clue cells track blank cells within 3 pts at EVERY level up to L11 (83.6% vs 80.5%),
  then L12 takes clues to 100%. The last level (89 codes, one per cell) does a big share of the work.
- The pre-registered reference rule FAILED: "widest menu" = L1, where ~6 of 9 digits have p >= 0.1 (fog, not a menu),
  leaving only 663 neutral digits (hard). Scored as written below; then POST-HOC (probe/sudoku_followup.py):
  (A) erasure from EVERY reference level L1..L11; (B) a pencil-mark test needing no reference; (C) error ranks.
- (A) From every reference level, rejected digits fall far more than neutral ones (hard, change to L12:
  ref L1 -1.78 vs -0.77; ref L8 -3.50 vs +0.41; ref L11 -5.47 vs -0.04; easy the same). But they NEVER end below
  neutral: paired per-cell final gap rejected minus neutral is +1.1 to +2.5 (hard) / +1.5 to +3.0 (easy) at every
  reference; rejected ends below neutral in only 15-28% of cells.
- (B) The menus barely know the rules. Wrong candidates (p >= 0.1, not the true digit) are legal pencil marks
  29-37% of the time at L1-L11 vs 28.8% chance (hard; easy 18-22% vs 17.4%). Legal-wrong minus illegal centred
  logit: +0.01 at L1, +0.08 at L6, +0.38 at L9, +0.50 at L11 — against a true-digit lead of +7 by L11. Then at L12 the
  wrong digit written is a legal pencil mark in 98% (hard) / 95% (easy) of wrong cells (chance 34% / 24%).
  => Coarse-to-fine here is NOT pencil-marking. Mid-level drafts are a blurry copy of the answer plus noise that
  ignores the clues; the clue constraints show up almost entirely at the finest level.
- (C) In wrong cells the true digit is runner-up only 13-15% of the time at L9-L11 and 23% at L12 (chance 12.5%):
  errors are not close calls between the right digit and one wrong one. Wrong cells lock in later than right cells
  (locked from L9: 38% vs 50%).
Scores (Claude, pre-registered):
  S1 easy >= 85% RIGHT; hard 30-80% WRONG (84.2%, better than I guessed); cell acc >= 90% RIGHT.
  S2 RIGHT on all four parts (seps at L1; ~chance through L3; passes 50% at L9; clue gap < 10 pts).
  S3 rule degenerate. Menu 2-4 digits WRONG (5.9 at the chosen level); true digit on menu >= 80% WRONG (72%);
     enrichment 1.0-1.5x technically RIGHT (1.03x), and the post-hoc test says it's even weaker than I thought.
  S4 "falls faster than neutral" RIGHT at every reference; "ends BELOW neutral, gap >= 1 logit" WRONG (ends 1-3 above).
  S5 66% >= 60% technically RIGHT but UNINFORMATIVE: L1 menus hold ~6 of 9 digits, so ~66% is chance.
Contrast with the stories: there, coarse levels held a meaningful menu (Fin/Finny/Bubbles) and dropped words went
below neutral. In Sudoku the weak form of suppression holds (dropped candidates fall faster) but the strong form
doesn't, and the "menu" at coarse levels isn't a set of plausible answers. My read (interpretation, not tested): the
Sudoku tokenizer's hierarchy is a compression code for the finished grid, not a reasoning order, and the
generator's constraint work lands in the per-cell final level.
Caveats: greedy only; one checkpoint per difficulty; the post-hoc analyses were chosen after seeing S1-S5;
legal = ruled out by clues only (not by deeper deduction).

---
## SUDOKU CODE SWAP — predictions written BEFORE running (2026-10-21 ~10:35 EDT; Gemini asked separately via DJ, sealed)
Question: the NCM blog hypothesizes "coarse-to-fine generation supports global planning by establishing an overall
structure before resolving finer details." Do the coarse levels carry commitments the later levels depend on?
Test (hard puzzles, greedy): generate puzzle A, but force the codes of levels 1..k to be the ones the model
generated for a DIFFERENT puzzle B (the donor); the model then generates levels k+1..12 itself, still seeing
puzzle A as its prompt. k = 1, 3, 6, 9, 11. Controls: k levels of A's OWN codes re-injected (must equal baseline);
k levels of uniformly RANDOM codes. Measures: exact solve rate on A; clue-cell accuracy (does it still honor A's
clues?); donor leak = share of A's blank cells whose final digit equals B's solution digit there and not A's.
Claude's predictions:
W1 own-code re-injection = baseline exactly (sanity).
W2 solve rate (baseline 84.2%): k=1 small drop (>= 70%); k=3 moderate (40-70%); k=6 large (< 20%);
   k=9 and k=11 near zero (< 5%).
W3 donor leak rises with k: k=9 >= 30% of blank cells take B's digit (the late levels mostly finish B's grid).
W4 clue cells: still >= 90% right for k <= 6 (the last level enforces the prompt's clues, as in the drafts);
   below 90% at k = 9 and 11.
W5 random codes hurt about as much as donor codes at the same k (solve rate within 15 points), but with no donor leak.
If W2-W3 hold: early levels carry commitments the model can't undo later = consistent with the planning hypothesis.
If solve rates barely move: the fine levels re-solve from the prompt and the coarse levels aren't a plan.

### SUDOKU CODE SWAP RESULTS (2026-10-21 ~10:55; 1000 hard + 1000 easy, greedy) — probe/sudoku_swap.py, I:\ncm\sudoku\swap_*
Solved exactly (hard / easy); baseline 84.2% / 99.6%. Donor = levels 1..k taken from another puzzle's generation.
  k   codes forced   own codes      DONOR solved   DONOR clue   donor leak   RANDOM solved
  1        1         84.2 / 99.6    74.9 / 99.4    99.8 / 100   0.4 / 0.0    69.6 / 98.5
  3        7         84.2 / 99.6    75.4 / 99.3    99.8 / 100   0.4 / 0.0    65.1 / 97.0
  6       35         84.2 / 99.6    72.2 / 99.2    99.8 / 100   0.7 / 0.0    61.9 / 96.0
  9      116         84.2 / 99.6    35.7 / 76.1    99.4 / 99.7  2.3 / 0.2    28.1 / 66.0
 11      233         84.2 / 99.6     0.0 /  0.0    92.7 / 90.5 12.6 / 9.1     1.3 /  3.0
(leak = share of A's blank cells, where B's answer differs, that end with B's digit; no-swap baseline 0.4% / 0.0%;
 random-code leak stays 0.4-1.4%. No run ever produced B's full grid.)
- Levels 1-6 are NOT binding: replace them with another puzzle's codes and the model still solves 72% of hard
  puzzles (vs 84%) and 99.2% of easy ones. Almost all of that cost comes from the single level-1 code (k=1: 74.9%);
  swapping 34 more codes (k=6) costs ~3 points more.
- Commitment builds in levels 7-11: k=9 (116 codes) halves the hard solve rate, k=11 (233 codes) takes it to zero.
- Even with 233 codes of the wrong grid, the model doesn't finish puzzle B: the last level alone pulls the grid back
  to puzzle A (77% of blank cells right on hard, 93% of clues) and only 12.6% of cells take B's digit.
- Wrong-but-plausible coarse codes (donor) hurt LESS than random codes (k=6: 72% vs 62%), so coarse codes are used
  as context; but they're a soft prior the later levels override, not a plan they have to follow.
Scores (Claude, pre-registered):
  W1 own codes = baseline: RIGHT (identical at every k).
  W2 k=1 >= 70%: RIGHT (74.9). k=3 40-70%: WRONG (75.4). k=6 < 20%: WRONG (72.2). k=9 < 5%: WRONG (35.7).
     k=11 < 5%: RIGHT (0.0). I badly overestimated how binding the coarse levels are.
  W3 donor leak >= 30% at k=9: WRONG (2.3%; even k=11 only 12.6%).
  W4 clues >= 90% for k <= 6: RIGHT (99.8). Below 90% at k = 9, 11: WRONG (99.4, 92.7).
  W5 random within 15 points of donor, no leak: RIGHT (gap <= 10 points; random leak <= 1.4%).
For the blog's hypothesis ("coarse-to-fine generation supports global planning by establishing an overall structure
before resolving finer details"): in this checkpoint the coarse levels are a soft prior, not a binding plan. The
early structure helps (wrong coarse codes cost points, random ones more), but the finer levels re-solve from the
prompt and override it, and the commitment that matters is made at levels 7-11. Caveat: this tests whether the coarse
codes BIND the answer, not whether they help when they're right; greedy only; one checkpoint per difficulty.

### GEMINI's sealed code-swap prediction (DJ pasted it after the run; Gemini had not seen results) — scored, hard puzzles
Gemini's table (solve / clue / donor leak, point estimates) vs actual:
  k=1:  83.0 / 99.8 / 0.5   vs 74.9 / 99.8 / 0.4   -> solve WRONG (range 81-84), clue RIGHT, leak RIGHT
  k=3:  74.0 / 99.3 / 2.2   vs 75.4 / 99.8 / 0.4   -> solve RIGHT (range 68-78), clue RIGHT (range edge), leak WRONG (1.5-3.5)
  k=6:  29.0 / 95.0 / 16.0  vs 72.2 / 99.8 / 0.7   -> all three WRONG (predicted "the cliff" here)
  k=9:   1.5 / 81.0 / 45.0  vs 35.7 / 99.4 / 2.3   -> all three WRONG
  k=11:  0.0 / 44.0 / 74.0  vs  0.0 / 92.7 / 12.6  -> solve RIGHT, clue WRONG, leak WRONG
  Table: 5 of 15 cells in range.
Gemini P4 "clue beats blank by > 25 points at k=6, 9": WRONG on size (k=6 99.8 vs 95.9; k=9 99.4 vs 90.6), right
  on direction (clues do hold up better than blanks under the swap; gap 15 points at k=11).
Gemini P5 "random hurts more than donor at k=3, 6 by ~10-15 points": RIGHT (10.3 and 10.3 points). Its add-ons:
  "both barely hurt at k=1 (80-84%)" WRONG (74.9 / 69.6); "both 0-2% at k=9" WRONG (35.7 / 28.1); donor leak far above
  random RIGHT at k=11 (12.6 vs 1.4) but small; the k=9 clue comparison was hedged both ways (not scorable).
Gemini P6 verdict: "no puzzle-specific planning at L1-L3" CONSISTENT (k=1..3 barely matter beyond the single L1 code).
  "Global constraint solving happens across L4-L9; L10-L12 are purely local readout sharpeners" WRONG: swapping
  L1-L6 barely matters, the commitment builds at L7-L11, and the final level alone pulls a 233-code wrong grid back
  to 77% of puzzle A's blank cells. Its "residual capacity cliff" mechanism (fine levels too small to correct) is
  contradicted by that recovery.
HEAD TO HEAD: both models predicted the same wrong thing (coarse levels binding, big donor leak); Gemini more
  extremely (cliff at k=6, leak up to 74%). Gemini nailed the k=3 point estimate and the random-vs-donor gap; Claude
  got own-code identity, the k=1 floor, clue robustness for k <= 6, and k=11. Neither predicted the main result:
  levels 1-6 are a soft prior the model overrides.
Small factual slips in Gemini's premise, for the record: cumulative codes at k=9/11 are 116/233 (it wrote 110/227);
  "codebook norms decay in TinyStories" was not something we measured (we measured Jacobian size by level); "Stage 2
  trained with corrupted conditioning codes" is unverified.

### AUTHOR REPLY + CORRECTION (2026-10-21 16:31 EDT) — Hermann Kumbong, reply-all with the team CC'd
Hermann: the swap result is EXPECTED. (1) In conditional generation NCM sees the prompt at every level, so later
levels can use the puzzle's constraints to recover. (2) NCM is explicitly TRAINED to be robust to wrong earlier
codes ("conditioning corruption", details in their paper): during training some preceding codes are replaced with
incorrect ones while the targets stay the same. (3) Agrees the coarse levels are a soft plan, not a hard one, BY
DESIGN (a hard plan couldn't correct early mistakes), and notes they still matter (random codes: 61.9%).
CORRECTION: above I filed Gemini's premise "Stage 2 trained with corrupted conditioning codes" as "unverified".
It was RIGHT. Gemini knew a training detail I didn't, though its numbers still predicted binding coarse levels.
Also: our page framed the swap as a test the planning hypothesis might fail ("If the early levels set a plan, the
model should get stuck..."). Per the authors, a soft, correctable plan IS the intended design, so the result is
consistent with their hypothesis, not evidence against it. We hadn't read the NCM paper; should have.

---
## TINYSTORIES CODE SWAP — Gemini's predictions, recorded BEFORE any run (2026-10-21 ~16:55 EDT)
(Not sealed: DJ pasted them into the chat, so Claude has seen them. Any Claude prediction written after this is
NOT blind to Gemini's.) Proposed test: the Sudoku swap applied to TinyStories: story A's generation gets story B's
codes for levels 1..k, then the model generates the rest. Leak = share of B's content (cast, nouns) in the final story.
Gemini noted: TinyStories is unconditional, so there's no prompt A pulling back; with greedy decoding injecting B's
codes would trivially reproduce B (our TinyStories runs SAMPLE at temperature 1, so "A" = seed A's noise).
Gemini, unconditional case: k=1 leak 10-25%; k=3 55-75%; k=6 85-95%.
Gemini, prompt-conditioned case: k=3 30-50%; k=6 65-85%; k=9 > 90%.
Gemini, headline (85% confidence): at k=6 TinyStories donor leak >= 40x Sudoku's 0.7% (i.e. >= 28%).
Gemini also corrected itself: its suggested "quote" of Hermann was its own paraphrase (its email tool returned a
summary), and "96% cat" was a midpoint, not a measured number.

## TINYSTORIES CODE SWAP — Claude's predictions (2026-10-21 ~17:20 EDT; NOT blind: written after reading Gemini's)
Design (fixed now): 128 stories sampled at temperature 1 from one seed = the A stories. Swap run: same seed, so the
same random draws, but levels 1..k forced to story B's codes (B = the previous story in the batch); the sampler is
still called at forced levels so the random stream stays aligned. k = 1, 2, 3, 5, 7, 9, 11 (cumulative codes 1, 9,
25, 81, 173, 313, 513 of 641; arithmetic fixed before the run). Controls: A's own codes (must reproduce A exactly) and uniformly random codes.
Measures (content words: alphabetic, >= 3 letters, not stopwords): LEAK = share of B's words not in A that appear in
the final story; KEEP = share of A's words not in B that survive; CHANCE = the same shares for an unrelated story C.
FOLLOWS-B = share of stories with LEAK > KEEP.
C1 own codes reproduce A exactly at every k (RNG alignment works).
C2 k=1: the story follows NEITHER: KEEP falls to <= 40% (one different code + same noise = a new story), and LEAK
   stays within 10 points of CHANCE.
C3 LEAK by k: k=2 20-45%; k=3 (the cast level) 50-75%; k=5 75-90%; k >= 7 >= 90%. FOLLOWS-B >= 80% from k=3.
   (Close to Gemini's numbers; I agree with its direction, so this is not an independent call.)
C4 Headline vs Sudoku: at matched depth (k=6 Sudoku, 0.7% leak; k=5-7 here) TinyStories leak is >= 50x higher.
C5 Random codes: stories stay fluent (the model was trained on random code swaps), with LEAK at chance and KEEP
   low (<= 40%) at k >= 2: random coarse codes push the story somewhere new rather than breaking it.
Mechanism I expect: with no prompt there is nothing for later levels to recover TOWARD, so whatever the early codes
say is treated as the story. Conditioning corruption teaches recovery to the evidence; here the only evidence is B.

### TINYSTORIES CODE SWAP RESULTS (2026-10-21 ~17:45; 128 stories x 2 seeds) — probe/stories_swap.py, I:\ncm\stories_swap_*
Chance (unrelated story): 12.1% / 12.8% of B's distinctive words, 11.4% / 12.4% of A's.  seed 21 / seed 22:
  k  codes   own = A   DONOR leak     keep         follows B    RANDOM leak   keep
  1      1    100%     9.4 / 8.7     30.0 / 33.1   16 / 12%     7.5 / 7.5    32.4 / 38.5
  2      9    100%    20.0 / 21.0    11.7 / 12.2   73 / 70%    10.8 / 10.7   14.3 / 15.1
  3     25    100%    34.5 / 34.2     7.8 / 8.6    96 / 97%    11.4 / 10.8   12.0 / 11.6
  5     81    100%    64.5 / 65.8     3.2 / 3.2   100 / 100%    9.1 / 8.8     8.6 / 9.6
  6    121    100%    77.1 / 77.1     2.0 / 2.1   100 / 100%    8.5 / 8.4     7.9 / 8.4   (run after, for Gemini's k=6)
  7    173    100%    84.2 / 83.5     1.4 / 1.3   100 / 100%    7.4 / 7.7     7.2 / 7.2
  9    313    100%    94.4 / 94.0     0.3 / 0.4   100 / 100%    6.1 / 6.3     6.3 / 5.9
 11    513    100%    97.2 / 96.6     0.1 / 0.1   100 / 100%    5.4 / 5.6     5.4 / 5.7
- With no prompt, the early codes ARE the story: from level 3 (25 codes) 96-97% of stories follow the donor, and by
  level 6 they carry 77% of its distinctive words (Sudoku at a similar depth: 0.7%). Same model family, same
  training trick, opposite outcome: what decides whether early codes stick is whether anything in the prompt
  contradicts them.
- Level 1 alone (one code) sends the story somewhere new: neither A's words (30%) nor B's (at chance) survive.
- Gradual, not a cliff: leak rises steadily with every level (20, 34, 65, 77, 84, 94, 97%).
- Random codes yield new stories (leak and keep both near or below chance). Fluency not measured.
Scores, Claude (not blind):
  C1 own codes reproduce A: RIGHT (100% at every k, both seeds).
  C2 k=1 follows neither (keep <= 40%, leak within 10 pts of chance): RIGHT.
  C3 leak k=2 20-45%: RIGHT (20-21, at the edge). k=3 50-75%: WRONG (34). k=5 75-90%: WRONG (65). k=7 >= 90%:
     WRONG (84). k=9, 11 >= 90%: RIGHT. Follows-B >= 80% from k=3: RIGHT (96-100%). Leak climbs slower than I said.
  C4 >= 50x Sudoku at matched depth: RIGHT (~110x at k=6).
  C5 random codes: leak at chance, keep <= 40% from k=2: RIGHT. "Stays fluent": NOT MEASURED.
Scores, Gemini (unconditional case):
  k=1 leak 10-25%: WRONG (just under: 8.7-9.4%, at chance). k=3 55-75%: WRONG (34). k=6 85-95%: WRONG (77).
  Headline, k=6 leak >= 40x Sudoku's 0.7% (>= 28%), 85% confidence: RIGHT (77%, ~110x).
  Gemini's "prompt strength decides whether coarse edits stick" (the hypothesis this tested): SUPPORTED, with the
  caveat that TinyStories vs Sudoku also differs in model and task, not only in prompt strength.
Both models had the direction right and the speed wrong: we both expected the cast level (L3) to carry most of the
story. It carries the cast (96% follow B) but only about a third of B's distinctive words; the rest is written
level by level.
