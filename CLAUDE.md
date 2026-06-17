# AGENT.md — Code Style

Universal, project-independent rules for how to write code (primarily Python).
Apply them to every file you create or edit, regardless of the repository.

**Guiding principle:** code is read top-to-bottom in blocks — make the block
structure visible. Every logical block gets a short introducing comment and is
separated from its neighbours by a blank line. Lines stay narrow; when a
construct does not fit, it explodes vertically rather than crowding the line.

---

## 1. Two tiers of structural comments

There are exactly two kinds of structural comment. Use the right tier for the
size of the thing it introduces.

### Tier 1 — section banners

Introduce a **large** unit: a class, a group of related methods, a major phase
of a long function. Use a boxed banner of `#`:

```
################################
#       Gate diagnostics       #
################################
```

Rules:

- Three lines: a full `#` border, the label line, another full `#` border.
- The label is **centered** between the side `#`. Split the padding evenly; if
  it cannot be exact, put the extra space on the right.
- **Every banner in one file shares the same width.** Pick one width (the
  length of the `#` run) per file — wide enough for the longest label plus a
  few spaces of breathing room — and reuse it for every banner, no matter how
  short the label. They must line up.
- Indent the banner to match the code it heads (e.g. 4 spaces for a
  method-group banner inside a class). The width is the `#`-run length and does
  not count indentation.

Same width, different labels — note how they align:

```
################################
#           Forward            #
################################
```
```
################################
#      Covariance factor       #
################################
```

### Tier 2 — inline block intros

Inside a function/method, a **short** comment introduces a small group of
tightly-related lines (sometimes a tiny undocumented helper). Keep it light:

```
# ── Normalizer ───────────────────────────────
obs_norm = self.obs_normalizer(sigs, ts)
x = obs_norm[:, :, -1]
```

- Prefix with `# ──` and trail a thin `─` line to a modest, consistent width.
  A plain `# comment` above the block is acceptable too — the rule is *one
  comment introduces one block*, not the exact glyphs.
- When the blocks form a sequence, number them so the flow is obvious:
  `# ── 1. Shared trunk ──`, `# ── 2. Shared expert ──`, `# ── 3. Gate ──`.

---

## 2. Blank lines separate blocks (most important)

Every logical block is set off by **a blank line AND its intro comment**. Never
let two unrelated blocks touch. One blank line between blocks inside a function;
two blank lines between top-level defs/classes. The blank-line + comment pair is
what makes the file skimmable — do not skip it to save space.

---

## 3. Narrow code — break early, never crowd a line

**Hard limit: 100 columns — the same `line-length` the project's `ruff format`
enforces, so the code, this guide and the linter all agree.** An extra line is
always better than a long one; when the content does not fit in its 100,
explode it vertically, one element per line.

Dicts / collections — one element per line, trailing comma:

```
stats = {
    "lb_total": loss.item(),
    "lb_ind_entropy": ind_entropy.mean().item(),
    "lb_batch_entropy": batch_entropy.item(),
}
```

Function calls — explode when they overflow; nested structures break too:

```
load_balance_loss, lb_stats = self.compute_sparse_balance_loss(
    top_k_indices, gate_probs,
)

W_all = torch.cat(
    [
        W_shared.unsqueeze(0),
        torch.stack([eh.weight for eh in self.expert_heads]),
    ],
    dim=0,
)
```

Long strings — never one long literal. Split into adjacent string literals
inside parentheses; Python concatenates them implicitly:

```
raise ValueError(
    f"shared_expert_units[-1] ({shared_expert_units[-1]}) != "
    f"expert_units[-1] ({expert_units[-1]}): "
    f"latent_dim must match for cov_factor aggregation"
)
```

---

## 4. Trailing commas (the "magic trailing comma")

Whenever a signature, call, or collection spans multiple lines, end the last
element with a comma. Two reasons: diffs stay clean (adding an element touches
one line, not two), and formatters (Black / ruff-format) read the trailing
comma as "keep this exploded one-per-line" instead of collapsing it.

Multi-parameter signatures go one parameter per line, each with a trailing
comma:

```
def __init__(
    self,
    state_dim: int,
    action_dim: int,
    num_experts: int,
    alpha: float = 0.3,
):
```

Short signatures that fit comfortably on one line stay on one line:

```
def set_profiler(self, profiler: Profiler) -> None:
```

Always type-annotate parameters and return values.

---

## 5. Block ladders — order sibling lines by length

Code is read as blocks; ordered line lengths make a block's structure pop.
Within a run of *parallel sibling lines* — call arguments, list/tuple elements,
dict entries, `__all__` exports, a group of CLI flags — order the lines so their
**lengths form a ladder**: monotonically rising or falling, no jagged spikes.
Write it laddered *from the first draft*, not as a later cleanup pass.

Approximate is fine. A "mountain" (rise then fall) or a single small bump still
reads as order — the goal is *chaos → order*, not a flawless staircase.

**Logic always outranks the ladder.** Reorder only where order is genuinely
free — i.e. it cannot change behaviour. Never trade meaning for shape:

- Primary stays ahead of secondary. Don't float an optional/secondary item in
  front of a primary one to win a length (keep `content_layers` before
  `content_weight`).
- The primary pair stays adjacent: `content` before `style`, always.
- A primary *group* stays first even if entering it costs one length jump —
  take the jump rather than invert primacy.
- An aggregate stays last: `total` after its parts, a sum after its terms.
- Never touch order that *is* meaning: positional parameters (the call
  signature), flag/value pairs, data-dependency chains (`b` built from `a`), the
  words of a split string literal.

Example — constructor kwargs (order is free) laddered while the layers group
stays first and the `content → style` pairs stay adjacent:

```
    cfg = TransferConfig(
        content_layers=content_layers,
        style_layers=style_layers,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        steps=args.steps,
    )
```

Lengths `38, 34, 43, 39, 33, 33, 30, 25`: the layers pair leads (38, 34), then a
jump up into the weights and a smooth descent 43 → 25. Run-params always go
`optimizer → learning_rate → steps`.

**Measure with a tool, never by eye** — counting columns in your head is
unreliable. Read each line's length with `awk` over the block's line range:

```
awk 'NR>=150 && NR<=157 { print length, $0 }' src/style_transfer/cli.py
```

A ladder rises or falls with no jagged spike. (`length` counts bytes — fine for
ASCII code lines; don't measure `─`/`→` banner lines.) Then reorder a jagged
block, or confirm its jump is logic-driven and leave it (a forced valley, e.g. a
dataclass's no-default fields first, is fine).

---

## 6. Import blocks — four ladders

Imports form up to four blocks, blank-line separated, in this fixed order
(`from __future__ …` is always pinned first, above everything):

1. `import <third-party>`        — plain imports of stdlib / third-party
2. `from <third-party> import …`
3. `import <first-party>`        — plain imports of this project's own packages
4. `from <first-party> import …`

Two ordering axes: `import` before `from …`, and *third-party before
first-party*. "Third-party" covers everything external — the standard library
and installed packages alike; "first-party" is this project's own code. **Always
use absolute imports** (`from package.models import …`), never relative
(`from .models …`). **Each block is an ascending ladder** — sort its lines
short → long, and sort the names inside a `from X import a, b, c` the same way.

```
from __future__ import annotations

import torch
import argparse

from pathlib import Path

from package.models import build_extractor
from package.transforms import load_image, save_image
from package.transfer import TransferConfig, run_style_transfer
```

This is why ruff's isort rule (`I`) is **off** — it sorts alphabetically by its
own stdlib/third-party/first-party sectioning and would fight this scheme.
`ruff format` leaves import order alone, so nothing re-sorts them; keep the
blocks tidy by hand (measure with `awk` as in §5).

---

## 7. Putting it together

```
class Router(nn.Module):

    ################################
    #         Gate routing         #
    ################################
    
    def gate_forward(self, h: torch.Tensor):

        # ── 1. Scores ──────────────────────────
        logits = self.gate(h)
        probs = F.softmax(logits, dim=-1)

        # ── 2. Top-k with tie-break ────────────
        noise = torch.rand_like(probs) * 1e-7
        vals, idx = torch.topk(
            probs + noise,
            self.top_k,
            dim=-1,
        )

        # ── 3. Renormalize the kept weights ────
        weights = vals / (vals.sum(dim=-1, keepdim=True) + 1e-8)
        return weights, idx, probs
```
