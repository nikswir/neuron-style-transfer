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

**Hard limit: 79 characters, measured from the first non-whitespace character
— leading indentation does not count.** So the budget is 79 chars of actual
content at any nesting depth, not 79 minus the indent. An extra line is always
better than a long one; when the content does not fit in its 79, explode it
vertically, one element per line.

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

## 5. Putting it together

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
