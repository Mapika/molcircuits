"""Muon (momentum-orthogonalized) for 2D hidden weights, AdamW for everything else.

Reference: Jordan et al., https://kellerjordan.github.io/posts/muon/. Embeddings,
norms and any 1D parameter go to AdamW; only the 2D matrices inside blocks are
orthogonalized. ``build_optimizers`` returns both so the caller can log which
parameter went where -- the previous project never recorded the split.

Performance note
----------------
``step`` used to walk the parameter list in Python and run the Newton-Schulz
iteration one matrix at a time: with 42 matrices and 5 iterations that is ~1,400
kernel launches per optimizer step, each on a matrix small enough (512x512 /
512x2048) that the launch dominates the arithmetic. The step was launch-bound,
not compute-bound, and Muon alone accounted for ~78% of the wall time of a
training step.

The matrices fall into a handful of distinct shapes, and the iteration's
``transposed`` normalization (work on the wide orientation) collapses them
further, so they are stacked once per step and orthogonalized with batched
``bmm``s; the momentum, weight-decay and parameter updates use the ``_foreach_``
kernels. Same arithmetic, same order per parameter, ~30 launches instead of
~1,400. ``_batched_newtonschulz5`` mirrors ``zeropower_via_newtonschulz5``
expression for expression, including the ``(c * A) @ A`` association, so the two
agree to float32 round-off.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration for an approximate orthogonalization."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def batched_newtonschulz5(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """``zeropower_via_newtonschulz5`` on a stack of same-shaped matrices.

    ``X`` is (K, m, n) float32 in the *wide* orientation (m <= n); the caller is
    responsible for the transpose bookkeeping. Returns a (K, m, n) tensor.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + (c * A) @ A          # matches `b * A + c * A @ A` exactly
        X = a * X + B @ X
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            ns_steps=ns_steps, weight_decay=weight_decay))
        self._plans: dict[int, tuple] = {}

    # -- grouping -------------------------------------------------------------
    def _plan(self, gi: int, params: list[torch.Tensor]) -> tuple[dict, dict]:
        """Cached bucketing of a param group.

        ``ns_groups`` maps (device, m, n) with m <= n -> list of (index,
        transposed); every member enters one batched Newton-Schulz call.
        ``apply_groups`` maps (device, rows, cols) -> (indices, scale); members
        share the update scale, so the parameter update is one ``_foreach_add_``.
        The bucketing depends only on the parameter shapes, so it is computed
        once and reused for the whole run.
        """
        plans = getattr(self, "_plans", None)
        if plans is None:
            plans = self._plans = {}
        key = tuple(id(p) for p in params)
        cached = plans.get(gi)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        ns_groups: dict[tuple, list] = {}
        apply_groups: dict[tuple, tuple] = {}
        for i, p in enumerate(params):
            rows, cols = p.size(0), p.size(1)
            transposed = rows > cols
            shape = (cols, rows) if transposed else (rows, cols)
            ns_groups.setdefault((p.device, *shape), []).append((i, transposed))
            ak = (p.device, rows, cols)
            if ak not in apply_groups:
                apply_groups[ak] = ([], max(1.0, rows / cols) ** 0.5)
            apply_groups[ak][0].append(i)
        plans[gi] = (key, ns_groups, apply_groups)
        return ns_groups, apply_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gi, group in enumerate(self.param_groups):
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            momentum, lr = group["momentum"], group["lr"]
            wd, ns_steps = group["weight_decay"], group["ns_steps"]
            grads = [p.grad for p in params]

            bufs = []
            for p, g in zip(params, grads):
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                bufs.append(st["buf"])
            # buf.mul_(momentum).add_(g), for every parameter at once
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)
            if group["nesterov"]:
                upds = torch._foreach_add(grads, bufs, alpha=momentum)
            else:
                upds = list(bufs)

            if any(p.ndim != 2 for p in params):
                # Not what build_optimizers produces; keep the original path so
                # the behaviour (including its errors) is unchanged.
                out = [zeropower_via_newtonschulz5(u, ns_steps) for u in upds]
                for p, o in zip(params, out):
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(o, alpha=-lr * scale)
                continue

            ns_groups, apply_groups = self._plan(gi, params)
            out: list[torch.Tensor | None] = [None] * len(params)
            for members in ns_groups.values():
                stack = torch.stack([
                    (upds[i].mT if tr else upds[i]).float() for i, tr in members])
                Y = batched_newtonschulz5(stack, ns_steps)
                for j, (i, tr) in enumerate(members):
                    y = Y[j]
                    out[i] = (y.mT if tr else y).to(upds[i].dtype)

            if wd:
                torch._foreach_mul_(params, 1 - lr * wd)
            for indices, scale in apply_groups.values():
                torch._foreach_add_([params[i] for i in indices],
                                    [out[i] for i in indices], alpha=-lr * scale)
        return loss


def build_optimizers(model: nn.Module, muon_lr: float = 0.02, adam_lr: float = 3e-4,
                     weight_decay: float = 0.01, betas=(0.9, 0.95)):
    """Split parameters and return (muon, adamw, assignment_log)."""
    muon_p, adam_p, log = [], [], {"muon": [], "adamw": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_p.append(p); log["muon"].append(name)
        else:
            adam_p.append(p); log["adamw"].append(name)
    opt_muon = Muon(muon_p, lr=muon_lr, weight_decay=weight_decay) if muon_p else None
    opt_adam = torch.optim.AdamW(adam_p, lr=adam_lr, betas=betas,
                                 weight_decay=weight_decay) if adam_p else None
    return opt_muon, opt_adam, log
