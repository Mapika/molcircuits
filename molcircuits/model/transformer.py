"""LLaMA-style decoder with a hook point on every activation an analysis needs.

Design notes
------------
* The FFN is a genuine three-matrix gated SwiGLU unit. The old codebase described
  this as "SiLU activations" in one place and "SwiGLU" in another; here there is
  one implementation and one name.
* Embeddings are tied. The old code allocated an unused ``lm_head`` when tying
  was on, inflating every reported parameter count; ``n_params()`` below counts
  what actually exists.
* Hook points follow TransformerLens naming (``blocks.{i}.hook_resid_post`` etc.)
  so activations are directly comparable with that ecosystem, and
  ``to_transformer_lens_config()`` emits a matching config.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 64
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    n_ctx: int = 256
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as fh:
            return cls(**json.load(fh))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope(n_ctx: int, d_head: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(n_ctx, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    # x: (B, H, T, Dh). ``offset`` is the number of cached positions preceding
    # this chunk, so an incrementally decoded token gets the rotation for its
    # true absolute position rather than for position 0.
    T = x.shape[-2]
    cos, sin = cos[offset:offset + T][None, None], sin[offset:offset + T][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head
        self.W_Q = nn.Linear(d, h * dh, bias=False)
        self.W_K = nn.Linear(d, h * dh, bias=False)
        self.W_V = nn.Linear(d, h * dh, bias=False)
        self.W_O = nn.Linear(h * dh, d, bias=False)
        # Per-head multiplicative gate: 1.0 normally, 0.0 to zero-ablate a head.
        # Kept as a buffer so ablation needs no weight surgery and is exactly
        # reversible -- the old implementation mutated W_O in place.
        self.register_buffer("head_mask", torch.ones(h), persistent=False)
        self.register_buffer("head_bias", torch.zeros(h, dh), persistent=False)

    def forward(self, x, cos, sin, hooks=None, cache=None, offset: int = 0):
        B, T, _ = x.shape
        h, dh = self.cfg.n_heads, self.cfg.d_head
        q = self.W_Q(x).view(B, T, h, dh).transpose(1, 2)
        k = self.W_K(x).view(B, T, h, dh).transpose(1, 2)
        v = self.W_V(x).view(B, T, h, dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin, offset), apply_rope(k, cos, sin, offset)

        if cache is not None:
            if cache.get("k") is not None:
                # Only single-token extension is supported: a multi-token chunk
                # appended to existing context would need an explicit rectangular
                # causal mask, which `is_causal` cannot express.
                assert T == 1, "KV cache supports one new token at a time"
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        need_pattern = hooks is not None and hooks.wants(f"blocks.{self.layer}.attn.hook_pattern")
        if need_pattern and cache is not None:
            raise ValueError("attention-pattern capture is incompatible with the "
                             "KV cache: only the new row of the pattern exists")
        if need_pattern:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            pattern = scores.softmax(-1)
            # `maybe`, not `emit`: emit is capture-only, so a registered EDIT on
            # the pattern was computed, ignored, and the unedited pattern used.
            # `wants()` returns True for edits, so the manual-softmax path was
            # entered for them -- the intervention looked wired up and did
            # nothing (a full-zero pattern edit moved the logits by 6e-8).
            pattern = hooks.maybe(f"blocks.{self.layer}.attn.hook_pattern", pattern)
            z = pattern @ v
        else:
            z = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

        # per-head ablation: mask replaces the head's output with head_bias
        m = self.head_mask.view(1, h, 1, 1)
        z = z * m + self.head_bias.view(1, h, 1, dh) * (1 - m)
        if hooks is not None:
            z = hooks.maybe(f"blocks.{self.layer}.attn.hook_z", z)
        z = z.transpose(1, 2).reshape(B, T, h * dh)
        return self.W_O(z)


class MLP(nn.Module):
    """Gated SwiGLU: down(silu(gate(x)) * up(x)). Three matrices, not two."""

    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.W_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.W_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.W_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, hooks=None):
        a = F.silu(self.W_gate(x)) * self.W_up(x)
        if hooks is not None:
            a = hooks.maybe(f"blocks.{self.layer}.mlp.hook_post", a)
        return self.W_down(a)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.ln1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg, layer)
        self.ln2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = MLP(cfg, layer)

    def forward(self, x, cos, sin, hooks=None, cache=None, offset: int = 0):
        if hooks is not None:
            x = hooks.maybe(f"blocks.{self.layer}.hook_resid_pre", x)
        x = x + self.attn(self.ln1(x), cos, sin, hooks, cache, offset)
        if hooks is not None:
            x = hooks.maybe(f"blocks.{self.layer}.hook_resid_mid", x)
        x = x + self.mlp(self.ln2(x), hooks)
        if hooks is not None:
            x = hooks.maybe(f"blocks.{self.layer}.hook_resid_post", x)
        return x


class MolecularTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model, cfg.rms_eps)
        if cfg.tie_embeddings:
            self.lm_head = None          # genuinely absent, not allocated-and-ignored
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._rope = None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def _rope_for(self, device, dtype):
        if self._rope is None or self._rope[0].device != device or self._rope[0].dtype != dtype:
            self._rope = build_rope(self.cfg.n_ctx, self.cfg.d_head, self.cfg.rope_theta,
                                    device, dtype)
        return self._rope

    def forward(self, tokens: torch.Tensor, hooks=None, cache=None) -> torch.Tensor:
        """``cache`` is a list of per-layer dicts, mutated in place.

        Passing one turns generation from O(T^2) into O(T): without it every
        sampled token re-runs the whole prefix through the model.
        """
        x = self.embed(tokens)
        cos, sin = self._rope_for(x.device, x.dtype)
        offset = 0
        if cache is not None and cache[0].get("k") is not None:
            offset = cache[0]["k"].shape[2]
        if hooks is not None:
            x = hooks.maybe("hook_embed", x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, hooks, None if cache is None else cache[i], offset)
        x = self.ln_f(x)
        if hooks is not None:
            x = hooks.maybe("hook_resid_final", x)
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        return x @ w.T

    # -- ablation -------------------------------------------------------------
    @contextmanager
    def ablate_head(self, layer: int, head: int, mode: str = "zero",
                    mean_z: torch.Tensor | None = None):
        """Temporarily disable one attention head.

        mode="zero"      : replace the head's output with 0 (off-distribution)
        mode="mean"      : replace with a precomputed mean activation
        Both are reversible and touch no weights.
        """
        attn = self.blocks[layer].attn
        # VALIDATE BEFORE MUTATING. The head_mask write used to happen first and
        # the raises sat outside the try/finally, so a bad mode or a missing
        # mean_z left the head permanently zeroed with nothing to restore it:
        # __enter__ propagated the error and every later forward pass in that
        # process silently used a lobotomised model (logits moved by 0.065 in
        # the repro). Nothing in the shipped pipeline hits it; an ad-hoc call or
        # a caught exception in a sweep would.
        if mode not in ("zero", "mean"):
            raise ValueError(f"unknown ablation mode {mode!r}")
        if mode == "mean" and mean_z is None:
            raise ValueError("mode='mean' requires mean_z of shape (n_heads, d_head)")

        old_mask = attn.head_mask.clone()
        old_bias = attn.head_bias.clone()
        attn.head_mask[head] = 0.0
        if mode == "mean":
            attn.head_bias[head] = mean_z[head].to(attn.head_bias.dtype)
        try:
            yield
        finally:
            attn.head_mask.copy_(old_mask)
            attn.head_bias.copy_(old_bias)

    # -- interop --------------------------------------------------------------
    def to_transformer_lens_config(self) -> dict:
        """Config dict shaped like HookedTransformerConfig.

        WARNING -- this is not drop-in. ``apply_rope`` here uses the interleaved
        (GPT-J) pairing ``x[..., 0::2] / x[..., 1::2]``, while TransformerLens
        rotary uses the split-half convention, and no ``rotary_dim`` is emitted.
        Loading these weights into TransformerLens without converting the rotary
        layout produces silently wrong activations, not an error. Hook *names*
        follow TransformerLens so the analyses read the same; the weights do not
        transfer unchanged.
        """
        c = self.cfg
        return {
            "d_model": c.d_model, "n_layers": c.n_layers, "n_heads": c.n_heads,
            "d_head": c.d_head, "d_mlp": c.d_ff, "n_ctx": c.n_ctx,
            "d_vocab": c.vocab_size, "act_fn": "silu", "gated_mlp": True,
            "normalization_type": "RMS", "positional_embedding_type": "rotary",
            "rotary_base": c.rope_theta, "tie_embeddings": c.tie_embeddings,
            "attn_only": False, "final_rms": True,
            "rotary_dim": c.d_head,
            "_rotary_layout": "interleaved (GPT-J); TransformerLens expects "
                              "split-half -- convert before loading weights",
        }
