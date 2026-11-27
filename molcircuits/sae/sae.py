"""Sparse autoencoders: ReLU, TopK and JumpReLU under one interface.

Two things the previous implementation did not do, both of which reviewers asked
for directly:

* ``lambda_l1`` / ``k`` and the realized ``L0`` are saved *into the checkpoint*.
  The old saved configs had the sparsity coefficient overwritten with 0.0, so the
  released dictionaries could not be reproduced even in principle.
* ``dead_feature_fraction`` is tracked during training rather than inferred
  afterwards.

Checkpoint layout is deliberately close to SAELens conventions (``W_enc``,
``b_enc``, ``W_dec``, ``b_dec``, with unit-norm decoder columns) so dictionaries
can be exported to that ecosystem for evaluation with SAEBench.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class _JumpReLU(torch.autograd.Function):
    """z = x * H(x - theta), with a straight-through estimator for theta.

    The Heaviside step has zero gradient almost everywhere, so theta cannot be
    learned by ordinary backprop. Following Rajamanoharan et al. (2024), the
    threshold gradient is estimated with a rectangle kernel of width
    ``bandwidth`` centred on the threshold; the factor of ``theta`` folds in
    d(theta)/d(log theta), since the parameter is stored in log space to keep it
    positive.
    """

    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        theta = log_threshold.exp()
        ctx.save_for_backward(x, theta, torch.tensor(bandwidth, device=x.device))
        return x * (x > theta)

    @staticmethod
    def backward(ctx, grad_out):
        x, theta, bw = ctx.saved_tensors
        x_grad = (x > theta) * grad_out
        t_grad = -(theta / bw) * _rectangle((x - theta) / bw) * grad_out
        return x_grad, t_grad.sum(tuple(range(t_grad.dim() - 1))), None


class _Step(torch.autograd.Function):
    """H(x - theta), used for the L0 penalty. Same estimator, without the x factor."""

    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        theta = log_threshold.exp()
        ctx.save_for_backward(x, theta, torch.tensor(bandwidth, device=x.device))
        return (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, theta, bw = ctx.saved_tensors
        t_grad = -(theta / bw) * _rectangle((x - theta) / bw) * grad_out
        return torch.zeros_like(x), t_grad.sum(tuple(range(t_grad.dim() - 1))), None


@dataclass
class SAEConfig:
    d_in: int
    expansion: int = 4
    variant: str = "relu"          # relu | topk | jumprelu
    # relu only. In normalized loss units the useful range is O(1): a sweep at
    # layer 3, expansion 16 gives L0 419 at 0.1, 172 at 0.3, 34 at 1.0, 11 at 3.0.
    # The previous 5e-4 default predates the loss-scale fix in `loss()` and is
    # indistinguishable from no penalty at all (L0 723.0 vs 723.9 at l1_coeff=0).
    l1_coeff: float = 1.0
    k: int = 32                    # topk only
    # jumprelu only. jump_threshold is the *initial* value of the learned
    # per-feature threshold; l0_coeff weights the explicit L0 penalty and
    # jump_bandwidth is the width of the straight-through estimator's kernel.
    # The previous implementation had jump_threshold as a fixed constant and
    # applied no sparsity term at all, so the variant produced a dense code
    # (L0 511 of 1024) that scored a perfect delta_ce -- the same trap the ReLU
    # loss-scale bug produced.
    # l0_coeff needs calibrating per activation distribution, like l1_coeff: on a
    # 16-factor synthetic set it gives L0 77 at 0.0, 48 at 0.1 and 14 at 0.5.
    # Note the threshold moves only in log space at the optimizer's learning
    # rate, so a short run cannot separate coefficients -- at 390 steps every
    # value from 0 to 0.2 looks identical.
    jump_threshold: float = 0.1
    jump_bandwidth: float = 0.1
    l0_coeff: float = 0.2
    layer: int = 0
    hook: str = "hook_resid_post"
    normalize_input: bool = True

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion


class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in, d_sae = cfg.d_in, cfg.d_sae
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        self.normalize_decoder()
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T.clone())
        # Registered only for jumprelu, so checkpoints written by the other
        # variants still load under strict state_dict matching.
        if cfg.variant == "jumprelu":
            self.log_threshold = nn.Parameter(
                torch.full((d_sae,), math.log(cfg.jump_threshold)))
        # running count of steps since each feature last fired
        self.register_buffer("steps_since_fired", torch.zeros(d_sae), persistent=True)
        # input scaling, estimated once from data (keeps l1_coeff comparable across layers)
        self.register_buffer("input_scale", torch.ones(1), persistent=True)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def set_input_scale(self, acts: torch.Tensor) -> None:
        if self.cfg.normalize_input:
            self.input_scale.fill_(float(acts.float().norm(dim=-1).mean()) /
                                   (self.cfg.d_in ** 0.5) + 1e-8)

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x / self.input_scale - self.b_dec) @ self.W_enc + self.b_enc

    def _activate(self, pre: torch.Tensor) -> torch.Tensor:
        v = self.cfg.variant
        if v == "relu":
            return F.relu(pre)
        if v == "jumprelu":
            return _JumpReLU.apply(F.relu(pre), self.log_threshold,
                                   self.cfg.jump_bandwidth)
        if v == "topk":
            z = F.relu(pre)
            k = min(self.cfg.k, z.shape[-1])
            thresh = z.topk(k, dim=-1).values[..., -1:]
            return z * (z >= thresh)
        raise ValueError(f"unknown variant {v!r}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._activate(self._pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W_dec + self.b_dec) * self.input_scale

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x: torch.Tensor):
        # pre is computed once and shared with the sparsity term; recomputing it
        # for the jumprelu L0 penalty doubled the encoder cost of every step.
        pre = self._pre(x)
        z = self._activate(pre)
        xhat = self.decode(z)
        # Both terms must live in the same space. ``encode`` divides by
        # ``input_scale``, so z -- and therefore the L1 penalty -- is in
        # normalized units; measuring MSE on raw activations made the
        # reconstruction term scale as input_scale**2 against it. At the observed
        # input_scale of ~59 that is a ~3500x imbalance: an l1_coeff of 5e-4 acted
        # like 1.4e-7, sparsity was effectively switched off (L0 ~ 405 of 512),
        # and the coefficient meant something different at every layer -- the
        # exact opposite of what set_input_scale is documented to guarantee.
        mse = ((xhat - x) / self.input_scale).pow(2).sum(-1).mean()
        l0 = (z > 0).float().sum(-1).mean()
        if self.cfg.variant == "relu":
            sparse = self.cfg.l1_coeff * (z.abs() @ self.W_dec.norm(dim=-1)).mean()
        elif self.cfg.variant == "jumprelu":
            # explicit L0: the count of features above threshold, differentiated
            # through the same straight-through estimator as the activation
            sparse = self.cfg.l0_coeff * _Step.apply(
                F.relu(pre), self.log_threshold,
                self.cfg.jump_bandwidth).sum(-1).mean()
        else:
            sparse = x.new_zeros(())
        return mse + sparse, {"mse": mse.detach(), "sparsity": sparse.detach(),
                              "l0": l0.detach(), "z": z.detach()}

    # -- persistence ----------------------------------------------------------
    def save(self, directory: str, extra: dict | None = None) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(directory, "sae.pt"))
        meta = {"config": asdict(self.cfg)}
        if extra:
            meta.update(extra)
        with open(os.path.join(directory, "sae_config.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, directory: str, map_location="cpu") -> "SparseAutoencoder":
        with open(os.path.join(directory, "sae_config.json")) as fh:
            meta = json.load(fh)
        sae = cls(SAEConfig(**meta["config"]))
        sae.load_state_dict(torch.load(os.path.join(directory, "sae.pt"),
                                       map_location=map_location))
        return sae

    def to_saelens_state_dict(self) -> dict:
        """Key names matching the SAELens convention, for export/eval.

        The weights are FOLDED so that an importer applying the plain SAELens
        forward pass -- ``relu((x - b_dec) @ W_enc + b_enc) @ W_dec + b_dec`` --
        reproduces this class's activations.

        Exporting the raw parameters does not, because ``_pre`` divides the
        input by ``input_scale`` (30.60 for the shipped layer-3 dictionary) and
        ``decode`` multiplies it back, and neither the scale nor any vehicle for
        it was in the exported dict. Measured on 100k real layer-3 activations,
        an export of the raw parameters agreed with this codebase on the TopK
        active set for 0.39% of tokens (Jaccard 0.628) and inflated FVU from
        0.0449 to 0.0691.

        The fold, with ``s = input_scale``. This class computes

            z    = act((x/s - b_dec) @ W_enc + b_enc)
            xhat = (z @ W_dec + b_dec) * s

        and an importer computes ``act((x - B_dec) @ W_enc' + b_enc')`` then
        ``z @ W_dec' + B_dec``. Rewriting the encoder,

            (x/s - b_dec) @ W_enc = (x - s*b_dec) @ (W_enc/s)

        and the decoder, ``(z @ W_dec + b_dec)*s = z @ (W_dec*s) + s*b_dec``.
        Both match with

            W_enc' = W_enc / s,  b_enc' = b_enc,
            W_dec' = W_dec * s,  B_dec  = b_dec * s.

        Note the exported decoder rows are no longer unit-norm; they carry ``s``.
        ``input_scale`` is exported alongside so an importer can undo that if it
        assumes unit norms. A jumprelu threshold is exported too -- without it
        such a dictionary imports dense.
        """
        out = {"W_enc": self.W_enc.data / self.input_scale,
               "b_enc": self.b_enc.data,
               "W_dec": self.W_dec.data * self.input_scale,
               "b_dec": self.b_dec.data * self.input_scale,
               "input_scale": self.input_scale.clone()}
        if self.cfg.variant == "jumprelu":
            out["threshold"] = self.log_threshold.data.exp()
        return out
