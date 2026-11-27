"""Exact causal-prefix reuse for repeated residual interventions at one model site.

A molecule can have several edited decisions. Recompute the entire span from
the first edit to the last scored position, using the unedited keys and values
for earlier tokens. All decisions in a molecule receive the edit together.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from molcircuits.model.hooks import HookManager


@dataclass
class SuffixBatch:
    activations: torch.Tensor
    prefix_kv: list
    cos: torch.Tensor
    sin: torch.Tensor
    mask: torch.Tensor
    rows: torch.Tensor
    positions: torch.Tensor


def _rotate(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CachedIntervention:
    def __init__(self, model, tokens, positions, site, layer, batch_size):
        self.model = model
        self.layer = layer
        self.site = site
        self.batches = []
        per_molecule = {}
        for i, (row, time) in enumerate(positions.tolist()):
            per_molecule.setdefault(row, []).append((time, i))
        starts = {row: min(t for t, _ in points) for row, points in per_molecule.items()}
        ends = {row: max(t for t, _ in points) + 1 for row, points in per_molecule.items()}
        # Sorting keeps the amount of padding in the changed span small. Bound
        # model input memory during the one-time prefix capture as well.
        rows = sorted(per_molecule, key=lambda row: (ends[row] - starts[row], starts[row]))
        device = tokens.device
        all_sites, all_logits, all_ids = [], [], []
        capture_name = f"blocks.{layer}.hook_resid_{site}"
        for i in range(0, len(rows), batch_size):
            members = rows[i : i + batch_size]
            width = max(ends[row] for row in members)
            span = max(ends[row] - starts[row] for row in members)
            prefix = max(starts[row] for row in members)
            row_indices = torch.tensor(members, device=device)
            offsets = torch.tensor([starts[row] for row in members], device=device)
            suffix_time = torch.arange(span, device=device)
            abs_time = (offsets[:, None] + suffix_time).clamp_max(width - 1)
            in_batch = torch.arange(len(members), device=device)
            score_rows, score_time, score_ids = zip(
                *(
                    (j, t - starts[row], idx)
                    for j, row in enumerate(members)
                    for t, idx in per_molecule[row]
                )
            )
            br, bt, ix = (
                torch.tensor(v, device=device) for v in (score_rows, score_time, score_ids)
            )
            hook = HookManager().capture(capture_name)
            kv = [{} for _ in model.blocks]
            full = model(tokens[row_indices, :width], hooks=hook, cache=kv)
            activations = hook[capture_name][in_batch[:, None], abs_time]
            cos, sin = model._rope_for(device, activations.dtype)
            prefix_valid = torch.arange(prefix, device=device)[None, :] < offsets[:, None]
            suffix_valid = suffix_time[None, :] <= suffix_time[:, None]
            mask = torch.cat(
                (
                    prefix_valid[:, None, :].expand(-1, span, -1),
                    suffix_valid[None, :, :].expand(len(members), -1, -1),
                ),
                dim=2,
            )[:, None]
            cached = [
                (
                    cache["k"][:, :, :prefix].clone(memory_format=torch.contiguous_format),
                    cache["v"][:, :, :prefix].clone(memory_format=torch.contiguous_format),
                )
                for cache in kv[layer + 1 :]
            ]
            self.batches.append(
                SuffixBatch(
                    activations,
                    cached,
                    cos[abs_time][:, None],
                    sin[abs_time][:, None],
                    mask,
                    br,
                    bt,
                )
            )
            all_sites.append(activations[br, bt])
            all_logits.append(full[br, abs_time[br, bt]])
            all_ids.append(ix)
        self.order = torch.cat(all_ids).argsort()
        self.edit_activations = torch.cat(all_sites)[self.order].float()
        self.base_logits = torch.cat(all_logits)[self.order].float()

    def logits(self, delta):
        weight = (
            self.model.embed.weight if self.model.lm_head is None else self.model.lm_head.weight
        )
        if self.layer == len(self.model.blocks) - 1:
            x = self.edit_activations + delta
            if self.site == "mid":
                block = self.model.blocks[self.layer]
                x = x + block.mlp(block.ln2(x))
            return self.model.ln_f(x) @ weight.T

        parts = []
        for batch in self.batches:
            x = batch.activations.clone()
            x[batch.rows, batch.positions] += delta.to(x.dtype)
            if self.site == "mid":
                block = self.model.blocks[self.layer]
                x = x + block.mlp(block.ln2(x))
            n, t, width = x.shape
            for block, (k0, v0) in zip(self.model.blocks[self.layer + 1 :], batch.prefix_kv):
                attn = block.attn
                normalized = block.ln1(x)
                shape = (n, t, attn.cfg.n_heads, attn.cfg.d_head)
                q = attn.W_Q(normalized).reshape(shape).transpose(1, 2)
                k = attn.W_K(normalized).reshape(shape).transpose(1, 2)
                v = attn.W_V(normalized).reshape(shape).transpose(1, 2)
                q = _rotate(q, batch.cos, batch.sin)
                k = _rotate(k, batch.cos, batch.sin)
                z = F.scaled_dot_product_attention(
                    q, torch.cat((k0, k), dim=2), torch.cat((v0, v), dim=2), attn_mask=batch.mask
                )
                m = attn.head_mask.view(1, -1, 1, 1)
                z = z * m + attn.head_bias.view(1, shape[2], 1, shape[3]) * (1 - m)
                z = z.transpose(1, 2).reshape(n, t, width)
                if block.layer == len(self.model.blocks) - 1:
                    # After the last attention operation, each position is independent.
                    z = z[batch.rows, batch.positions]
                    x = x[batch.rows, batch.positions]
                x = x + attn.W_O(z)
                x = x + block.mlp(block.ln2(x))
            parts.append(self.model.ln_f(x) @ weight.T)
        return torch.cat(parts)[self.order].float()
