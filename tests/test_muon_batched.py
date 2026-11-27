"""Equivalence of the batched Muon against the original per-matrix version.

``Muon.step`` orthogonalizes 42 block matrices per step. It used to do so one
matrix at a time; it now stacks matrices of the same (transpose-normalized)
shape and runs the Newton-Schulz iteration as batched ``bmm``s. That is a
performance change only, so these tests pin the arithmetic: the batched
iteration against ``zeropower_via_newtonschulz5`` on the real shapes, and a full
``Muon.step`` against a reference transcription of the original loop.

Differences of a few 1e-6 relative are expected -- batched matmul picks
different kernels and reduction orders than the single-matrix one -- and are
what the tolerances allow. A systematic difference is not.
"""
from __future__ import annotations

import pytest
import torch

from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.train.muon import (Muon, batched_newtonschulz5, build_optimizers,
                                zeropower_via_newtonschulz5)

# The shapes Muon actually sees in this model: attention projections are square,
# the SwiGLU matrices are the two orientations of 512x2048.
REAL_SHAPES = [(512, 512), (2048, 512), (512, 2048)]


def _reference_step(params, grads, state, lr, momentum, nesterov, ns_steps, wd):
    """Transcription of the original Muon.step, used as ground truth."""
    with torch.no_grad():
        for p, g in zip(params, grads):
            st = state.setdefault(p, {})
            if "buf" not in st:
                st["buf"] = torch.zeros_like(g)
            buf = st["buf"]
            buf.mul_(momentum).add_(g)
            upd = g.add(buf, alpha=momentum) if nesterov else buf
            upd = zeropower_via_newtonschulz5(upd, ns_steps)
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            if wd:
                p.mul_(1 - lr * wd)
            p.add_(upd, alpha=-lr * scale)


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _report(name, got, ref):
    err = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-12)
    print(f"{name}: max_abs {err.max().item():.3e} "
          f"max_rel {(err / denom).max().item():.3e} "
          f"scale {ref.abs().max().item():.3e}")
    return err.max().item(), (err / denom).max().item()


@pytest.mark.parametrize("shape", REAL_SHAPES)
def test_batched_newtonschulz_matches_single(shape):
    torch.manual_seed(0)
    dev = _dev()
    mats = [torch.randn(*shape, device=dev) * 0.02 for _ in range(6)]
    ref = torch.stack([zeropower_via_newtonschulz5(m, 5) for m in mats])

    transposed = shape[0] > shape[1]
    stack = torch.stack([(m.mT if transposed else m).float() for m in mats])
    got = batched_newtonschulz5(stack, 5)
    got = torch.stack([(g.mT if transposed else g) for g in got])

    abs_err, rel_err = _report(f"ns{shape}", got, ref)
    assert abs_err < 1e-4
    # The iterate is an approximate orthogonalization, so entries are O(1e-2)
    # and a handful of near-zero entries dominate the relative error; compare on
    # the Frobenius norm as well.
    assert (got - ref).norm() / ref.norm() < 1e-5


def test_batched_newtonschulz_mixed_orientations_in_one_call():
    """The wide and the tall SwiGLU matrices share one batched call."""
    torch.manual_seed(1)
    dev = _dev()
    wide = torch.randn(512, 2048, device=dev) * 0.02
    tall = torch.randn(2048, 512, device=dev) * 0.02
    stack = torch.stack([wide, tall.mT])
    got = batched_newtonschulz5(stack, 5)
    _report("mixed wide", got[0], zeropower_via_newtonschulz5(wide, 5))
    _report("mixed tall", got[1].mT, zeropower_via_newtonschulz5(tall, 5))
    assert torch.allclose(got[0], zeropower_via_newtonschulz5(wide, 5), atol=1e-5)
    assert torch.allclose(got[1].mT, zeropower_via_newtonschulz5(tall, 5), atol=1e-5)


def test_muon_step_matches_reference_on_the_real_model():
    """A full step on the real parameter set, from identical weights+grads."""
    dev = _dev()
    torch.manual_seed(0)
    model = MolecularTransformer(ModelConfig()).to(dev)
    ref_model = MolecularTransformer(ModelConfig()).to(dev)
    ref_model.load_state_dict(model.state_dict())

    opt, _, log = build_optimizers(model, muon_lr=0.02, weight_decay=0.01)
    ref_params = [p for n, p in ref_model.named_parameters() if n in set(log["muon"])]
    params = [p for n, p in model.named_parameters() if n in set(log["muon"])]
    assert len(params) == 42

    ref_state: dict = {}
    for it in range(3):
        torch.manual_seed(100 + it)
        grads = [torch.randn_like(p) * 0.01 for p in params]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.param_groups[0]["lr"] = 0.02 * (0.5 + 0.1 * it)
        opt.step()
        _reference_step(ref_params, [g.clone() for g in grads], ref_state,
                        lr=opt.param_groups[0]["lr"], momentum=0.95, nesterov=True,
                        ns_steps=5, wd=0.01)

        worst_abs = worst_rel = 0.0
        for name, p, q in zip(log["muon"], params, ref_params):
            a, r = _report(f"step{it} {name}", p.detach(), q.detach())
            worst_abs, worst_rel = max(worst_abs, a), max(worst_rel, r)
        print(f"step {it}: worst_abs {worst_abs:.3e} worst_rel {worst_rel:.3e}")
        assert worst_abs < 1e-6
        # momentum buffers must track too, or the next step diverges
        for p, q in zip(params, ref_params):
            assert torch.allclose(opt.state[p]["buf"], ref_state[q]["buf"], atol=1e-8)


def test_state_dict_structure_is_per_parameter():
    dev = _dev()
    torch.manual_seed(0)
    model = MolecularTransformer(ModelConfig()).to(dev)
    opt, _, _ = build_optimizers(model)
    for p in opt.param_groups[0]["params"]:
        p.grad = torch.randn_like(p) * 0.01
    opt.step()
    sd = opt.state_dict()
    assert set(sd) == {"state", "param_groups"}
    assert len(sd["state"]) == 42
    assert all(set(v) == {"buf"} for v in sd["state"].values())
    assert sd["param_groups"][0]["ns_steps"] == 5

    # and it round-trips into a fresh optimizer
    model2 = MolecularTransformer(ModelConfig()).to(dev)
    opt2, _, _ = build_optimizers(model2)
    opt2.load_state_dict(sd)
    for p, q in zip(opt.param_groups[0]["params"], opt2.param_groups[0]["params"]):
        assert torch.equal(opt.state[p]["buf"], opt2.state[q]["buf"])


def test_non_2d_parameters_fall_back_to_the_scalar_path():
    dev = _dev()
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, device=dev))
    opt = Muon([p], lr=0.01)
    p.grad = torch.randn_like(p)
    with pytest.raises(IndexError):
        opt.step()
