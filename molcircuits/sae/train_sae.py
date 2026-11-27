"""SAE training on cached residual-stream activations."""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..model.hooks import HookManager, resid_post
from .sae import SAEConfig, SparseAutoencoder


@dataclass
class SAETrainConfig:
    n_activations: int = 50_000_000
    batch_size: int = 2048
    lr: float = 3e-4
    warmup: int = 500
    grad_clip: float = 1.0
    resample_dead_every: int = 0     # 0 disables; otherwise steps between resamples
    dead_threshold_steps: int = 10_000
    tf32: bool = True                # tensor-core matmuls during SAE training
    seed: int = 0


def _cache_plan(n_target: int, d_in: int, device: str,
                dtype: torch.dtype) -> tuple[str, torch.dtype]:
    """Keep the activation cache in VRAM when it fits, otherwise host RAM.

    Residency is what sets the speed of this stage. On the host, every training
    step random-gathers 2048 rows scattered across a ~100 GB array and copies
    them over PCIe -- ~100 GB of transfer per dictionary, 18 dictionaries deep,
    with the GPU sitting at ~16% memory utilization throughout. In VRAM the
    gather is a device-side index and the copy disappears.

    float16 rather than bfloat16 on purpose: measured FVU at these layers is
    ~1e-4, and bfloat16's 8-bit mantissa contributes rounding error of the same
    order, which would corrupt the metric being reported. float16's 11-bit
    mantissa puts its contribution ~2 orders below the signal, and the activation
    range here (|x| ~ 50) is nowhere near its 65504 ceiling.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "cpu", torch.float32
    free, _ = torch.cuda.mem_get_info()
    need = n_target * d_in * torch.finfo(dtype).bits // 8
    # leave headroom for the model, the SAE and its optimizer state
    return (device, dtype) if need < free - (8 << 30) else ("cpu", torch.float32)


@torch.no_grad()
def collect_activations(model, loader: DataLoader, layer: int, n_target: int,
                        device: str = "cpu", d_in: int = 512,
                        store_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Cache residual-stream activations at one layer (token-level, unpadded).

    The cache is preallocated and filled in place. Accumulating a list and
    ``torch.cat``-ing it holds both copies at once, which at the default 50M
    activations is 102 GB of float32 twice over -- more host RAM than the machine
    has, and an OOM that only lands after the model has already been trained.

    The forward stays in float32 (no autocast): the bottleneck here is cache
    residency and transfer, not matmul throughput, so there is nothing to buy by
    trading away precision in the activations being cached.
    """
    model.eval()
    store_dev, store_dt = _cache_plan(n_target, d_in, device, store_dtype)
    out = torch.empty((n_target, d_in), dtype=store_dt, device=store_dev)
    got = 0
    for batch in loader:
        x = batch[0].to(device)
        mask = batch[1].to(device) if len(batch) > 2 else None
        h = HookManager().capture(resid_post(layer))
        model(x, hooks=h)
        a = h[resid_post(layer)]
        a = a[mask] if mask is not None else a.reshape(-1, d_in)
        a = a.reshape(-1, d_in)
        take = min(a.shape[0], n_target - got)
        out[got:got + take] = a[:take].to(store_dev, store_dt)
        got += take
        if got >= n_target:
            break
    return out[:got]


@torch.no_grad()
def collect_activations_multi(model, loader: DataLoader, layers: list[int],
                              n_target: int, device: str = "cpu", d_in: int = 512,
                              store_dtype: torch.dtype = torch.float16) -> dict:
    """Cache several layers' residual streams in ONE pass over the corpus.

    The captured VALUES are the ones a dedicated pass computes: a capture hook
    records ``resid_post(l)`` without modifying the stream, so each layer's
    activations are the same whether one layer is hooked or six, and a staged
    cache is element-wise identical to a direct VRAM collection of the same
    layer (the float16 cast runs on the GPU with the same kernel either way).
    What changes is the corpus cost -- one forward pass instead of
    ``len(layers)``.

    CAUTION -- this is NOT drop-in equivalent to the per-layer pipeline flow.
    There, ``_cache_plan`` is consulted while the previous layer's ~51 GB cache
    still counts against free VRAM (``torch.cuda.mem_get_info`` cannot see
    blocks parked in the caching allocator), so only the first collected layer
    gets a float16 VRAM cache and later layers fall back to float32 host
    caches. This function plans once, with VRAM free, and caches EVERY layer in
    float16. Uniform precision across layers -- but a different training input
    for any layer the per-layer flow would have kept in float32, so
    dictionaries trained from these caches differ from the committed reference
    artifacts at those layers (metrics agree to ~1e-4; weights land in a
    different basin).

    Residency: only one ~51 GB cache fits in VRAM next to the model, so the
    first layer in ``layers`` (the first one trained) keeps the VRAM cache and
    the rest are staged in pinned host RAM. The caller uploads each staged
    cache wholesale before training; on this Grace-Hopper host that is a
    ~320 GB/s NVLink-C2C copy (~0.2 s for 51 GB), nothing like the per-step
    random-gather pattern ``_cache_plan``'s docstring warns about.
    """
    model.eval()
    if not layers:
        return {}
    store_dev, store_dt = _cache_plan(n_target, d_in, device, store_dtype)
    pin = device.startswith("cuda") and torch.cuda.is_available()
    out = {}
    for j, layer in enumerate(layers):
        if j == 0:
            out[layer] = torch.empty((n_target, d_in), dtype=store_dt,
                                     device=store_dev)
        else:
            out[layer] = torch.empty((n_target, d_in), dtype=store_dt,
                                     pin_memory=pin)
    names = [resid_post(layer) for layer in layers]
    got = 0
    for batch in loader:
        x = batch[0].to(device)
        mask = batch[1].to(device) if len(batch) > 2 else None
        h = HookManager().capture(*names)
        model(x, hooks=h)
        take = None
        for layer, name in zip(layers, names):
            a = h[name]
            a = a[mask] if mask is not None else a.reshape(-1, d_in)
            a = a.reshape(-1, d_in)
            take = min(a.shape[0], n_target - got)
            dst = out[layer]
            if dst.device == a.device:
                dst[got:got + take] = a[:take].to(store_dt)
            else:
                # cast on the GPU (identical kernel to the VRAM path), then a
                # contiguous async D2H copy into the pinned staging buffer
                dst[got:got + take].copy_(a[:take].to(store_dt),
                                          non_blocking=pin)
        got += take
        if got >= n_target:
            break
    if pin:
        torch.cuda.synchronize()
    return {layer: t[:got] for layer, t in out.items()}


@torch.no_grad()
def resample_dead(sae, acts: torch.Tensor, dead: torch.Tensor, opt,
                  batch: int = 8192, device: str = "cpu") -> int:
    """Re-initialise features that have stopped firing, pointing them at inputs
    the dictionary currently reconstructs worst.

    Follows Bricken et al. (2023): a dead feature is redirected at a
    high-residual input rather than re-randomised, so it starts somewhere the
    dictionary is actually failing. The decoder row becomes that (unit-norm)
    direction, the encoder column copies it scaled to a fraction of the surviving
    encoder norm -- small enough not to immediately dominate -- and the Adam
    moments for the touched entries are cleared, since moments accumulated for a
    feature that no longer exists would otherwise drag the fresh one.

    ``SAETrainConfig.resample_dead_every`` previously appeared in the config and
    was never read anywhere in the package, so the knob silently did nothing.
    """
    n = int(dead.sum())
    if n == 0:
        return 0

    idx = torch.randint(0, len(acts), (min(batch, len(acts)),), device=acts.device)
    x = acts[idx].to(device, torch.float32)
    xhat, _ = sae(x)
    resid = (x - xhat).pow(2).sum(-1)
    # sample proportional to squared residual, so the worst-explained inputs win
    pick = torch.multinomial(resid / resid.sum().clamp_min(1e-12), n,
                             replacement=n > len(resid))
    dirs = x[pick] - sae.b_dec * sae.input_scale
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    alive = ~dead
    enc_norm = (sae.W_enc[:, alive].norm(dim=0).mean() * 0.2 if alive.any()
                else sae.W_enc.norm(dim=0).mean())

    sae.W_dec[dead] = dirs.to(sae.W_dec.dtype)
    sae.W_enc[:, dead] = (dirs * enc_norm).T.to(sae.W_enc.dtype)
    sae.b_enc[dead] = 0.0
    if sae.cfg.variant == "jumprelu":
        sae.log_threshold[dead] = math.log(sae.cfg.jump_threshold)

    for p, mask, dim in ((sae.W_enc, dead, 1), (sae.W_dec, dead, 0),
                         (sae.b_enc, dead, 0)):
        st = opt.state.get(p)
        if st:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    if dim == 0:
                        st[key][mask] = 0.0
                    else:
                        st[key][:, mask] = 0.0
    sae.steps_since_fired[dead] = 0
    return n


@contextmanager
def _tf32(enabled: bool):
    """Tensor-core matmuls for SAE training only.

    The step is compute-bound (~103 GFLOP at 2048x512x8192), so fp32 matmuls
    leave the tensor cores idle; TF32 is measured at 1.49x here for a 0.03%
    change in final MSE. It is deliberately *not* enabled around
    ``collect_activations``: TF32's 10-bit mantissa carries ~5e-4 relative error,
    which is larger than the ~1e-4 FVU these dictionaries are measured at, so the
    cached activations stay in strict fp32.
    """
    if not enabled:
        yield
        return
    mm, cud = (torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = mm
        torch.backends.cudnn.allow_tf32 = cud


def train_sae(acts: torch.Tensor, cfg: SAEConfig, tcfg: SAETrainConfig,
              device: str = "cpu", out_dir: str | None = None) -> tuple:
    with _tf32(tcfg.tf32 and device.startswith("cuda")):
        return _train_sae(acts, cfg, tcfg, device, out_dir)


def _train_sae(acts: torch.Tensor, cfg: SAEConfig, tcfg: SAETrainConfig,
               device: str = "cpu", out_dir: str | None = None) -> tuple:
    torch.manual_seed(tcfg.seed)
    sae = SparseAutoencoder(cfg).to(device)
    sae.set_input_scale(acts[: min(len(acts), 100_000)].to(device, torch.float32))

    # ``acts.is_cuda`` gates the graph deliberately: a host-resident cache
    # (which `_cache_plan` chooses for layers >= 1 in the production pipeline)
    # gathers each batch on the CPU, so the step is not capturable and runs on
    # the identical eager path the reference artifacts were trained with.
    use_graph = (device.startswith("cuda") and torch.cuda.is_available()
                 and acts.is_cuda and len(acts) >= tcfg.batch_size
                 and not os.environ.get("MOLCIRCUITS_SAE_NO_GRAPH"))
    # ``capturable`` is a permission flag for capturing step() in a CUDA graph;
    # the FUSED implementation runs the identical kernel either way (verified
    # bitwise against capturable=False), it only skips the health check.
    opt = torch.optim.Adam(sae.parameters(), lr=tcfg.lr, betas=(0.9, 0.999),
                           fused=device.startswith("cuda"), capturable=use_graph)
    n_steps = max(1, min(tcfg.n_activations, len(acts)) // tcfg.batch_size)
    hist, resampled = [], []
    # The permutation must live wherever the cache does; indexing a CUDA tensor
    # with CPU indices forces a synchronizing round trip on every step.
    perm = torch.randperm(len(acts), device=acts.device)

    def step_math(x):
        """One optimizer step, from loss to bookkeeping.

        Shared verbatim between the eager path and CUDA-graph capture, so both
        execute the identical kernel sequence -- graph replay of these kernels
        is bitwise-identical to running them eagerly (verified: a graphed run
        reproduces an eager run's checkpoint exactly). Gradients are expected
        to be None on entry (``zero_grad(set_to_none=True)``), so backward
        writes fresh grads rather than accumulating; under capture those
        buffers come from the graph's private pool and are rewritten on every
        replay.
        """
        loss, parts = sae.loss(x)
        loss.backward()
        # remove the gradient component along each decoder row before the step,
        # so the unit-norm constraint is not fighting the optimizer
        with torch.no_grad():
            proj = (sae.W_dec.grad * sae.W_dec).sum(-1, keepdim=True) * sae.W_dec
            sae.W_dec.grad -= proj
        torch.nn.utils.clip_grad_norm_(sae.parameters(), tcfg.grad_clip)
        opt.step()
        sae.normalize_decoder()
        with torch.no_grad():
            # Reuse the code from loss() rather than running a second full
            # encoder pass per step just to update a diagnostic counter. This
            # measures firing with the weights from before opt.step() instead of
            # after; for a boolean "did this feature fire", one Adam step at
            # lr 3e-4 does not change the answer in any way that matters.
            # (``masked_fill_`` is the kernel ``counter[fired] = 0`` lowers to,
            # written explicitly because index_put via __setitem__ is not
            # allowed during CUDA-graph capture in all torch versions.)
            fired = (parts["z"] > 0).any(0)
            sae.steps_since_fired += 1
            sae.steps_since_fired.masked_fill_(fired, 0)
        return loss, parts

    # -- CUDA-graph replay of the steady-state step ---------------------------
    # The step is ~50 kernels of a few tens of microseconds each, so eagerly it
    # is host-dispatch-bound (measured 2.3 ms of CPU dispatch against ~0.8 ms
    # of GPU work per step). Everything about the step is static once the LR
    # warmup has finished: fixed shapes, no data-dependent control flow, and
    # the fused Adam keeps its step count on-device, so the whole step is
    # captured once and replayed with a single launch. Per replay the host does
    # one D2D index copy; the LR is baked into the capture, which is exact
    # because the schedule is constant after warmup. The first
    # ``max(warmup, 3)`` steps run eagerly: the LR still changes there, and
    # they double as the capture warmup that initializes optimizer state and
    # cuBLAS workspaces.
    eager_steps = min(n_steps, max(tcfg.warmup, 3)) if use_graph else n_steps
    if use_graph and n_steps - eager_steps < 64:
        eager_steps = n_steps           # too short for capture to pay for itself
    graph, static_idx = None, None

    # The eager warmup steps run on the SAME side stream the capture will use:
    # the autograd engine caches the stream each AccumulateGrad node last ran
    # on, and a warmup on the default stream leaves stale default-stream
    # references that invalidate the capture. Streams carry no numerics -- the
    # kernel sequence is unchanged.
    capture_stream = torch.cuda.Stream() if use_graph else None
    if capture_stream is not None:
        capture_stream.wait_stream(torch.cuda.current_stream())

    def _stream_ctx(active: bool):
        if capture_stream is not None and active:
            return torch.cuda.stream(capture_stream)
        return nullcontext()

    for step in range(n_steps):
        in_eager = step < eager_steps
        with _stream_ctx(in_eager):
            idx = perm[(step * tcfg.batch_size) % len(acts):][: tcfg.batch_size]
            if len(idx) < tcfg.batch_size:
                perm = torch.randperm(len(acts), device=acts.device)
                idx = perm[: tcfg.batch_size]

            for g in opt.param_groups:
                g["lr"] = tcfg.lr * min(1.0, (step + 1) / max(tcfg.warmup, 1))

            if in_eager:
                x = acts[idx].to(device, torch.float32)
                opt.zero_grad(set_to_none=True)
                loss, parts = step_math(x)
                loss_t, mse_t, l0_t = loss, parts["mse"], parts["l0"]

        if not in_eager:
            if graph is None:
                # order the capture (and every later default-stream replay)
                # after the eager-phase work queued on the side stream
                torch.cuda.current_stream().wait_stream(capture_stream)
                static_idx = torch.zeros_like(idx)
                opt.zero_grad(set_to_none=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    static_x = acts[static_idx].to(device, torch.float32)
                    loss, parts = step_math(static_x)
                # capture records the kernels without executing them; the real
                # step happens at the replay below
            static_idx.copy_(idx)
            graph.replay()
            loss_t, mse_t, l0_t = loss, parts["mse"], parts["l0"]

        if (tcfg.resample_dead_every and step
                and step % tcfg.resample_dead_every == 0
                and step < n_steps - tcfg.resample_dead_every):
            with _stream_ctx(in_eager):
                dead = sae.steps_since_fired > tcfg.dead_threshold_steps
                n_res = resample_dead(sae, acts, dead, opt, device=device)
            if n_res:
                resampled.append({"step": step, "n": n_res})

        if step % max(1, n_steps // 20) == 0 or step == n_steps - 1:
            # loss_t/mse_t/l0_t are the graph's static output tensors during
            # replay; float() syncs, which is why this only happens ~20 times.
            with _stream_ctx(in_eager):
                hist.append({"step": step, "loss": float(loss_t.detach()),
                             "mse": float(mse_t), "l0": float(l0_t),
                             "dead_frac": float((sae.steps_since_fired >
                                                 tcfg.dead_threshold_steps).float().mean())})

    if capture_stream is not None:
        # order any following default-stream work (saving, evaluation) after
        # an eager-only run that never reached the capture point
        torch.cuda.current_stream().wait_stream(capture_stream)

    summary = {"final_l0": hist[-1]["l0"] if hist else None,
               "final_mse": hist[-1]["mse"] if hist else None,
               "dead_feature_fraction": hist[-1]["dead_frac"] if hist else None,
               "n_steps": n_steps, "history": hist,
               "resample_dead_every": tcfg.resample_dead_every,
               "resampled": resampled,
               "n_resampled_total": sum(r["n"] for r in resampled)}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        sae.save(out_dir, extra={"train": summary, "train_config": tcfg.__dict__})
        with open(os.path.join(out_dir, "train_metrics.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
    return sae, summary
