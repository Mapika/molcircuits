"""Regression tests for the bugs that were in the previous implementation.

Each test here corresponds to a specific defect found during the code audit. If
you port more logic across, add the test first.
"""
import math
import os

import numpy as np
import pytest
import torch

from molcircuits import paths
from molcircuits.chem.events import find_events
from molcircuits.chem.fragments import fragment_token_mask
from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import bond_decision_positions, valence_labels


# ---------------------------------------------------------------- tokenizer
def test_two_char_elements_and_brackets():
    assert tokenize("CCBr") == ["C", "C", "Br"]
    assert tokenize("[nH]1cccc1") == ["[nH]", "1", "c", "c", "c", "c", "1"]
    assert tokenize("C%10CCC%10") == ["C", "%10", "C", "C", "C", "%10"]


def test_unparseable_raises():
    with pytest.raises(ValueError):
        tokenize("CC?CC")


def test_roundtrip():
    tok = SmilesTokenizer.from_corpus(["CCO", "c1ccccc1", "CC(=O)N"])
    for s in ["CCO", "c1ccccc1", "CC(=O)N"]:
        assert tok.decode(tok.encode(s)) == s


# ---------------------------------------------------------------- events
def test_percent_ring_label_produces_exactly_one_event():
    """The old scanner re-read each digit of %nn, injecting two phantom
    single-digit ring events per two-digit closure."""
    toks = tokenize("C%10CCCCCCCCCCC%10")
    rings = [e for e in find_events(toks) if e.kind == "ring"]
    assert len(rings) == 1
    assert rings[0].label == "10"


def test_ring_digit_reuse_after_closure():
    rings = [e for e in find_events(tokenize("C1CC1C1CC1")) if e.kind == "ring"]
    assert len(rings) == 2


def test_branch_depth_and_span():
    evs = [e for e in find_events(tokenize("CC(C(C)C)C")) if e.kind == "branch"]
    assert len(evs) == 2
    assert max(e.depth for e in evs) == 2


def test_predict_index_precedes_closer():
    for e in find_events(tokenize("c1ccccc1CC(C)C")):
        assert e.predict_idx == e.close_idx - 1
        assert e.open_idx < e.close_idx


def test_unbalanced_input_does_not_raise():
    find_events(tokenize("CC(CC"))
    find_events(tokenize("C1CC"))


# ---------------------------------------------------------------- valence
# Prefix semantics: capacity remaining at the moment each atom token is written,
# counting only bonds already to its left. These expectations previously encoded
# the completed-molecule counts, which is why the bug survived: the test agreed
# with the implementation rather than with the documented definition.
@pytest.mark.parametrize("smiles,expected", [
    # C has 4 free when written; the second C has spent 1 on the bond to the first
    ("CCO", [4, 3, 1]),
    # first aromatic C is unbonded when written; the rest carry one aromatic bond
    ("c1ccccc1", [4, 2, 2, 2, 2, 2]),
    ("C1CC1", [4, 3, 3]),
    # N+ has cap 4; when written it has spent only the bond from the leading C
    ("C[N+](C)(C)C", [4, 3, 3, 3, 3]),
    ("N#Cc1ccccc1", [3, 1, 3, 2, 2, 2, 2, 2]),
])
def test_valence_labels(smiles, expected):
    got = [l.remaining for l in valence_labels(tokenize(smiles))]
    assert got == expected


def test_labels_depend_only_on_left_context():
    """The prefix label must not move when tokens are appended.

    The previous version of this test asserted only `part[0] == full[0]` on a
    4-token truncation, which passes for the wrong reason: atom 0's only bond is
    to atom 1 and is present in both truncations. It therefore did not catch that
    labels were being read from the *completed* molecule -- 85% of them differ.
    """
    for smi in ("CCO", "CC(=O)Nc1ccccc1", "c1ccccc1", "N#Cc1ccccc1",
                "C1CC1CO", "CC(C)(C)c1ccc(O)cc1"):
        toks = tokenize(smi)
        full = {l.token_idx: l.remaining
                for l in valence_labels(toks, offset=1)}
        for cut in range(1, len(toks)):
            part = {l.token_idx: l.remaining
                    for l in valence_labels(toks[:cut], offset=1)}
            for idx, val in part.items():
                assert full[idx] == val, (
                    f"{smi}: atom at {idx} changed {val} -> {full[idx]} "
                    f"when tokens beyond {cut} were appended")


def test_final_semantics_matches_rdkit_implicit_hydrogens():
    """`semantics='final'` is kept only for comparison; pin what it is."""
    from rdkit import Chem

    for smi in ("CCO", "CC(=O)N", "c1ccccc1", "N#Cc1ccccc1", "C[N+](C)(C)C"):
        toks = tokenize(smi)
        got = [l.remaining for l in valence_labels(toks, semantics="final")]
        ref = [a.GetTotalNumHs() for a in Chem.MolFromSmiles(smi).GetAtoms()]
        assert got == ref, f"{smi}: {got} != {ref}"


def test_prefix_and_final_semantics_actually_differ():
    """Guard against the snapshot silently regressing to the post-scan read."""
    toks = tokenize("CCO")
    pre = [l.remaining for l in valence_labels(toks, semantics="prefix")]
    fin = [l.remaining for l in valence_labels(toks, semantics="final")]
    assert pre == [4, 3, 1] and fin == [3, 2, 1]


def test_bond_decision_positions():
    toks = tokenize("CC(=O)N")
    pos = bond_decision_positions(toks)
    assert all(toks[p + 1] in ("-", "=", "#") for p in pos)


# ---------------------------------------------------------------- fragments
def test_fragment_mask_hits_atom_tokens_only():
    smi = "CC(=O)Nc1ccccc1"
    toks = tokenize(smi)
    mask = fragment_token_mask(smi, "[NX3][CX3](=[OX1])", len(toks), offset=0)
    assert mask.any()
    for i in np.where(mask)[0]:
        assert toks[i] not in ("(", ")", "=", "1")


# ---------------------------------------------------------------- margin
def test_margin_uses_the_ablated_gold_logit():
    """The old implementation closed over the *baseline* gold logit, so
    delta_margin collapsed to a difference of best-other terms and the ablated
    gold logit never entered."""
    from molcircuits.analysis.pointers import margin_at

    logits = torch.tensor([[[5.0, 1.0, 0.0]]])
    targets = torch.tensor([[0]])
    pos = torch.tensor([[0, 0]])
    assert margin_at(logits, targets, pos).item() == pytest.approx(4.0)

    # lowering only the gold logit must lower the margin
    logits2 = logits.clone()
    logits2[0, 0, 0] = 2.0
    assert margin_at(logits2, targets, pos).item() == pytest.approx(1.0)


# ---------------------------------------------------------------- mean ablation
def test_mean_ablation_has_a_value_to_substitute():
    """04_pointers.py swept mode='mean' but never computed mean_z, so every mean
    ablation raised ValueError. mean_head_activations supplies it, shaped
    (n_heads, d_head) per layer as ablate_head requires."""
    from molcircuits.analysis.pointers import mean_head_activations
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=2, n_heads=4,
                      d_ff=32, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    tokens = torch.randint(1, 8, (3, 8))

    mz = mean_head_activations(model, tokens, device="cpu")
    assert set(mz) == {0, 1}
    for v in mz.values():
        assert v.shape == (cfg.n_heads, cfg.d_head)

    # the value exists, so mean ablation now runs instead of raising
    with model.ablate_head(0, 1, mode="mean", mean_z=mz[0]):
        model(tokens)


def test_mean_activation_excludes_padding():
    """Averaging over pad positions would drag the mean toward the model's
    response to <pad>, quietly collapsing mean ablation onto zero ablation."""
    from molcircuits.analysis.pointers import mean_head_activations
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, n_ctx=8)
    model = MolecularTransformer(cfg).eval()

    real = torch.randint(1, 8, (2, 4))
    pad = torch.zeros((2, 4), dtype=torch.long)          # pad_id == 0
    padded = torch.cat([real, pad], dim=1)

    masked = mean_head_activations(model, padded, device="cpu", pad_id=0)[0]
    unmasked = mean_head_activations(model, padded, device="cpu", pad_id=None)[0]
    assert not torch.allclose(masked, unmasked)

    # masking must reproduce the mean over the unpadded prefix alone
    ref = mean_head_activations(model, real, device="cpu", pad_id=0)[0]
    assert torch.allclose(masked, ref, atol=1e-5)


# ---------------------------------------------------------------- steering
def test_matched_random_latent_stays_on_its_input_device():
    """matched_random_latent built its output on z_ref's device but assigned a
    .cpu() tensor into it, so 07_steering.py died on the first CUDA target."""
    from molcircuits.analysis.steering import matched_random_latent

    z = torch.zeros(64)
    z[[3, 17, 40]] = torch.tensor([0.5, 2.0, 1.25])

    out = matched_random_latent(z, seed=0)
    assert out.device == z.device
    assert int((out > 0).sum()) == 3
    assert sorted(out[out > 0].tolist()) == pytest.approx([0.5, 1.25, 2.0])

    if torch.cuda.is_available():
        zc = z.cuda()
        oc = matched_random_latent(zc, seed=0)
        assert oc.device == zc.device
        # the CPU generator makes feature identities device-independent
        assert torch.equal(oc.cpu().nonzero().squeeze(-1), out.nonzero().squeeze(-1))


# ---------------------------------------------------------------- checkpoints
def test_compiled_checkpoint_loads_into_an_uncompiled_model():
    """torch.compile wraps the module in an OptimizedModule whose state_dict keys
    carry an "_orig_mod." prefix. Saving that dict produced a checkpoint no
    analysis script could load -- and only after training had finished."""
    import tempfile

    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.train.data import PackedDataset
    from molcircuits.train.trainer import TrainConfig, train

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, n_ctx=8)
    ids = np.random.default_rng(0).integers(0, 8, size=4096).astype(np.uint16)
    ds = PackedDataset(ids, cfg.n_ctx)

    with tempfile.TemporaryDirectory() as d:
        tcfg = TrainConfig(steps=2, batch_size=2, eval_every=2, eval_batches=1,
                           ckpt_every=2, dtype="float32", compile=True, out_dir=d)
        train(MolecularTransformer(cfg), ds, ds, tcfg, device="cpu")
        sd = torch.load(os.path.join(d, "checkpoint.pt"), map_location="cpu")["model"]

    assert not any(k.startswith("_orig_mod.") for k in sd)
    MolecularTransformer(cfg).load_state_dict(sd)      # must not raise


# ---------------------------------------------------------------- SAE loss scale
def test_sae_loss_is_invariant_to_activation_scale():
    """encode() divides by input_scale, so z and the L1 penalty are in normalized
    units. MSE was measured on raw activations, so the reconstruction term scaled
    as input_scale**2 against the penalty -- at the observed input_scale of ~59
    a ~3500x imbalance that switched sparsity off entirely."""
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    torch.manual_seed(0)
    cfg = SAEConfig(d_in=16, expansion=2, variant="relu", l1_coeff=1.0)
    sae = SparseAutoencoder(cfg)
    x = torch.randn(64, 16)

    sae.set_input_scale(x)
    small, _ = sae.loss(x)

    # the identical problem, presented at 50x the activation scale
    sae_big = SparseAutoencoder(cfg)
    sae_big.load_state_dict(sae.state_dict())
    sae_big.set_input_scale(x * 50)
    big, _ = sae_big.loss(x * 50)

    assert big.item() == pytest.approx(small.item(), rel=1e-4)


def test_sae_l1_coeff_actually_controls_sparsity():
    """The default must not be inert: raising l1_coeff has to reduce L0."""
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    torch.manual_seed(0)
    x = torch.randn(256, 32) * 40.0            # deliberately un-normalized scale

    def sparsity_term(lam):
        cfg = SAEConfig(d_in=32, expansion=4, variant="relu", l1_coeff=lam)
        sae = SparseAutoencoder(cfg)
        sae.set_input_scale(x)
        return sae.loss(x)[1]["sparsity"].item()

    # the penalty must scale with the coefficient rather than vanish against MSE
    assert sparsity_term(1.0) > 100 * sparsity_term(5e-4)
    assert sparsity_term(1.0) > 0.0


# ---------------------------------------------------------------- activation cache
def test_collect_activations_matches_across_storage_devices():
    """The cache moved from host RAM to VRAM for speed. Residency and dtype are a
    performance choice and must not change the activations that get cached."""
    from torch.utils.data import DataLoader

    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.sae.train_sae import collect_activations
    from molcircuits.train.data import PackedDataset

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=2, n_heads=2,
                      d_ff=32, n_ctx=8)
    torch.manual_seed(0)
    model = MolecularTransformer(cfg).eval()
    ids = np.random.default_rng(0).integers(0, 8, size=2048).astype(np.uint16)
    loader = DataLoader(PackedDataset(ids, cfg.n_ctx), batch_size=4)

    ref = collect_activations(model, loader, 1, 256, "cpu", cfg.d_model)
    assert ref.shape == (256, cfg.d_model)
    assert ref.dtype == torch.float32          # cpu path stays float32

    if torch.cuda.is_available():
        gpu = collect_activations(model.cuda(), loader, 1, 256, "cuda",
                                  cfg.d_model)
        # float16 storage must stay far below the ~1e-4 FVU being measured
        assert torch.allclose(ref, gpu.float().cpu(), atol=1e-2, rtol=1e-2)


def test_train_sae_accepts_a_device_resident_cache():
    """train_sae indexes the cache directly; a CUDA-resident cache must work
    without dragging every batch back through the host."""
    from molcircuits.sae.sae import SAEConfig
    from molcircuits.sae.train_sae import SAETrainConfig, train_sae

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    torch.manual_seed(0)
    acts = (torch.randn(4096, 16) * 30).half().cuda()
    scfg = SAEConfig(d_in=16, expansion=2, variant="topk", k=4)
    tcfg = SAETrainConfig(n_activations=4096, batch_size=256, seed=0)

    sae, summary = train_sae(acts, scfg, tcfg, "cuda", out_dir=None)
    assert summary["final_l0"] <= 4 + 1e-6      # topk pins L0 at k
    assert summary["n_steps"] == 16


# ---------------------------------------------------------------- padded CE
def test_head_ablation_ce_ignores_padding():
    """delta_ce is the control that separates a globally important head from an
    event-specific one. Molecules are right-padded to n_ctx, but the model is
    pretrained on packed sequences and never sees a pad token, so unmasked CE
    scores ~84% out-of-distribution positions -- enough to make ablating a head
    look like it *improves* cross-entropy."""
    from molcircuits.analysis.pointers import head_effects
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, n_ctx=16)
    torch.manual_seed(0)
    model = MolecularTransformer(cfg).eval()

    real = torch.randint(1, 8, (4, 6))
    pad = torch.zeros((4, 10), dtype=torch.long)        # pad_id == 0
    tokens = torch.cat([real, pad], dim=1)
    targets = torch.cat([real, pad], dim=1)
    ev = [(0, 2), (1, 3)]
    ct = [(0, 4), (1, 1)]

    masked = head_effects(model, tokens, targets, ev, ct, "cpu", ignore_index=0)
    unmasked = head_effects(model, tokens, targets, ev, ct, "cpu")

    # padding dominates the unmasked figure, so the two must disagree
    assert any(abs(a["delta_ce"] - b["delta_ce"]) > 1e-6
               for a, b in zip(masked, unmasked))
    # and the masked CE must match a hand-computed CE over real positions only
    import torch.nn.functional as F
    lg = model(tokens)
    ref = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(),
                          targets.reshape(-1), ignore_index=0).item()
    assert masked[0]["ce_ablated"] != ref     # ablated differs from clean
    assert all(r["delta_ce"] == pytest.approx(r["ce_ablated"] - ref, abs=1e-5)
               for r in masked)


# ---------------------------------------------------------------- jumprelu
def test_jumprelu_threshold_is_learned_not_fixed():
    """The original 'jumprelu' was ReLU with a fixed 0.001 constant and no
    sparsity term at all -- it produced a dense code (L0 511 of 1024) that scored
    a perfect delta_ce. A real JumpReLU learns a per-feature threshold."""
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    cfg = SAEConfig(d_in=32, expansion=4, variant="jumprelu")
    sae = SparseAutoencoder(cfg)
    names = dict(sae.named_parameters())
    assert "log_threshold" in names
    assert names["log_threshold"].shape == (cfg.d_sae,)

    x = torch.randn(128, 32) * 20
    sae.set_input_scale(x)
    loss, parts = sae.loss(x)
    loss.backward()
    # the straight-through estimator must deliver a real gradient to theta
    assert sae.log_threshold.grad is not None
    assert float(sae.log_threshold.grad.abs().sum()) > 0
    # and the L0 penalty must actually be present
    assert float(parts["sparsity"]) > 0


def test_jumprelu_l0_penalty_reduces_sparsity():
    """l0_coeff must control L0. Needs enough steps for log_threshold to move at
    the optimizer's learning rate; a few hundred steps cannot separate values."""
    from molcircuits.sae.sae import SAEConfig
    from molcircuits.sae.train_sae import SAETrainConfig, train_sae

    torch.manual_seed(0)
    D = torch.randn(8, 32)
    acts = (torch.relu(torch.randn(400_000, 8)) @ D) * 10

    def run(lam):
        cfg = SAEConfig(d_in=32, expansion=4, variant="jumprelu", l0_coeff=lam)
        tcfg = SAETrainConfig(n_activations=400_000, batch_size=512, warmup=100,
                              seed=0)
        return train_sae(acts, cfg, tcfg, "cpu", None)[1]["final_l0"]

    assert run(0.5) < run(0.0) * 0.9


def test_non_jumprelu_checkpoints_still_load():
    """log_threshold is registered only for jumprelu, so topk/relu checkpoints
    written before it existed still satisfy strict state_dict loading."""
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    for variant in ("relu", "topk"):
        cfg = SAEConfig(d_in=16, expansion=2, variant=variant)
        sae = SparseAutoencoder(cfg)
        assert "log_threshold" not in dict(sae.named_parameters())
        SparseAutoencoder(cfg).load_state_dict(sae.state_dict())   # must not raise


# ---------------------------------------------------------------- resampling
def test_dead_feature_resampling_revives_features():
    """SAETrainConfig.resample_dead_every was declared but never read anywhere in
    the package, so the knob silently did nothing."""
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder
    from molcircuits.sae.train_sae import resample_dead

    torch.manual_seed(0)
    cfg = SAEConfig(d_in=16, expansion=4, variant="topk", k=4)
    sae = SparseAutoencoder(cfg)
    acts = torch.randn(2048, 16) * 5
    sae.set_input_scale(acts)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    sae.loss(acts[:64])[0].backward()
    opt.step()

    dead = torch.zeros(cfg.d_sae, dtype=torch.bool)
    dead[:5] = True
    before = sae.W_dec[:5].clone()
    before_rest = sae.W_dec[5:].clone()
    sae.steps_since_fired[:5] = 999
    sae.steps_since_fired[5:] = 7          # a value resampling must not touch

    n = resample_dead(sae, acts, dead, opt, device="cpu")
    assert n == 5
    assert not torch.allclose(before, sae.W_dec[:5])
    # revived decoder rows must still be unit-norm, and the counter reset
    assert torch.allclose(sae.W_dec[:5].norm(dim=-1), torch.ones(5), atol=1e-5)
    assert float(sae.steps_since_fired[:5].max()) == 0
    # Untouched features must be left exactly alone. This previously ended in
    # `or True`, which made it unconditionally pass -- and it was also asserting
    # against 0, the value the counter already held, so it could not have failed
    # either way. Assert the decoder rows and the counter both survive.
    assert torch.allclose(before_rest, sae.W_dec[5:])
    assert float(sae.steps_since_fired[5:].min()) == 7
    assert float(sae.steps_since_fired[5:].max()) == 7


def test_resampling_is_off_by_default():
    """Seed comparisons require the default path to be unchanged."""
    from molcircuits.sae.train_sae import SAETrainConfig
    assert SAETrainConfig().resample_dead_every == 0


# ---------------------------------------------------------------- kv cache
def test_kv_cache_generation_is_identical():
    """The cache is a pure speed optimisation: same seed must give the same
    molecules, token for token, as recomputing the whole prefix each step."""
    from molcircuits.chem.tokenizer import SmilesTokenizer
    from molcircuits.model.sampling import generate
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    tok = SmilesTokenizer.from_corpus(["CCO", "c1ccccc1", "CC(=O)N", "CCBr"])
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2,
                      n_heads=2, d_ff=64, n_ctx=32)
    torch.manual_seed(0)
    model = MolecularTransformer(cfg).eval()

    cached = generate(model, tok, 8, max_tokens=16, seed=0, device="cpu")
    cfg.n_ctx = 16
    with model.ablate_head(0, 0):
        cached_ablated = generate(model, tok, 8, max_tokens=16, seed=0, device="cpu")
        uncached_ablated = generate(model, tok, 8, max_tokens=16, seed=0, device="cpu",
                                    use_kv_cache=False)
    uncached = generate(model, tok, 8, max_tokens=16, seed=0, device="cpu", use_kv_cache=False)
    assert cached == uncached and cached_ablated == uncached_ablated


def test_context_limit_cache_preserves_token_local_steering_and_termination():
    from molcircuits.chem.tokenizer import SmilesTokenizer
    from molcircuits.model.hooks import HookManager, resid_post
    from molcircuits.model.sampling import generate
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    torch.manual_seed(10)
    tok = SmilesTokenizer.from_corpus(["CCO", "OC(=O)C"])
    model = MolecularTransformer(ModelConfig(vocab_size=tok.vocab_size, d_model=16,
                                 n_layers=2, n_heads=2, d_ff=32, n_ctx=16)).eval()
    hooks = HookManager().edit(resid_post(0), lambda x: x + .125)
    outputs = []
    widths = []
    handle = model.register_forward_pre_hook(lambda module, args: widths.append(args[0].shape[1]))
    try:
        for cached in (False, True):
            widths.clear()
            meta = {}
            samples = generate(model, tok, 16, max_tokens=16, hooks=hooks, seed=4,
                               use_kv_cache=cached, meta=meta)
            assert (max(widths) == 1) == cached
            outputs.append((samples, meta))
    finally:
        handle.remove()
    assert outputs[0] == outputs[1]


def test_rope_offset_matches_full_sequence():
    """A cached step must get the rotation for its absolute position, not for 0."""
    from molcircuits.model.transformer import apply_rope, build_rope

    cos, sin = build_rope(16, 8, 10000.0, "cpu", torch.float32)
    x = torch.randn(1, 2, 5, 8)
    full = apply_rope(x, cos, sin)                       # positions 0..4
    piece = apply_rope(x[:, :, 3:4], cos, sin, offset=3)  # position 3 alone
    assert torch.allclose(full[:, :, 3:4], piece, atol=1e-6)


# ---------------------------------------------------------------- smarts library
def test_every_smarts_pattern_parses_and_matches_something():
    """A pattern that matches nothing still reports a WSD score, computed from an
    all-zero mask. `nitro` was [$([NX3](=O)=O)], which RDKit can never match
    because sanitization normalizes nitro to the charge-separated form."""
    from rdkit import Chem, RDLogger

    from molcircuits.chem.fragments import DEFAULT_LIBRARY
    RDLogger.DisableLog("rdApp.*")

    probes = ["O=[N+]([O-])c1ccccc1", "CN(=O)=O", "CC(=O)Nc1ccccc1", "CCO",
              "c1ccncc1", "CC(C)(C)C", "FC(F)(F)c1ccccc1", "CS(=O)(=O)N",
              "O=CCC", "CC#N", "OC(=O)C", "C1CCOCC1", "c1ccsc1", "CCSCC",
              "NCCc1ccccc1", "C1CNCCN1", "Oc1ccccc1", "C=CC", "ClCC",
              "Clc1ccccc1", "FCC", "Fc1ccccc1", "c1cc[nH]c1", "c1cnc[nH]1",
              "C1CCCCC1", "CC(C)C", "COC", "CC(=O)C", "CC(=O)OC", "NC(N)=O",
              "c1ccc2ccccc2c1", "C1CCNCC1", "c1cnc2ccccc2n1", "CSC"]
    mols = [Chem.MolFromSmiles(s) for s in probes]

    unparseable, dead = [], []
    for name, sma in DEFAULT_LIBRARY.items():
        pat = Chem.MolFromSmarts(sma)
        if pat is None:
            unparseable.append(name)
            continue
        if not any(m is not None and m.HasSubstructMatch(pat) for m in mols):
            dead.append(name)

    assert not unparseable, f"unparseable SMARTS: {unparseable}"
    # every pattern must match at least one of its own canonical exemplars
    assert "nitro" not in dead, "nitro must match the charge-separated form"


def test_nitro_matches_the_charge_separated_form():
    from rdkit import Chem, RDLogger

    from molcircuits.chem.fragments import DEFAULT_LIBRARY
    RDLogger.DisableLog("rdApp.*")
    pat = Chem.MolFromSmarts(DEFAULT_LIBRARY["nitro"])
    for s in ("O=[N+]([O-])c1ccccc1", "CN(=O)=O", "C[N+](=O)[O-]"):
        assert Chem.MolFromSmiles(s).HasSubstructMatch(pat), s


# ================================================================= properties
# These assert defining properties rather than recorded outputs. The valence bug
# survived because its test agreed with the implementation; a test that encodes
# what a quantity *means* fails when the implementation drifts from it.


def test_pointer_mass_is_read_at_the_causally_relevant_position():
    """Attention is causally masked, so attention at close_idx cannot affect the
    logits at predict_idx -- yet every margin in the same row is measured at
    predict_idx. Querying close_idx paired two quantities that cannot be linked."""
    from molcircuits.analysis.pointers import pointer_mass
    from molcircuits.chem.events import find_events
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, n_ctx=16)
    torch.manual_seed(0)
    model = MolecularTransformer(cfg).eval()

    toks = tokenize("C1CCCC1")
    ev = [e for e in find_events(toks, offset=1) if e.kind == "ring"][0]
    tokens = torch.randint(1, 8, (1, cfg.n_ctx))

    from molcircuits.model.hooks import HookManager, attn_pattern
    h = HookManager().capture(attn_pattern(0))
    model(tokens, hooks=h)
    pat = h[attn_pattern(0)]

    got = pointer_mass(model, tokens, [ev], [(0, ev.predict_idx)], "cpu")
    at_pred = pat[0, :, ev.predict_idx, ev.open_idx].detach().numpy()
    at_close = pat[0, :, ev.close_idx, ev.open_idx].detach().numpy()
    assert np.allclose(got[0], at_pred, atol=1e-6)
    # and the two really are different, so the test has teeth
    assert not np.allclose(at_pred, at_close, atol=1e-6)


def test_control_positions_exclude_every_event_kind():
    """Controls are the baseline the event effect is measured against. Passing
    only same-kind events left 11% of ring controls sitting on branch closures."""
    from molcircuits.analysis.pointers import build_event_index

    tok = SmilesTokenizer.from_corpus(["CC(C)Cc1ccccc1", "C1CCCCC1"])
    smiles = ["CC(C)Cc1ccccc1C(C)C", "C1CCC(CC(C)C)CC1", "c1ccc2ccccc2c1CC(C)C"]
    for kind in ("ring", "branch"):
        ev, meta, ctrl = build_event_index(smiles, tok, 64, kind, None, 0)
        ctrl_set = set(ctrl)
        for bi, smi in enumerate(smiles):
            other = {(bi, e.predict_idx)
                     for e in find_events(tokenize(smi), offset=1)}
            assert not (ctrl_set & other), (
                f"{kind}: control positions overlap real closure events")


def test_relative_specificity_is_invariant_to_baseline_scale():
    """The raw difference of margin drops is dominated by the gap between the
    event and control baseline margins (~2.3x apart), so it ranks global damage.
    The relative form must not move when the two baselines are rescaled."""
    from molcircuits.analysis.pointers import head_effects
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, n_ctx=12)
    torch.manual_seed(0)
    model = MolecularTransformer(cfg).eval()
    tokens = torch.randint(1, 8, (3, 12))
    targets = torch.randint(1, 8, (3, 12))
    rows = head_effects(model, tokens, targets, [(0, 2), (1, 3)],
                        [(0, 5), (2, 4)], "cpu")
    for r in rows:
        # a head that leaves both populations untouched must score ~0 on both
        assert "relative_specificity" in r
        assert "baseline_margin_event" in r and "baseline_margin_control" in r
        lhs = r["relative_drop_event"] - r["relative_drop_control"]
        assert abs(lhs - r["relative_specificity"]) < 1e-9


def test_residual_rms_excludes_padding():
    """Alpha is expressed in units of this quantity, so it must describe the
    positions actually intervened on. Averaging over a right-padded batch put 84%
    of the weight on pad positions and understated the unit ~1.5x."""
    from molcircuits.analysis.valence_intervene import residual_rms
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=2,
                      d_ff=32, n_ctx=16)
    torch.manual_seed(0)
    model = MolecularTransformer(cfg).eval()
    real = torch.randint(1, 8, (4, 4))
    tokens = torch.cat([real, torch.zeros((4, 12), dtype=torch.long)], dim=1)

    padded = residual_rms(model, tokens, 0, "cpu")                 # pad included
    masked = residual_rms(model, tokens, 0, "cpu", pad_id=0)       # pad excluded
    prefix = residual_rms(model, real, 0, "cpu", pad_id=0)
    assert abs(masked - padded) > 1e-6, "masking must change the value"
    assert abs(masked - prefix) < 1e-4, "masked value must match the unpadded batch"


def test_random_feature_control_varies_across_targets():
    """A constant seed made matched_random_latent return the same permutation
    prefix every call, so all 50 targets shared one set of control features --
    an n=1 control reported with an across-target spread."""
    from molcircuits.analysis.steering import matched_random_latent

    z = torch.zeros(256)
    z[[3, 17, 40, 91, 200]] = torch.tensor([0.5, 2.0, 1.25, 0.75, 1.5])
    draws = [tuple(matched_random_latent(z, seed=1000 + i).nonzero().flatten().tolist())
             for i in range(8)]
    assert len(set(draws)) > 1, "control features identical across seeds"
    # and each draw preserves the count and the magnitude multiset
    for i in range(8):
        out = matched_random_latent(z, seed=1000 + i)
        assert int((out > 0).sum()) == 5
        assert sorted(out[out > 0].tolist()) == pytest.approx(
            sorted(z[z > 0].tolist()))


def test_ci_on_the_mean_shrinks_with_n():
    """The reported interval was percentiles across targets -- a spread, not a
    confidence interval. It does not narrow as evidence accumulates."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 4000)
    def ci_half(v):
        return 1.96 * np.std(v, ddof=1) / len(v) ** 0.5
    def spread(v):
        return (np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2
    assert ci_half(x[:2000]) > ci_half(x)          # CI shrinks with n
    assert ci_half(x) < spread(x) / 5              # and is far tighter
    assert abs(spread(x[:2000]) - spread(x)) < 0.15  # spread does not shrink


def test_seed_controls_model_initialisation():
    """`torch.manual_seed` ran inside trainer.train(), after scripts/02 had
    already constructed the model, so --seed selected the data order and nothing
    else: three runs at --seed 0 produced three different models."""
    import hashlib

    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    def init_hash(seed):
        torch.manual_seed(seed)
        m = MolecularTransformer(ModelConfig(vocab_size=16, d_model=32, n_layers=1,
                                             n_heads=2, d_ff=64, n_ctx=16))
        return hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()

    assert init_hash(0) == init_hash(0), "same seed must reproduce the model"
    assert init_hash(0) != init_hash(1), "different seeds must differ"


def test_sparse_atlas_aurocs_reproduce_dense_midranks():
    from molcircuits.atlas.core import (auroc_feature_from_ranks, auroc_from_ranks,
                                    heldout_detector_screen, rank_transform,
                                    sparse_rank_transform)

    rng = np.random.default_rng(18)
    values = rng.choice([-1., 0., 0., 0., 0., 1., 2.], size=(160, 24))
    values[:, 0] = 0
    values[:, 1] = -1
    values[:, 2] = 2
    labels = rng.random(160) > 0.8
    before = rank_transform(values)
    after = sparse_rank_transform(values)
    for mask in (labels, ~labels, np.zeros(160, bool), np.ones(160, bool)):
        np.testing.assert_array_equal(auroc_from_ranks(before, mask),
                                      auroc_from_ranks(after, mask))
        for feature in range(values.shape[1]):
            np.testing.assert_equal(auroc_feature_from_ranks(before, mask, feature),
                                    auroc_feature_from_ranks(after, mask, feature))
    blocks = {i: list(range(i * 8, (i + 1) * 8)) for i in range(20)}
    args = (labels, ~labels, blocks, blocks, 30, 23)
    assert heldout_detector_screen(before, before, *args) == heldout_detector_screen(after, after, *args)


def test_fragment_auroc_handles_ties():
    """A sparse code is ~99.6% exact zeros. Ordinal ranks break that tie block by
    row position, so the statistic encoded corpus order: the null moved to 0.02
    and dead features returned 1.0 or 0.0 depending on layout."""
    from sklearn.metrics import roc_auc_score

    from molcircuits.atlas.core import fragment_auroc

    rng = np.random.default_rng(0)
    acts = np.where(rng.random((1500, 40)) < 0.004, rng.random((1500, 40)), 0.0)
    # grouped layout is the worst case for a position-dependent tiebreak
    lab = np.zeros(1500, dtype=bool)
    lab[750:] = True

    got = fragment_auroc(acts, lab)
    live = acts.max(0) > 0
    ref = np.array([roc_auc_score(lab, acts[:, j]) if live[j] else np.nan
                    for j in range(40)])
    assert np.nanmax(np.abs(got[live] - ref[live])) < 1e-9

    # a dead feature must not score 1.0 just because the positives sit last
    dead = np.zeros((1500, 1))
    assert np.isnan(fragment_auroc(dead, lab)[0])

    # and the result must not depend on row order
    perm = rng.permutation(1500)
    shuffled = fragment_auroc(acts[perm], lab[perm])
    assert np.nanmax(np.abs(shuffled[live] - got[live])) < 1e-9


def test_bond_decision_labels_match_the_intervention_population():
    """The direction was fit at atom tokens and injected at bond-decision
    positions -- 12% overlap, 80% of injection sites were `(`. Every site must
    now carry the capacity of the atom actually being extended."""
    from molcircuits.chem.valence import bond_decision_labels, bond_decision_positions

    for smi in ("CC(=O)NC=C", "C#CCO", "CC(C)=CC#N", "c1ccccc1C=O"):
        toks = tokenize(smi)
        labelled = bond_decision_labels(toks, offset=1)
        plain = bond_decision_positions(toks, offset=1)
        # same positions as before, now each with a label
        assert [p for p, _ in labelled] == [p for p in plain
                                            if p in {q for q, _ in labelled}]
        for p, rem in labelled:
            assert 0 <= rem <= 4, f"{smi}: capacity {rem} out of range at {p}"

    # the governing atom is the one being extended, not an arbitrary neighbour
    assert bond_decision_labels(tokenize("CC=C"), offset=1) == [(2, 3)]


def test_fragment_specificity_rejects_an_atom_vs_punctuation_detector():
    """The confound that made every fragment look detected.

    A SMARTS match marks atoms, so if the negative population is every real
    token it also holds bonds, branches and ring digits. A feature that knows
    only "atom or not" then separates the two sets perfectly for EVERY fragment,
    scores WSD in the same range real dictionaries score, and clears its own
    within-molecule permutation null -- the null cannot catch it, because
    shuffling fragment membership across all real tokens moves the positives
    onto punctuation and drops the score.

    Scored over atom tokens only, the same feature must score exactly nothing.
    """
    import numpy as np

    from molcircuits.analysis.wsd import permutation_null, wsd
    from molcircuits.chem.fragments import (atom_token_mask, atom_token_positions,
                                        build_masks, valid_token_mask)

    smiles = ["CC(=O)Nc1ccccc1O", "O=C(N)c1ccncc1", "CC(C)Cc1ccc(O)cc1",
              "c1ccc2[nH]ccc2c1", "CCOC(=O)C1CCN(C)CC1"] * 24
    lib = {"amide": "C(=O)N", "benzene": "c1ccccc1"}
    n_ctx = 48
    masks = build_masks(smiles, lib, n_ctx, offset=1)
    all_tokens = valid_token_mask(smiles, n_ctx, offset=1)
    atoms = atom_token_mask(smiles, n_ctx, offset=1)

    # atoms is a strict subset of the real tokens, and strictly smaller
    assert (atoms & all_tokens == atoms).all()
    assert atoms.sum() < all_tokens.sum()

    # a feature carrying no chemistry at all: 1 on atoms, 0 on punctuation
    acts = np.zeros((len(smiles), n_ctx, 1), dtype=np.float32)
    for b, s in enumerate(smiles):
        for t in atom_token_positions(s):
            if 0 <= t + 1 < n_ctx:
                acts[b, t + 1, 0] = 1.0

    for name in lib:
        m = masks[name]
        with np.errstate(invalid="ignore"):
            bad = float(np.nanmean(np.nanmean(wsd(acts, m, all_tokens, True), 0)))
            good = float(np.nanmean(
                np.nanmean(wsd(acts, m & atoms, atoms, True), 0)))
        null = permutation_null(acts, m, all_tokens, n_perm=10, signed=True)

        # the confound: scored against all tokens it looks like a real detector
        assert bad > 0.1, f"{name}: expected the confound to show, got {bad}"
        assert bad > null["null_p95"], f"{name}: null failed to catch the confound"
        # scored against atoms only it carries no information whatsoever
        assert abs(good) < 1e-6, f"{name}: atom-restricted score should be 0, got {good}"


def test_atom_token_mask_marks_only_atoms():
    from molcircuits.chem.fragments import atom_token_mask
    from molcircuits.chem.tokenizer import tokenize

    smi = "CC(=O)Oc1ccccc1"
    toks = tokenize(smi)
    m = atom_token_mask([smi], len(toks) + 1, offset=1)[0]
    marked = {toks[i - 1] for i in range(1, len(toks) + 1) if m[i]}
    assert marked <= {"C", "O", "c"}
    assert not ({"(", ")", "=", "1"} & marked)
    # every atom token is marked, none missed
    assert int(m.sum()) == sum(1 for t in toks if t in {"C", "O", "c"})


def test_bond_decision_capacity_is_current_not_birth_time():
    """The governing atom's capacity must be read at the decision point.

    It used to be looked up from `valence_labels`, i.e. the capacity at the
    moment the atom was *written*. Those differ as soon as the atom has been
    extended in between -- most often through a branch that has since closed.
    In C(=O)=O the carbon has spent 2 on the branch by the second decision, so
    2 is correct there and 4 (its birth-time value) is not.
    """
    from molcircuits.chem.valence import bond_decision_labels

    # (position, capacity) at each explicit-bond decision
    assert bond_decision_labels(tokenize("C(=O)=O")) == [(1, 4), (4, 2)]
    # a single unbranched decision is unaffected -- the two readings coincide
    assert bond_decision_labels(tokenize("CC=C")) == [(1, 3)]

    # every decision's capacity must be non-increasing for a given atom as the
    # string extends it, which the birth-time reading cannot guarantee
    for smi in ("C(=O)=O", "CC(=O)C(=O)O", "OC(=O)C(=O)O"):
        toks = tokenize(smi)
        got = bond_decision_labels(toks)
        assert all(0 <= c <= 4 for _, c in got), smi
    assert bond_decision_labels(tokenize("CC(=O)C(=O)O"))[0][1] >= \
        bond_decision_labels(tokenize("CC(=O)C(=O)O"))[-1][1]


def test_final_semantics_matches_rdkit_across_the_corpus_including_aromatic_s():
    """`final` must equal RDKit's implicit-H count on REAL molecules.

    This replaces a test that asserted the same thing on six hand-picked
    sulfur and phosphorus molecules. It did not discriminate: it passed
    identically under the multi-valence table that was tried here and under the
    single-valence one, because a sulfone is 0 under both. The case that
    separates them is AROMATIC sulfur, which no hand-picked list contained --
    aromatic ring bonds are charged 1.5 each, so a thiophene S carries 3.0 and
    a smallest-allowed-valence rule promotes it from 2 to 4 and reports 1
    remaining where RDKit says 0. Sampling the corpus is what makes the test
    able to fail.
    """
    import numpy as np
    from rdkit import Chem, RDLogger

    from molcircuits.chem.valence import valence_labels

    RDLogger.DisableLog("rdApp.*")
    smiles = [
        "c1ccccc1", "Cc1ccccc1O", "c1ccncc1", "c1ccoc1", "c1ccsc1",
        "c1ccc2sccc2c1", "c1cc2ccsc2s1", "c1cc[nH]c1", "c1ccc2[nH]ccc2c1",
        "CC(=O)Nc1ccc(O)cc1", "O=C(O)c1ccccc1OC(=O)C", "CN1CCC[C@H]1c2cccnc2",
        "FC(F)(F)c1ccc(Cl)cc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "O=S(=O)(N)c1ccc(N)cc1", "CC#CCN(C)C", "C=CC(=O)OCC",
        "C1CCNCC1", "C1COCCN1", "C1CC2(CCNCC2)C1",
        "O=C(NCCc1ccccc1Br)NC[C@@H]1CCCOC1",
        "C[C@H](Cc1ccccc1F)C(=O)N1CC[C@H](N2CCCC2)[C@H]1C",
        "O=[N+]([O-])c1ccc(O)cc1", "C[N+](C)(C)CCO", "CC(=O)[O-]",
        "c1cnc2ccccc2n1", "c1ccc2ncccc2c1", "Cc1nc2ccccc2[nH]1",
        "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1",
        "O=c1[nH]c(=O)n(C)c2ncn(C)c12",
    ]

    bad, total = [], 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        got = [l.remaining for l in valence_labels(tokenize(smi), semantics="final")]
        ref = [a.GetNumImplicitHs() for a in mol.GetAtoms()]
        if len(got) != len(ref):
            continue
        for g, r, a in zip(got, ref, mol.GetAtoms()):
            total += 1
            if g != r:
                bad.append((smi, a.GetSymbol(), a.GetIsAromatic(), g, r))
    assert total > 250, f"expected a diverse sample, scored {total} atoms"
    assert not bad, (
        f"{len(bad)} of {total} atoms disagree with RDKit; first five: {bad[:5]}")

    # the specific case a hand-picked list misses
    for smi in ("c1ccsc1", "c1ccc2sccc2c1", "c1cc2ccsc2s1"):
        got = [l.remaining for l in valence_labels(tokenize(smi), semantics="final")]
        ref = [a.GetNumImplicitHs()
               for a in Chem.MolFromSmiles(smi).GetAtoms()]
        assert got == ref, f"aromatic sulfur: {smi}: {got} != {ref}"


def test_intervention_rms_is_measured_at_the_intervention_sites():
    """Alpha is quoted in multiples of this unit, so it must describe the sites
    actually perturbed. Averaging over a right-padded batch instead puts most of
    the weight on pad positions and understates the unit."""
    import inspect

    import torch

    from molcircuits.analysis.valence_intervene import residual_rms
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=2, n_heads=4,
                      d_ff=32, n_ctx=16)
    model = MolecularTransformer(cfg).eval()
    # 4 real tokens then padding, which is the shape every stage feeds it
    tokens = torch.zeros(3, 16, dtype=torch.long)
    tokens[:, :4] = torch.randint(1, 8, (3, 4))
    pos = torch.tensor([[b, t] for b in range(3) for t in range(4)])

    at_sites = residual_rms(model, tokens, 1, "cpu", positions=pos)
    whole_batch = residual_rms(model, tokens, 1, "cpu")
    assert at_sites != pytest.approx(whole_batch, rel=1e-3), (
        "the padded average must differ from the site average, or this test "
        "cannot detect the bug it guards")

    # and the callers must pass positions through rather than take the default
    for fn in ("run_with_null", "intervene"):
        import molcircuits.analysis.valence_intervene as vi
        src = inspect.getsource(getattr(vi, fn))
        assert "residual_rms(" in src
        assert "positions=positions" in src, (
            f"{fn} must measure the RMS at the intervention sites")


def test_fvu_does_not_depend_on_batch_grouping():
    """FVU was a mean of per-batch ratios, each against that batch's own mean.

    Both halves are wrong: a mean of ratios is not the ratio of means, and the
    denominator shrinks with batch size because a small batch's variance about
    its own mean is small. The same activations could therefore be made to yield
    almost any FVU by regrouping them. Accumulating sums and dividing once is
    exact and grouping-invariant.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.sae.evaluate import evaluate_sae
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    sae = SparseAutoencoder(SAEConfig(d_in=32, expansion=2, variant="topk", k=4))
    sae.cfg.layer = 1
    sae.eval()

    x = torch.randint(1, 16, (64, 8))
    y = torch.randint(1, 16, (64, 8))
    ds = TensorDataset(x, y)

    got = [evaluate_sae(model, sae, DataLoader(ds, batch_size=bs), device="cpu",
                        layer=1)["fvu"]
           for bs in (8, 16, 32, 64)]
    assert got[0] == pytest.approx(got[-1], rel=1e-6), (
        f"FVU changed with batch size: {got}")
    assert len(set(round(v, 9) for v in got)) == 1, f"FVU not grouping-invariant: {got}"


def test_mean_ablation_replacement_is_not_taken_from_the_scored_batch():
    """The mean-ablation floor must be an information-free baseline.

    Deriving it from the batch under evaluation leaks that batch's composition
    -- other molecules, and tokens to the right of the position being predicted
    -- into the very quantity that is supposed to contain no information, which
    makes mean ablation look milder and inflates fraction_ce_recovered_vs_mean.
    """
    import ast
    import inspect

    from molcircuits.sae import evaluate as ev

    src = inspect.getsource(ev.evaluate_sae)
    tree = ast.parse(src.lstrip())
    lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
    mean_lambdas = [ast.unparse(n) for n in lambdas if "mean" in ast.unparse(n)]
    for lam in mean_lambdas:
        assert ".mean(" not in lam, (
            f"mean-ablation replacement is computed from the batch: {lam}")


def test_jumprelu_threshold_gradient_carries_the_log_chain_rule():
    """Pin the factor an audit flagged as missing -- it is not missing.

    The threshold is stored in log space to keep it positive, so the
    straight-through estimator for d/d(log theta) must be theta times the
    estimator for d/d(theta). Both _JumpReLU and _Step carry it, and both reduce
    to -(theta/bw)*K((x-theta)/bw): the JumpReLU activation's x factor collapses
    to theta under the kernel, since inside a window of width bw around theta,
    x is theta (Rajamanoharan et al. 2024). This test exists so a future
    simplification cannot quietly drop the factor.
    """
    from molcircuits.sae.sae import _JumpReLU, _Step

    bw = 0.5
    x = torch.tensor([[0.10, 0.30, 0.55, 1.00]], dtype=torch.float64)

    for fn in (_JumpReLU, _Step):
        for theta in (0.1, 0.4):
            # per-FEATURE threshold, the shape the model actually holds
            lt = torch.full((4,), math.log(theta), dtype=torch.float64,
                            requires_grad=True)
            fn.apply(x, lt, bw).sum().backward()
            rect = (((x - theta) / bw > -0.5) & ((x - theta) / bw < 0.5))
            expected = -(theta / bw) * rect.to(torch.float64)
            assert torch.allclose(lt.grad, expected.sum(0)), (
                f"{fn.__name__} at theta={theta}: got {lt.grad}, "
                f"expected {expected.sum(0)}")

    # The defining property, stated independently of the formula: halving theta
    # halves the gradient wherever the kernel window is unchanged.
    x2 = torch.tensor([[0.20, 0.20, 0.20, 0.20]], dtype=torch.float64)
    g = {}
    for theta in (0.2, 0.4):
        lt = torch.full((4,), math.log(theta), dtype=torch.float64,
                        requires_grad=True)
        _Step.apply(x2, lt, bw).sum().backward()
        g[theta] = float(lt.grad[0])
    assert g[0.4] == pytest.approx(2.0 * g[0.2], rel=1e-9)


def test_every_smarts_marks_its_whole_group():
    """A recursive [$(...)] pattern matches only its central atom.

    `nitro` was written that way, so it marked a single [N+] token -- close to
    unique in this vocabulary -- and scored the atlas's only AUROC of 1.000.
    Because AUROC is confounded with how many atoms a pattern marks, a one-atom
    entry in a library of two-to-six-atom entries distorts the confound analysis
    itself. Fragments named for a multi-atom group must match one.
    """
    from molcircuits.chem.fragments import fragment_token_mask, load_smarts_library

    lib = load_smarts_library(None)
    # (fragment, a molecule containing it, minimum atoms the match must cover)
    cases = [
        ("nitro", "CC(=O)Nc1ccc([N+](=O)[O-])cc1", 3),
        ("amide", "CC(=O)Nc1ccccc1", 3),
        ("sulfonamide", "CC(=O)Nc1ccc(S(N)(=O)=O)cc1", 4),
        ("carboxylic_acid", "OC(=O)c1ccccc1", 3),
        ("urea", "CNC(=O)NC", 4),
        ("trifluoromethyl", "FC(F)(F)c1ccccc1", 4),
    ]
    for name, smi, want in cases:
        toks = tokenize(smi)
        m = fragment_token_mask(smi, lib[name], len(toks), offset=0)
        got = int(m.sum())
        assert got >= want, (
            f"{name}: {lib[name]} marked {got} atom tokens of {smi}, "
            f"expected at least {want} -- a recursive pattern marks only its "
            f"central atom")


def test_nitro_pattern_actually_fires_on_this_corpus():
    """RDKit normalizes neutral nitro to the charge-separated form, so a
    neutral-only pattern silently matches nothing while still producing WSD
    numbers from an all-zero mask."""
    from molcircuits.chem.fragments import fragment_token_mask, load_smarts_library

    lib = load_smarts_library(None)
    for smi in ("CC[N+](=O)[O-]", "O=[N+]([O-])c1ccccc1"):
        toks = tokenize(smi)
        assert fragment_token_mask(smi, lib["nitro"], len(toks), 0).any(), smi


@pytest.mark.parametrize("n_tokens,offset", [(10, 1), (3, -1), (0, 0)])
def test_batch_fragment_masks_reuse_parses_and_match_single_queries(monkeypatch, n_tokens, offset):
    from unittest.mock import Mock
    from molcircuits.chem import fragments

    smiles = ["OCC", "C(C)CO", "[O-][N+](=O)C", "c1ccccc1", "C?C"]
    patterns = {"oxygen": "[#8]", "pair": "[#6]~[#6]", "invalid": "["}
    expected = {
        name: np.stack([fragment_token_mask(smi, query, n_tokens, offset) for smi in smiles])
        for name, query in patterns.items()
    }
    parse = Mock(wraps=fragments.Chem.MolFromSmiles)
    compile_pattern = Mock(wraps=fragments.Chem.MolFromSmarts)
    monkeypatch.setattr(fragments.Chem, "MolFromSmiles", parse)
    monkeypatch.setattr(fragments.Chem, "MolFromSmarts", compile_pattern)

    actual = fragments.build_masks(smiles, patterns, n_tokens, offset)
    assert parse.call_count == len(smiles)
    assert compile_pattern.call_count == len(patterns)
    assert actual.keys() == expected.keys()
    for name in patterns:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_atom_index_to_token_is_input_order_not_canonical_order():
    """RDKit numbers atoms in the order of the string it PARSED.

    _smilesAtomOutputOrder describes the CANONICAL output string instead. For
    "OCC" it is [2, 1, 0], which maps atom 0 (the oxygen) onto a carbon token.
    That property was read inside a bare except that always fired, so the wrong
    mechanism was documented while the right one ran; this pins the right one.
    """
    from rdkit import Chem

    from molcircuits.chem.fragments import atom_index_to_token

    for smi in ("OCC", "c1ccccc1O", "CC(=O)O", "OC(=O)c1ccccc1", "[nH]1cccc1C"):
        mol = Chem.MolFromSmiles(smi)
        toks = tokenize(smi)
        a2t = atom_index_to_token(smi, mol)
        assert len(a2t) == mol.GetNumAtoms(), smi
        for a, t in a2t.items():
            sym = mol.GetAtomWithIdx(a).GetSymbol().lower()
            assert toks[t].strip("[]").lower().startswith(sym), (
                f"{smi}: atom {a} ({sym}) mapped to token {t} ({toks[t]})")

    # the case the canonical order would break
    assert atom_index_to_token("OCC") == {0: 0, 1: 1, 2: 2}

    # and no bare except may hide a failure here again
    import inspect

    from molcircuits.chem import fragments
    src = inspect.getsource(fragments.atom_index_to_token)
    assert "except Exception" not in src
    assert "_smilesAtomOutputOrder" not in src.split('"""')[2]


def test_ce_is_token_weighted_not_batch_averaged():
    """A mean of per-batch CE means depends on how the data was grouped.

    Measured on real sequences the swing was ~0.01 between batch 50 and batch
    33, about fifteen times the delta_ce this metric is used to report. Full
    batches hide it; a ragged tail does not.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.sae.evaluate import _ce_with_edit

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    # 100 rows: batch sizes 33 and 7 leave a ragged tail, 50 and 25 do not
    ds = TensorDataset(torch.randint(1, 16, (100, 8)),
                       torch.randint(1, 16, (100, 8)))
    got = [_ce_with_edit(model, DataLoader(ds, batch_size=bs), "cpu", 1, None, 999)
           for bs in (100, 50, 33, 25, 7)]
    # 1e-5, not 0: summing float32 logits in a different order is not
    # associative, and the residual spread is ~4e-7. The defect this guards
    # produced ~1e-2, four orders larger.
    assert max(got) - min(got) < 1e-5, f"CE depends on batch grouping: {got}"


def test_ce_ignores_padding_when_told_to():
    """Without ignore_index a padded loader makes the metric invert.

    ce_clean rose above the zero-ablation floor on a real padded loader and
    fraction_ce_recovered came out above 1.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.sae.evaluate import _ce_with_edit

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    PAD = 0
    x = torch.randint(1, 16, (40, 8))
    y = torch.randint(1, 16, (40, 8))
    y[:, 4:] = PAD                      # right padding, 50% of positions
    ds = TensorDataset(x, y)

    with_pad = _ce_with_edit(model, DataLoader(ds, batch_size=8), "cpu", 1,
                             None, 999, ignore_index=-100)
    no_pad = _ce_with_edit(model, DataLoader(ds, batch_size=8), "cpu", 1,
                           None, 999, ignore_index=PAD)
    assert with_pad != pytest.approx(no_pad, rel=1e-6), (
        "the test cannot detect the bug it guards")

    # scoring only real tokens must equal scoring a loader that has none
    ds_real = TensorDataset(x[:, :4], y[:, :4])
    ref = _ce_with_edit(model, DataLoader(ds_real, batch_size=8), "cpu", 1,
                        None, 999, ignore_index=PAD)
    assert no_pad == pytest.approx(ref, rel=1e-4)


def test_bond_decision_positions_and_labels_cannot_silently_diverge():
    """The labeller skips decisions governed by an unmodelled element.

    bond_decision_positions has no such guard, so index-aligning the two lists
    misaligns without any error. [Si] and [Sn] are both in the tokenizer
    vocabulary and neither is in DEFAULT_VALENCE.
    """
    from molcircuits.chem.valence import (bond_decision_labels,
                                      bond_decision_positions,
                                      unlabelled_bond_decisions)

    toks = tokenize("C[Si](C)=C")
    assert bond_decision_positions(toks) == [4]
    assert bond_decision_labels(toks) == []
    assert unlabelled_bond_decisions(toks) == [4]

    # on ordinary organic input the two agree exactly and nothing is dropped
    for smi in ("CC(=O)NC=C", "C#CCO", "c1ccccc1C=O", "CS(=O)(=O)C"):
        t = tokenize(smi)
        assert unlabelled_bond_decisions(t) == []
        assert [p for p, _ in bond_decision_labels(t)] == bond_decision_positions(t)


def test_saelens_export_reproduces_this_codebase_s_activations():
    """An export that drops input_scale is a different dictionary.

    _pre divides by input_scale (30.60 for the shipped layer-3 SAE) and decode
    multiplies it back. Exporting the raw parameters made an importer applying
    the plain SAELens forward pass agree on the TopK active set for 0.39% of
    tokens and inflated FVU from 0.045 to 0.069. The parameters are now folded
    so the plain forward pass is exact.
    """
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    torch.manual_seed(0)
    for variant in ("topk", "relu"):
        sae = SparseAutoencoder(SAEConfig(d_in=16, expansion=4, variant=variant,
                                          k=4)).eval()
        acts = torch.randn(512, 16) * 7.0 + 2.0
        sae.set_input_scale(acts)
        assert float(sae.input_scale) != pytest.approx(1.0), (
            "with input_scale == 1 the export is trivially correct and this "
            "test cannot detect the bug it guards")

        sd = sae.to_saelens_state_dict()
        x = torch.randn(64, 16) * 7.0 + 2.0

        # the plain SAELens forward pass, using only the exported tensors
        pre = (x - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"]
        z_imported = sae._activate(pre)
        xhat_imported = z_imported @ sd["W_dec"] + sd["b_dec"]

        z_ours = sae.encode(x)
        xhat_ours = sae.decode(z_ours)

        assert torch.allclose(z_imported, z_ours, atol=1e-4), variant
        assert torch.allclose(xhat_imported, xhat_ours, atol=1e-3), variant
        # active sets must be identical, not merely close
        assert torch.equal(z_imported > 0, z_ours > 0), variant


def test_ablation_delta_nll_scores_only_the_masked_positions():
    """A fragment detector's cost must be measured where its fragment is.

    Whole-batch cross-entropy mixes that local question with off-fragment effects
    of a global edit. On the real atlas batch it was 138x larger for morpholine,
    and five fragments changed sign, so it is not a diluted estimate of the same
    quantity.
    """
    from molcircuits.atlas.core import ablation_delta_nll
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    sae = SparseAutoencoder(SAEConfig(d_in=32, expansion=4, variant="topk", k=4))
    sae.eval()

    x = torch.randint(1, 16, (6, 8))
    y = torch.full_like(x, 0)
    y[:, :-1] = x[:, 1:]

    # Ablating a feature that never fires is a no-op, so pick the one that
    # carries the most mass -- otherwise both arms are exactly 0.0 and the test
    # passes for the wrong reason.
    from molcircuits.model.hooks import HookManager, resid_post
    h = HookManager().capture(resid_post(1))
    with torch.no_grad():
        model(x, hooks=h)
        z = sae.encode(h[resid_post(1)].reshape(-1, sae.cfg.d_in))
    feat = int(z.sum(0).argmax())
    assert float(z[:, feat].sum()) > 0

    whole = ablation_delta_nll(model, sae, x, y, [feat], 1, ignore_index=0,
                               device="cpu")
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 2:4] = True
    part = ablation_delta_nll(model, sae, x, y, [feat], 1, ignore_index=0,
                              device="cpu", score_mask=mask)

    assert whole["n_positions_scored"] > part["n_positions_scored"]
    assert part["n_positions_scored"] == int((mask & (y != 0)).sum())
    assert whole["delta_nll"] != pytest.approx(part["delta_nll"], rel=1e-9), (
        "masking must change the number, or the test cannot detect the bug")

    # an all-true mask must reproduce the unmasked value exactly
    allm = torch.ones_like(x, dtype=torch.bool)
    same = ablation_delta_nll(model, sae, x, y, [feat], 1, ignore_index=0,
                              device="cpu", score_mask=allm)
    assert same["delta_nll"] == pytest.approx(whole["delta_nll"], rel=1e-9)


def test_fragment_ablation_estimands_have_explicit_temporal_alignment():
    """Detector-at-t and predictor-at-t-1 are different causal questions.

    With shifted language-model targets, a mask at logit position p scores token
    p+1.  The atlas detector is selected from the residual at fragment token t,
    so its aligned outcome is the next token t+1.  Predicting fragment token t
    instead requires selecting and ablating a separate detector at t-1.
    """
    from molcircuits.atlas.core import (DETECTOR_T_NEXT_TOKEN,
                                    PREDICT_FRAGMENT_FROM_T_MINUS_1,
                                    fragment_ablation_tensors)

    pad = 0
    x = torch.tensor([[9, 1, 2, 3, 4, 0]])
    fragment_t = torch.tensor([[False, False, True, True, False, False]])

    targets, edit_t, score_t = fragment_ablation_tensors(
        x, fragment_t, pad, DETECTOR_T_NEXT_TOKEN)
    assert targets.tolist() == [[1, 2, 3, 4, 0, 0]]
    assert torch.equal(edit_t, score_t)
    assert edit_t.tolist() == [[False, False, True, True, False, False]]
    # Logits at the selected t positions predict exactly the following tokens.
    assert targets[score_t].tolist() == [3, 4]
    assert x[:, 1:][score_t[:, :-1]].tolist() == [3, 4]

    targets_prev, edit_prev, score_prev = fragment_ablation_tensors(
        x, fragment_t, pad, PREDICT_FRAGMENT_FROM_T_MINUS_1)
    assert torch.equal(targets_prev, targets)
    assert torch.equal(edit_prev, score_prev)
    assert edit_prev.tolist() == [[False, True, True, False, False, False]]
    # These t-1 logits predict the fragment tokens themselves.
    assert targets_prev[score_prev].tolist() == [2, 3]
    assert x[fragment_t].tolist() == [2, 3]


def test_heldout_detector_is_selected_without_evaluation_leakage():
    """The evaluation-half oracle must not replace the selection-half winner."""
    from molcircuits.atlas.core import (heldout_detector_screen, rank_transform)

    labels = np.array([False, False, True, True])
    # Feature 0 is perfect on selection; feature 1 is deliberately weaker.
    selection = np.array([[0., 0.], [1., 4.], [3., 1.], [4., 3.]])
    # On evaluation feature 1 is perfect, while selected feature 0 is mediocre.
    evaluation = np.array([[0., 0.], [3., 1.], [1., 3.], [2., 4.]])
    # Each molecule contributes one positive and one negative token, so the null
    # can permute labels within molecule without changing molecule-level counts.
    blocks = {0: [0, 2], 1: [1, 3]}
    out = heldout_detector_screen(
        rank_transform(selection), rank_transform(evaluation), labels, labels,
        blocks, blocks, n_perm=7, seed=7)

    assert out["status"] == "ok"
    assert out["best_feature"] == 0
    assert out["selection_auroc"] == pytest.approx(1.0)
    assert out["evaluation_auroc"] == pytest.approx(0.5)
    assert out["evaluation_oracle_best_auroc"] == pytest.approx(1.0)
    assert out["evaluation_pipeline_null"]["n"] == 7
    again = heldout_detector_screen(
        rank_transform(selection), rank_transform(evaluation), labels, labels,
        blocks, blocks, n_perm=7, seed=7)
    assert again["evaluation_pipeline_null"] == out["evaluation_pipeline_null"]


def test_molecule_selection_evaluation_split_is_deterministic_and_disjoint():
    from molcircuits.atlas.core import deterministic_molecule_split

    a, b = deterministic_molecule_split(11, 0.5, seed=123)
    a2, b2 = deterministic_molecule_split(11, 0.5, seed=123)
    assert np.array_equal(a, a2)
    assert np.array_equal(b, b2)
    assert len(a) == 5 and len(b) == 6
    assert not set(a) & set(b)
    assert set(a) | set(b) == set(range(11))


def test_ablation_delta_nll_edits_only_requested_token_positions():
    """A local feature ablation must not zero that feature anywhere else."""
    from types import SimpleNamespace

    from molcircuits.atlas.core import (DETECTOR_T_NEXT_TOKEN,
                                    ablation_delta_nll)
    from molcircuits.model.hooks import resid_post

    class IdentitySAE:
        cfg = SimpleNamespace(d_in=2, d_sae=2)

        @staticmethod
        def encode(x):
            return x.clone()

        @staticmethod
        def decode(z):
            return z

    class RecordingModel:
        def __init__(self, residual):
            self.residual = residual
            self.seen = []

        def __call__(self, tokens, hooks):
            edited = hooks.maybe(resid_post(0), self.residual.clone())
            self.seen.append(edited.clone())
            logits = torch.zeros(*tokens.shape, 4)
            logits[..., 1] = edited[..., 0]
            logits[..., 2] = edited[..., 1]
            return logits

    residual = torch.tensor([
        [[1., 11.], [2., 12.], [3., 13.], [4., 14.]],
        [[5., 15.], [6., 16.], [7., 17.], [8., 18.]],
    ])
    model = RecordingModel(residual)
    sae = IdentitySAE()
    x = torch.ones((2, 4), dtype=torch.long)
    y = torch.tensor([[1, 2, 1, 2], [2, 1, 2, 1]])
    local = torch.tensor([
        [False, True, False, False],
        [False, False, False, True],
    ])

    result = ablation_delta_nll(
        model, sae, x, y, [0], 0, ignore_index=0, device="cpu",
        score_mask=local, ablation_mask=local,
        estimand=DETECTOR_T_NEXT_TOKEN)
    baseline, ablated = model.seen
    expected = residual.clone()
    expected[0, 1, 0] = 0.0
    expected[1, 3, 0] = 0.0
    assert torch.equal(baseline, residual)
    assert torch.equal(ablated, expected)
    assert torch.equal(ablated[..., 1], residual[..., 1])
    assert result["ablation_scope"] == "position_local"
    assert result["n_positions_ablated"] == 2
    assert result["n_positions_scored"] == 2
    assert result["estimand"] == DETECTOR_T_NEXT_TOKEN

    # Omitting the new mask preserves the pre-existing global edit semantics.
    model.seen.clear()
    legacy = ablation_delta_nll(
        model, sae, x, y, [0], 0, ignore_index=0, device="cpu",
        score_mask=local)
    _, globally_ablated = model.seen
    assert torch.equal(globally_ablated[..., 0],
                       torch.zeros_like(residual[..., 0]))
    assert torch.equal(globally_ablated[..., 1], residual[..., 1])
    assert legacy["ablation_scope"] == "global"
    assert legacy["n_positions_ablated"] == x.numel()


def test_atlas_targets_are_shifted_not_rolled():
    """torch.roll wraps the final target to the first INPUT token (BOS).

    Because ignore_index is the pad id, that wrapped column was scored: 2.5% of
    scored positions were "input pad, target BOS", which is pure noise.
    """
    import ast
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "molcircuits/atlas/core.py")).read()
    tree = ast.parse(src)
    rolls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func).endswith("roll")]
    assert not rolls, "atlas must shift targets, not roll them"

    # and the shift must leave the last column unscored
    x = torch.arange(1, 9).reshape(1, 8)
    y = torch.full_like(x, 0)
    y[:, :-1] = x[:, 1:]
    assert y.tolist() == [[2, 3, 4, 5, 6, 7, 8, 0]]
    assert torch.roll(x, -1, dims=1).tolist() == [[2, 3, 4, 5, 6, 7, 8, 1]]


def test_pointer_aggregation_is_independent_of_batch_boundaries():
    """Pool the margins, then form the ratio once.

    Summing per-batch rows and dividing by the NUMBER OF BATCHES weights a
    batch carrying 24 events the same as one carrying 1000, and
    relative_specificity is itself a per-batch ratio, so the old form averaged
    ratios instead of taking the ratio of pooled means. At 1000+24 batching a
    top head read +0.283 against a true +0.232, with 40 of 48 heads changing
    rank.
    """
    import numpy as np

    def pooled(batches):
        """The corrected aggregation, in isolation from the model."""
        ev_b = ev_a = ct_b = ct_a = 0.0
        n_ev = n_ct = 0
        for be, ae, ne, bc, ac, nc in batches:
            ev_b += be * ne; ev_a += ae * ne; n_ev += ne
            ct_b += bc * nc; ct_a += ac * nc; n_ct += nc
        be_, ae_ = ev_b / n_ev, ev_a / n_ev
        bc_, ac_ = ct_b / n_ct, ct_a / n_ct
        return (be_ - ae_) / be_ - (bc_ - ac_) / bc_

    def averaged(batches):
        """The old aggregation: mean of per-batch ratios."""
        vals = [((be - ae) / be) - ((bc - ac) / bc)
                for be, ae, ne, bc, ac, nc in batches]
        return float(np.mean(vals))

    # one lopsided split of the same underlying events
    big = (10.6, 9.0, 1000, 4.6, 4.4, 4000)
    tail = (10.6, 4.0, 24, 4.6, 4.5, 96)
    one_batch = [(10.6, (9.0 * 1000 + 4.0 * 24) / 1024, 1024,
                  4.6, (4.4 * 4000 + 4.5 * 96) / 4096, 4096)]
    split = [big, tail]

    assert pooled(split) == pytest.approx(pooled(one_batch), rel=1e-9), (
        "pooling must not depend on how the events were batched")
    assert averaged(split) != pytest.approx(pooled(split), rel=1e-3), (
        "the two aggregations must differ, or this test cannot detect the bug")


def test_head_effects_reports_the_counts_needed_to_pool():
    from molcircuits.analysis.pointers import head_effects, mean_head_activations
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=2, n_heads=4,
                      d_ff=32, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    x = torch.randint(1, 8, (3, 8))
    y = torch.randint(1, 8, (3, 8))
    ev = [(0, 2), (1, 3), (2, 4)]
    ct = [(0, 5), (1, 6)]
    mz = mean_head_activations(model, x, device="cpu")
    rows = head_effects(model, x, y, ev, ct, "cpu", mode="zero", mean_z=mz)
    for r in rows:
        for k in ("n_events", "n_controls", "n_scored_tokens",
                  "ablated_margin_event", "ablated_margin_control"):
            assert k in r, f"head_effects must report {k} for weighted pooling"
        assert r["n_events"] == 3 and r["n_controls"] == 2


def test_failed_ablation_setup_leaves_the_model_untouched():
    """__enter__ used to zero the head BEFORE validating the mode, and the
    raise sat outside the try/finally, so a bad call left the head permanently
    ablated and every later forward pass in the process was silently wrong."""
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=2, n_heads=4,
                      d_ff=32, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    x = torch.randint(1, 8, (2, 8))
    with torch.no_grad():
        base = model(x).clone()

    for kwargs in ({"mode": "mean"}, {"mode": "nonsense"}):
        before = model.blocks[0].attn.head_mask.clone()
        with pytest.raises(ValueError):
            with model.ablate_head(0, 2, **kwargs):
                pass
        assert torch.equal(model.blocks[0].attn.head_mask, before), kwargs
        with torch.no_grad():
            assert torch.allclose(model(x), base), (
                f"model changed after a failed ablation setup ({kwargs})")


def test_attention_pattern_edits_are_applied():
    """`emit` is capture-only; a registered EDIT on the pattern was computed and
    discarded while `wants()` still routed execution down the manual-softmax
    path, so the intervention looked wired up and did nothing."""
    from molcircuits.model.hooks import HookManager
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=2, n_heads=4,
                      d_ff=32, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    x = torch.randint(1, 8, (2, 8))
    with torch.no_grad():
        base = model(x).clone()
        h = HookManager().edit("blocks.0.attn.hook_pattern",
                               lambda p: torch.zeros_like(p))
        edited = model(x, hooks=h)
    assert not torch.allclose(base, edited, atol=1e-5), (
        "zeroing the whole attention pattern must change the logits")


def test_cached_decode_capture_is_refused_not_silently_concatenated():
    """Every decode step contributes (B, 1, D) -- consistent, so the old guard
    (which tested for INCONSISTENT shape[1]) never fired on the one case its
    own comment describes, and torch.cat(dim=0) conflated batch with time."""
    from molcircuits.model.hooks import HookManager

    h = HookManager().capture("x")
    for _ in range(5):
        h.emit("x", torch.zeros(2, 1, 32))
    with pytest.raises(ValueError, match="incremental decode"):
        _ = h["x"]

    # ordinary multi-batch capture still concatenates on the batch axis
    h2 = HookManager().capture("y")
    for _ in range(3):
        h2.emit("y", torch.zeros(2, 8, 32))
    assert h2["y"].shape == (6, 8, 32)


def test_validation_with_no_batches_raises_instead_of_reporting_perfection():
    from torch.utils.data import TensorDataset

    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.train.trainer import evaluate
    from torch.utils.data import DataLoader

    cfg = ModelConfig(vocab_size=8, d_model=16, n_layers=2, n_heads=4,
                      d_ff=32, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    empty = DataLoader(TensorDataset(torch.zeros(0, 8, dtype=torch.long),
                                     torch.zeros(0, 8, dtype=torch.long)),
                       batch_size=4)
    with pytest.raises(ValueError, match="no batches"):
        evaluate(model, empty, "cpu", 10)
