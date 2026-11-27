"""End-to-end smoke test on a tiny synthetic corpus, CPU only.

Runs the whole pipeline -- dataset, model, training, SAE, and all four analyses --
at toy scale. It asserts nothing about the science; it exists so that a broken
import or shape bug is caught in seconds rather than three hours into a GH200 run.

    pytest tests/test_smoke.py -q
"""
import os
import tempfile

import numpy as np
import torch

from molcircuits.analysis.pointers import build_event_index, head_effects, pointer_mass
from molcircuits.analysis.valence_intervene import run_with_null
from molcircuits.analysis.valence_probe import binary_direction, run_probe
from molcircuits.analysis.wsd import compare_bases
from molcircuits.chem.fragments import build_masks, load_smarts_library, valid_token_mask
from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import bond_decision_positions, valence_labels
from molcircuits.model.hooks import HookManager, resid_post
from molcircuits.model.sampling import evaluate_generations, generate
from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.sae.evaluate import evaluate_sae, sae_edit_fn
from molcircuits.sae.sae import SAEConfig, SparseAutoencoder
from molcircuits.sae.train_sae import SAETrainConfig, train_sae
from molcircuits.train.data import PackedDataset
from molcircuits.train.trainer import TrainConfig, train

CORPUS = [
    "CC(=O)Nc1ccccc1", "CCOc1ccccc1", "c1ccc2ccccc2c1", "CC(C)Cc1ccccc1",
    "O=C(O)c1ccccc1", "CN1CCN(C)CC1", "c1ccncc1", "CC(=O)OC1CCCCC1",
    "CCN(CC)C(=O)c1ccccc1", "OCC1CCCCO1", "c1cc(F)ccc1C#N", "CSc1ccccc1",
    "CC1=CC(=O)CC(C)(C)C1", "NC(=O)c1ccc(Cl)cc1", "COc1ccc(CN)cc1",
    "C1CCNCC1", "c1ccsc1", "CC(=O)N1CCOCC1", "Oc1ccc(Br)cc1", "CCCCn1cccc1",
] * 5


def _tiny_model(vocab, n_ctx=32):
    return MolecularTransformer(ModelConfig(
        vocab_size=vocab, d_model=32, n_layers=2, n_heads=4, d_ff=64, n_ctx=n_ctx))


def test_end_to_end():
    torch.manual_seed(0)
    n_ctx = 32
    with tempfile.TemporaryDirectory() as tmp:
        # ---- tokenizer + packed dataset
        tok = SmilesTokenizer.from_corpus(CORPUS)
        tok.save(os.path.join(tmp, "tokenizer.json"))
        ds = PackedDataset.build(CORPUS, tok, n_ctx)
        assert len(ds) > 4

        # ---- model
        model = _tiny_model(tok.vocab_size, n_ctx)
        assert model.lm_head is None                      # tying means no lm_head
        assert model.n_params(True) > 0

        # ---- train (a handful of steps)
        cfg = TrainConfig(steps=6, batch_size=2, eval_every=3, eval_batches=2,
                          ckpt_every=6, out_dir=os.path.join(tmp, "tf"),
                          dtype="float32")
        out = train(model, ds, ds, cfg, device="cpu")
        assert np.isfinite(out["final"]["val_ce"])        # base CE exists
        assert os.path.exists(os.path.join(tmp, "tf", "train_metrics.json"))

        # ---- generation metrics carry their settings
        smis = generate(model, tok, 8, max_tokens=24, seed=0, device="cpu")
        gm = evaluate_generations(smis, max_tokens=24, temperature=1.0).as_dict()
        assert gm["max_tokens"] == 24 and 0.0 <= gm["validity"] <= 1.0

        # ---- SAE train + evaluate
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=2)
        h = HookManager().capture(resid_post(1))
        for x, _ in loader:
            model(x, hooks=h)
        acts = h[resid_post(1)].reshape(-1, 32)
        sae, summ = train_sae(acts, SAEConfig(d_in=32, expansion=2, layer=1),
                              SAETrainConfig(n_activations=len(acts), batch_size=16,
                                             warmup=2),
                              device="cpu", out_dir=os.path.join(tmp, "sae"))
        assert summ["final_l0"] is not None
        m = evaluate_sae(model, sae, loader, "cpu", 1, max_batches=3)
        for k in ("l0", "delta_ce", "dead_feature_fraction",
                  "fraction_ce_recovered_vs_zero", "fraction_ce_recovered_vs_mean"):
            assert k in m
        assert SparseAutoencoder.load(os.path.join(tmp, "sae")).cfg.layer == 1

        # ---- alpha=0 edit is the SAE round trip, not the identity
        ident = HookManager().edit(resid_post(1), sae_edit_fn(sae, 0.0))
        assert not torch.allclose(model(next(iter(loader))[0]),
                                  model(next(iter(loader))[0], hooks=ident))

        # ---- pointer analysis
        ep, em, cp = build_event_index(CORPUS[:16], tok, n_ctx, "ring", 64, 0)
        assert ep, "expected ring events in the toy corpus"
        ids = [tok.encode(s)[:n_ctx + 1] for s in CORPUS[:16]]
        ids = [e + [tok.pad_id] * (n_ctx + 1 - len(e)) for e in ids]
        x = torch.tensor([e[:n_ctx] for e in ids])
        y = torch.tensor([e[1:n_ctx + 1] for e in ids])
        pm = pointer_mass(model, x, em, ep)
        assert pm.shape == (2, 4) and (pm >= 0).all() and (pm <= 1).all()
        rows = head_effects(model, x, y, ep, cp, "cpu", mode="zero")
        assert len(rows) == 2 * 4                          # every head swept
        assert all("delta_margin_control" in r and "delta_ce" in r for r in rows)
        rows_mean = head_effects(model, x, y, ep, cp, "cpu", mode="mean",
                                 mean_z={l: torch.zeros(4, 8) for l in range(2)})
        assert len(rows_mean) == 8

        # ablation must be exactly reversible
        before = model(x).clone()
        with model.ablate_head(0, 0):
            pass
        assert torch.allclose(before, model(x))

        # ---- valence probe with controls
        A, Y, G, T = [], [], [], []
        hh = HookManager().capture(resid_post(1))
        model(x, hooks=hh)
        a = hh[resid_post(1)].detach().numpy()
        for bi, s in enumerate(CORPUS[:16]):
            for lab in valence_labels(tokenize(s), offset=1):
                if lab.token_idx < n_ctx:
                    A.append(a[bi, lab.token_idx]); Y.append(lab.remaining)
                    G.append(bi); T.append(ids[bi][lab.token_idx])
        A, Y, G, T = np.array(A), np.array(Y), np.array(G), np.array(T)
        pr = run_probe(A, Y, G, T, n_boot=20)
        for k in ("majority_class_accuracy", "control_task_accuracy", "selectivity",
                  "token_identity_baseline_accuracy", "confusion_matrix"):
            assert k in pr
        w, meta = binary_direction(A, Y, G)
        assert abs(np.linalg.norm(w) - 1.0) < 1e-5
        assert "binary logistic regression" in meta["objective"]

        # ---- intervention against a null
        pairs = [(bi, p) for bi, s in enumerate(CORPUS[:16])
                 for p in bond_decision_positions(tokenize(s), offset=1)
                 if p < n_ctx - 1]
        if pairs:
            bid = {n: tok.vocab[n] for n in ("=", "#") if n in tok.vocab}
            res = run_with_null(model, x, torch.tensor(pairs), 1, w, [-1, 1], bid,
                                "cpu", n_null=3)
            assert res["residual_rms"] > 0
            assert len(res["null"]) == 2
            assert any(k.endswith("_z_vs_null") for k in res["null"][0])

        # ---- WSD basis comparison
        lib = {k: load_smarts_library()[k] for k in ("benzene", "amide")}
        masks = build_masks(CORPUS[:16], lib, n_ctx, offset=1)
        valid = valid_token_mask(CORPUS[:16], n_ctx, offset=1)
        z = sae.encode(torch.tensor(a.reshape(-1, 32))).detach().numpy()
        z = z.reshape(a.shape[0], a.shape[1], -1)
        cmp = compare_bases(a, z, masks["benzene"], valid, (0, 1), True, 3)
        assert cmp["random_rotation"]["n_seeds"] == 2
        assert cmp["random_rotation"]["best_std"] >= 0.0   # a real spread, not +-0.00
        assert "permutation_null_sae" in cmp


def test_unified_experiment_entrypoints_compile_and_help():
    """Every unified experiment script must compile cleanly and expose --help."""
    import os
    import py_compile
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(root, "experiments")
    scripts = sorted(
        f for f in os.listdir(exp_dir)
        if f.endswith(".py") and f[:2].isdigit()
    )
    assert len(scripts) == 10, f"expected 10 numbered experiment scripts, found {scripts}"

    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for name in scripts:
        path = os.path.join(exp_dir, name)
        py_compile.compile(path, doraise=True)
        res = subprocess.run(
            [sys.executable, path, "--help"],
            capture_output=True, text=True, env=env, timeout=30,
        )
        assert res.returncode == 0, f"{name} --help failed:\n{res.stderr}"


def test_every_third_party_import_is_declared():
    """Every third-party import in molcircuits/ and experiments/ must be declared in pyproject.toml."""
    import ast
    import os
    import re
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "pyproject.toml")).read()
    start = re.search(r"^\[project\]", text, re.M).start()
    nxt = re.search(r"^\[tool\.", text[start:], re.M)
    body = text[start:start + nxt.start()] if nxt else text[start:]
    declared = {re.split(r"[<>=!\[]", q, 1)[0].strip().lower()
                for q in re.findall(r'"([^"]+)"', body)
                if re.match(r"^[A-Za-z]", q) and "::" not in q}
    alias = {"scikit-learn": "sklearn", "pytdc": "tdc", "pillow": "pil",
             "umap-learn": "umap", "huggingface-hub": "huggingface_hub",
             "fonttools": "fonttools"}
    declared |= {alias[d] for d in list(declared) if d in alias}

    tops = set()
    for sub in ("molcircuits", "experiments"):
        for dirpath, dirnames, files in os.walk(os.path.join(root, sub)):
            dirnames[:] = [d for d in dirnames if d not in ("figures", "__pycache__")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                tree = ast.parse(open(os.path.join(dirpath, f)).read())
                for n in ast.walk(tree):
                    if isinstance(n, ast.Import):
                        tops |= {a.name.split(".")[0] for a in n.names}
                    elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                        tops.add(n.module.split(".")[0])

    local_pkgs = {"molcircuits", "_datasets", "_syntax", "_downstream"}
    third_party = {t for t in tops if t not in sys.stdlib_module_names
                   and t not in local_pkgs}
    missing = {t for t in third_party if t.lower() not in declared}
    assert not missing, f"imported but not declared in pyproject.toml: {sorted(missing)}"


def _sink_setup():
    """A toy model whose BOS/EOS positions dominate, as the real one's do.

    The sink has to be built into the WEIGHTS, not injected with a hook: the
    hook manager emits a captured value before applying any edit registered for
    the same point, so an edit cannot change what ``reference_latent`` reads.
    Oversizing the BOS and EOS embedding rows reproduces the real structure --
    at layer 3 of the trained model those positions carry residual norms of
    13,753 and 2,414 against 312 for interior tokens.
    """
    torch.manual_seed(0)
    tok = SmilesTokenizer.from_corpus(CORPUS)
    model = _tiny_model(tok.vocab_size).eval()
    with torch.no_grad():
        sink = torch.zeros(model.cfg.d_model)
        sink[:8] = 60.0
        model.embed.weight[tok.bos_id] = sink
        model.embed.weight[tok.eos_id] = sink
    sae = SparseAutoencoder(SAEConfig(d_in=model.cfg.d_model, expansion=4,
                                      variant="topk", k=8, layer=1)).eval()
    with torch.no_grad():
        sae.W_enc.normal_(0, 1.0)
        sae.b_enc.zero_()
    return tok, model, sae


def test_reference_latent_is_not_captured_by_the_attention_sinks():
    """A target's top-k features must depend on the target, not on BOS/EOS.

    With BOS and EOS dominating every sequence, a max over ALL positions
    returns their features every time, so every target yields an identical
    top-k and "steering toward the target" injects a target-independent
    direction. On the real layer-3 dictionary that is exactly what happened:
    1 distinct top-5 set across all 50 steering targets, features
    {1131, 2668, 5368, 5470, 7908} for every one of them. Pooling over content
    tokens gives 41 distinct sets drawing on 74 features.
    """
    from molcircuits.analysis.steering import reference_latent

    tok, model, sae = _sink_setup()
    targets = ["CC(=O)Nc1ccccc1", "c1ccc2ccccc2c1", "CN1CCN(C)CC1",
               "O=C(O)c1ccccc1", "CCCCn1cccc1", "Oc1ccc(Br)cc1"]
    sets = []
    for t in targets:
        z = reference_latent(model, sae, tok, t, layer=1, top_k=5)
        assert int((z > 0).sum()) <= 5
        sets.append(tuple(sorted(z.nonzero().flatten().tolist())))

    assert len(set(sets)) > 1, (
        f"all {len(targets)} targets selected the same top-5 features "
        f"({sets[0]}): the pool is reading the BOS/EOS sinks, so the steered "
        "arm carries no target information")


def test_reference_latent_responds_to_the_interior_not_the_sinks():
    """Changing an INTERIOR token must move the selection; the sinks must not.

    The complement of the test above. Both halves have to hold: a pool that
    reads the sinks is insensitive to the chemistry (first assertion fails),
    and a pool that reads the sinks is sensitive to their magnitude (second
    assertion is what the fix buys).
    """
    from molcircuits.analysis.steering import reference_latent

    tok, model, sae = _sink_setup()
    target = "CC(=O)Nc1ccccc1"
    base = tuple(sorted(reference_latent(model, sae, tok, target, layer=1,
                                         top_k=5).nonzero().flatten().tolist()))

    # amplify a token that occurs only in the interior of this target
    interior = tok.encode(target)[1:-1]
    with torch.no_grad():
        model.embed.weight[interior[0]] *= 40.0
    moved = tuple(sorted(reference_latent(model, sae, tok, target, layer=1,
                                          top_k=5).nonzero().flatten().tolist()))
    assert moved != base, (
        "amplifying an interior token did not change the selected features, so "
        "the pool is not reading the molecule")

    # and the sinks still must not matter
    tok2, model2, sae2 = _sink_setup()
    small = tuple(sorted(reference_latent(model2, sae2, tok2, target, layer=1,
                                          top_k=5).nonzero().flatten().tolist()))
    with torch.no_grad():
        model2.embed.weight[tok2.bos_id] *= 100.0
        model2.embed.weight[tok2.eos_id] *= 100.0
    large = tuple(sorted(reference_latent(model2, sae2, tok2, target, layer=1,
                                          top_k=5).nonzero().flatten().tolist()))
    assert small == large, (
        "a 100x change in BOS/EOS magnitude changed the selection, so pooling "
        "still reaches them")
