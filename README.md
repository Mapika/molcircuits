# MolCircuits

Code and archived numerical results for **Molecular Graphs and String-Syntax Circuits in Chemical Language Models** (Márk Marosi, Kristóf Váradi, Mátyás Antal, and Péter Antal — Budapest University of Technology and Economics).

## Overview

`molcircuits` is a PyTorch library and experiment suite for training and mechanistically analyzing chemical language models trained on SMILES strings:

- **Controlled Notation & Scale Grid:** Train matched canonical-SMILES and randomized-SMILES causal transformers (4.2M, 25.2M, and 113.3M parameters; 3 seeds each) on a deterministic Murcko-scaffold split of ZINC20 (`molcircuits.model`, `molcircuits.train`).
- **Graph Invariance & Chemical Hierarchy:** Measure atom-aligned cross-traversal invariance, graph-vs-string distance geometry, multi-head ring-closure circuits, RDKit-validated valence tracking, nonlocal topology probes, and mid-generation physicochemical property readouts (`molcircuits.chem`, `molcircuits.analysis`).
- **Sparse Autoencoders & Downstream Transfer:** Train layer-wise TopK SAEs, evaluate held-out fragment localization (WSD) against width-matched and random controls, and benchmark embeddings on MoleculeACE, TDC ADMET, and Biogen cADME (`molcircuits.sae`, `molcircuits.atlas`, `molcircuits.downstream`).
- **Activation Steering & Logit Control:** Diagnose residual-steering failure modes (norm growth, directional rotation, syntax disruption, and bounded-property sign inversion) and evaluate decision-gated steering, output-level logit biasing (`BoT-Ridge`, `logit_unemb`), and Best-of-$N$ filtering.
- **External Model Validation:** Run the same mechanistic probes and causal ablations across five public HuggingFace models (`gpt2_zinc_87m`, `ChemBERTa-77M-MLM`, `ChemBERTa-77M-MTR`, `MoLFormer-XL-both-10pct`, and `ChemBERTa-zinc-base-v1`).

## Installation

Requires Python 3.10–3.12.

```bash
git clone https://github.com/Mapika/molcircuits.git
cd molcircuits
pip install -e ".[dev]"
```

For TDC ADMET benchmarks, install the optional `tdc` extra:

```bash
pip install -e ".[tdc]"
```

### Data & Checkpoint Paths

Bulk datasets and model checkpoints live outside the repository and are configured via environment variables (see `molcircuits/paths.py`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOLCIRCUITS_ROOT` | `runs` | Base directory for datasets and run outputs |
| `MOLCIRCUITS_DATA` | `$MOLCIRCUITS_ROOT/data` | SMILES splits, tokenizer, and packed binary shards |
| `MOLCIRCUITS_ARTIFACTS` | `$MOLCIRCUITS_ROOT/artifacts` | Model checkpoints, SAEs, and activation caches |
| `MOLCIRCUITS_TOKENIZER` | *(unset)* | Optional `tokenizer.json` override |
| `MOLCIRCUITS_MANIFEST` | `results/manifest.jsonl` | Provenance manifest path |

## Repository Structure

```text
molcircuits/
├── pyproject.toml              # Package metadata and dependencies
├── molcircuits/                # Core Python package
│   ├── chem/                   # SMILES tokenizer, syntax events, valence FSM, SMARTS
│   ├── model/                  # Causal MolecularTransformer, hook manager, sampling
│   ├── train/                  # Murcko-scaffold data packing, Muon/AdamW trainer
│   ├── sae/                    # TopK / ReLU sparse autoencoders
│   ├── analysis/               # Pointers, patching, valence probes, WSD, steering
│   ├── atlas/                  # Deterministic molecule splits and feature-atlas core
│   ├── downstream/             # MoleculeACE, TDC ADMET, Biogen cADME, ECFP4 baselines
│   ├── utils/                  # Atomic JSON I/O, provenance manifest, staleness checks
│   └── paths.py                # Environment-driven path resolution
├── experiments/                # End-to-end training, evaluation, and audit scripts
│   ├── 01_build_datasets.py
│   ├── 02_train_transformer.py
│   ├── 03_train_saes.py
│   ├── 04_graph_invariance.py
│   ├── 05_chemical_hierarchy.py
│   ├── 06_syntax_and_valence.py
│   ├── 07_sparse_and_downstream.py
│   ├── 08_steering_diagnostics.py
│   ├── 09_constructive_steering.py
│   ├── 10_external_models.py
│   └── audit_*.py
├── results/                    # Archived numerical JSON summaries for all 23 models
└── tests/                      # Self-contained unit and smoke tests
```

## Usage

Each stage in `experiments/` exposes a standalone CLI (`python experiments/<script>.py --help`):

1. **Datasets & training (`01`–`03`):** Build the Murcko-scaffold ZINC20 canonical and randomized SMILES splits (`01_build_datasets.py`), train the 18-model transformer grid (`02_train_transformer.py`), and fit TopK SAEs (`03_train_saes.py`).
2. **Mechanistic analysis (`04`–`06`):** Run cross-traversal invariance and graph-distance probing (`04_graph_invariance.py`), four-level chemical hierarchy probing (`05_chemical_hierarchy.py`), and ring/valence causal patching (`06_syntax_and_valence.py`).
3. **Dictionaries, downstream benchmarks & steering (`07`–`09`):** Evaluate WSD fragment localization, MoleculeACE, and TDC/Biogen ADMET (`07_sparse_and_downstream.py`), residual-steering diagnostics (`08_steering_diagnostics.py`), and logit-level / decision-gated control (`09_constructive_steering.py`).
4. **Public pretrained models & control audits (`10`, `audit_*`):** Run cross-architecture evaluations on the five HuggingFace models (`10_external_models.py`) and additional causal/randomization audits (`audit_*.py`).

Archived numerical outputs from all runs are stored as lightweight JSON files under `results/`.

## Testing

Unit and smoke tests run without external datasets or checkpoints:

```bash
pytest -q
```
