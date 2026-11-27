"""Event-specificity controls must be matched to the events they baseline."""
from molcircuits.analysis.pointers import build_event_index
from molcircuits.chem.tokenizer import SmilesTokenizer

SMILES = ["C1CCCCC1CC(C)C", "CC(C)(C)c1ccccc1C2CCCC2", "CCOC(=O)C1CCN(C)CC1",
          "c1ccc2c(c1)CCCC2", "CC(=O)Nc1ccc(O)cc1", "C1CC1CC(C)CC2CCC2",
          "CN(C)C(=O)c1ccc(Cl)cc1", "OCC1CCC(CC1)C(F)(F)F"] * 8


def _tok():
    return SmilesTokenizer.from_corpus(SMILES)


def test_controls_come_only_from_molecules_that_kept_an_event():
    """The bug this guards: controls were drawn from every molecule before the
    event cap, so a third of them baselined molecules with no event at all."""
    tok = _tok()
    ev, _, ctrl = build_event_index(SMILES, tok, 128, "ring", max_events=20, seed=0)
    ev_mols = {b for b, _ in ev}
    ctrl_mols = {b for b, _ in ctrl}
    assert ctrl_mols <= ev_mols
    assert ctrl_mols


def test_controls_are_weighted_by_event_count_not_per_molecule():
    tok = _tok()
    ev, _, ctrl = build_event_index(SMILES, tok, 128, "ring", max_events=20,
                                    seed=0, n_controls_per_mol=3)
    per_mol_events = {}
    for b, _ in ev:
        per_mol_events[b] = per_mol_events.get(b, 0) + 1
    per_mol_ctrl = {}
    for b, _ in ctrl:
        per_mol_ctrl[b] = per_mol_ctrl.get(b, 0) + 1
    # A molecule contributing more events must not contribute fewer controls.
    for b, n in per_mol_events.items():
        other = [(m, k) for m, k in per_mol_events.items() if k < n]
        for m, _ in other:
            assert per_mol_ctrl.get(b, 0) >= per_mol_ctrl.get(m, 0)


def test_ring_and_branch_do_not_share_one_generic_control_pool():
    """They did: the saved control baseline margins were byte-identical."""
    tok = _tok()
    ring = build_event_index(SMILES, tok, 128, "ring", max_events=20, seed=0)[2]
    branch = build_event_index(SMILES, tok, 128, "branch", max_events=20, seed=0)[2]
    assert ring != branch


def test_no_control_position_is_an_event_position():
    tok = _tok()
    for kind in ("ring", "branch"):
        ev, _, ctrl = build_event_index(SMILES, tok, 128, kind, max_events=20, seed=0)
        assert not (set(map(tuple, ev)) & set(map(tuple, ctrl)))
