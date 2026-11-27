"""Inputs, completion checks and shared-writer coordination for grid summaries."""
import fcntl
from functools import lru_cache
import json
from pathlib import Path

PRIMARY = {
    "03_saes": "saes/sae_summary.json",
    "04_pointers": "pointers/pointer_results.json",
    "05_valence": "valence/valence_results.json",
    "13_valence_geometry": "valence_geom/valence_geometry.json",
    "03_dictionary_stats": "saes/dictionary_stats.json",
    "06_fragments": "fragments/fragment_results.json",
    "07_steering": "steering_rms/steering_results.json",
    "12_atlas": "atlas/feature_atlas.json",
    "17_fragment_controls": "fragment_controls/fragment_alignment_controls_v1.json",
    "17_fragment_wsd_controls": "fragment_controls/fragment_alignment_controls_v2.json",
    "10_stability": "stability/feature_stability.json",
    "11_robustness": "robustness/augmentation_robustness.json",
    "08_moleculeace": "moleculeace/moleculeace_results.json",
    "08b_moleculeace_dimcontrol": "moleculeace_dimcontrol/moleculeace_dimcontrol.json",
    "09_tdc": "tdc/tdc_results.json",
    "14_steerprop": "steerprop/steering_property.json",
    "14_steermotif": "steermotif/steering_property.json",
}
CROSS = {
    "15_wsd_inference": ("wsd_inference.json", ["06_fragments"]),
    "16_steering_outcomes": (
        "steering_outcomes/steering_outcomes_summary.json",
        ["14_steerprop", "14_steermotif"],
    ),
    "18_claim_summary": (
        "claim_summary/claim_summary.json",
        [
            "06_fragments",
            "12_atlas",
            "14_steerprop",
            "14_steermotif",
            "17_fragment_controls",
            "17_fragment_wsd_controls",
            "08_moleculeace",
            "08b_moleculeace_dimcontrol",
            "09_tdc",
        ],
    ),
}

def prerequisites(art, name):
    files = []
    for stage in CROSS[name][1]:
        directory, filename = PRIMARY[stage].split("/")
        for seed in range(3):
            files.extend((art / f"{directory}_seed{seed}" / filename, art / ".done" / f"{stage}_seed{seed}"))
    return files


def timestamps(paths):
    try:
        return tuple(p.stat().st_mtime_ns for p in paths)
    except FileNotFoundError:
        return ()


@lru_cache(maxsize=32)
def valid_document(path, name, mtime):
    try:
        with path.open() as stream:
            doc = json.load(stream)
        if name == "15_wsd_inference":
            return doc["seeds"] == [0, 1, 2] and len(doc["comparators"]) == 3
        if name == "16_steering_outcomes":
            keys = {f"{stem}_seed{seed}" for stem in ("steermotif", "steerprop") for seed in range(3)}
            return (set(doc["source_sha256"]) == keys and bool(doc["comparisons"]) and bool(doc["dose_response"]))
        return bool(doc["stage_status"]) and set(doc["stage_status"].values()) == {"available"}
    except (ValueError, TypeError, AttributeError, KeyError, OSError):
        return False


def outputs(art, name):
    main = art / CROSS[name][0]
    if name == "16_steering_outcomes":
        return [main, *(main.parent / f"{stem}_seed{s}/steering_property.json"
                        for stem in ("steerprop", "steermotif") for s in range(3))]
    if name == "18_claim_summary":
        return [main, main.with_suffix(".md")]
    return [main]


def state(art, name):
    source_times = timestamps(prerequisites(art, name))
    if not source_times:
        return "waiting"
    result = outputs(art, name)
    produced = timestamps(result)
    stamp = timestamps([art / ".done" / f"{name}_seedall"])
    if (produced and stamp and min(produced + stamp) >= max(source_times)
            and valid_document(result[0], name, produced[0])):
        return "complete"
    return "ready"


def run_summary(args, name, compute, canonical=True):
    """Serialize production writers, including calls from older shell processes."""
    art = Path(args.artifacts).resolve()
    target = outputs(art, name)[0]
    expected_out = target if name == "15_wsd_inference" else target.parent
    grid = art.parent.name == "grid" and (art / ".done").is_dir()
    if not (grid and canonical and args.out and Path(args.out).resolve() == expected_out):
        return compute(args)
    marker = art / ".done" / f"{name}_seedall"
    # Current shell workers hold marker.lock before calling this script. Legacy
    # shell workers do not; a distinct inner lock coordinates both entry points.
    with marker.with_suffix(".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = state(art, name)
        if status == "complete":
            return
        inputs = prerequisites(art, name)
        before = timestamps(inputs)
        if not before:
            raise RuntimeError(f"{name}: prerequisite outputs and completion markers are still pending")
        compute(args)
        files = outputs(art, name)
        produced = timestamps(files)
        if (not produced or not valid_document(files[0], name, produced[0])
                or before != timestamps(inputs) or min(produced) < max(before)):
            raise RuntimeError(f"{name}: inputs changed or summary outputs are incomplete")
        marker.touch()
