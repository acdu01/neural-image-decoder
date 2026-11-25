"""A small, reproducible image-decoding experiment on Allen Neuropixels data."""
from pathlib import Path
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import shutil

import h5py
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from download import DEFAULT_SESSION, fetch


def text_array(dataset):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]])


def spike_counts(spikes, starts, stops):
    """Count spikes in half-open [start, stop) windows, without boundary reuse."""
    spikes = np.asarray(spikes)
    starts, stops = np.asarray(starts), np.asarray(stops)
    if np.any(stops <= starts):
        raise ValueError("Every counting window must have positive duration")
    if np.any(np.diff(spikes) < 0):
        raise ValueError("Spike times must be sorted")
    return np.searchsorted(spikes, stops, side="left") - np.searchsorted(spikes, starts, side="left")


def split_by_block(labels, blocks):
    """Hold out the last complete stimulus block, for every condition."""
    labels, blocks = np.asarray(labels), np.asarray(blocks)
    unique = np.unique(blocks)
    if len(unique) < 2:
        raise ValueError("Need at least two stimulus blocks for an independent test")
    test = np.flatnonzero(blocks == unique[-1])
    train = np.flatnonzero(blocks != unique[-1])
    classes = np.unique(labels)
    if not np.array_equal(np.unique(labels[train]), classes) or not np.array_equal(np.unique(labels[test]), classes):
        raise ValueError("Every image must appear in both training and test blocks")
    if min(Counter(labels[train]).values()) < 2 or min(Counter(labels[test]).values()) < 2:
        raise ValueError("Each image needs at least two train and two test repeats")
    return train, test


def stimulus_rows(nwb, frames):
    groups = [g for g in nwb["intervals"].values()
              if isinstance(g, h5py.Group) and "frame" in g and "start_time" in g]
    collected = []
    for group in groups:
        n = len(group["start_time"])
        names = text_array(group["stimulus_name"]) if "stimulus_name" in group else np.repeat(group.name, n)
        selected = np.char.find(names.astype(str), "natural_scenes") >= 0
        raw_frames = text_array(group["frame"])
        frame_ids = np.asarray([int(float(v)) if v not in ("null", "nan", "", "None") else -1 for v in raw_frames])
        selected &= np.isin(frame_ids, frames)
        starts, stops = group["start_time"][:], group["stop_time"][:]
        if "stimulus_block" not in group:
            raise ValueError("Natural-scenes table has no stimulus_block column")
        raw_blocks = text_array(group["stimulus_block"])
        ids = group["id"][:]
        for i in np.flatnonzero(selected):
            collected.append((float(starts[i]), float(stops[i]), int(frame_ids[i]),
                              int(float(raw_blocks[i])), int(ids[i])))
    if not collected:
        raise ValueError("No natural-scenes trials matched the requested images")
    collected.sort()
    rows = np.asarray(collected)
    if len(np.unique(rows[:, 0])) != len(rows):
        raise ValueError("Duplicate stimulus starts in NWB intervals")
    if set(rows[:, 2].astype(int)) != set(frames):
        raise ValueError("Some requested images are missing from this session")
    return rows


def select_units(nwb, region_prefix="VIS"):
    units = nwb["units"]
    electrodes = nwb["general/extracellular_ephys/electrodes"]
    channel_ids = electrodes["id"][:]
    locations = text_array(electrodes["location"])
    valid = electrodes["valid_data"][:] if "valid_data" in electrodes else np.ones(len(channel_ids), dtype=bool)
    channels = {int(cid): (str(region), bool(ok)) for cid, region, ok in zip(channel_ids, locations, valid)}
    ids = units["id"][:]
    regions, channel_valid = zip(*(channels.get(int(cid), ("unknown", False)) for cid in units["peak_channel_id"][:]))
    regions = np.asarray(regions)
    mask = np.asarray(channel_valid, dtype=bool)
    mask &= np.isfinite(units["amplitude_cutoff"][:]) & (units["amplitude_cutoff"][:] <= 0.1)
    mask &= np.isfinite(units["presence_ratio"][:]) & (units["presence_ratio"][:] >= 0.95)
    mask &= np.isfinite(units["isi_violations"][:]) & (units["isi_violations"][:] <= 0.5)
    if "quality" in units:
        mask &= text_array(units["quality"]) == "good"
    mask &= ~np.isin(regions, ["unknown", "None", "nan", "", "root"])
    if region_prefix:
        mask &= np.char.startswith(regions, region_prefix)
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        raise ValueError(f"Only {len(indices)} units passed quality/region filters")
    return indices, ids[indices], regions[indices]


def prepare(session=DEFAULT_SESSION, data="data", frames=(0, 30, 60, 90), region_prefix="VIS"):
    data = Path(data)
    raw = data / "raw" / f"session_{session}.nwb"
    target = data / "prepared" / str(session)
    target.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw, "r") as nwb:
        identifier = nwb["identifier"][()]
        identifier = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
        if identifier != str(session):
            raise ValueError(f"NWB identifies session {identifier}, expected {session}")
        rows = stimulus_rows(nwb, frames)
        # 30 ms visual-response latency; stop before the next image appears.
        starts = rows[:, 0] + 0.030
        stops = np.minimum(rows[:, 0] + 0.250, rows[:, 1])
        labels = rows[:, 2].astype(int)
        blocks = rows[:, 3].astype(int)
        selected, ids, regions = select_units(nwb, region_prefix)
        counts = np.zeros((len(rows), len(selected)), dtype=np.int32)
        offsets = nwb["units/spike_times_index"][:]
        for j, unit in enumerate(selected):
            begin = int(offsets[unit - 1]) if unit else 0
            end = int(offsets[unit])
            spikes = nwb["units/spike_times"][begin:end]
            counts[:, j] = spike_counts(spikes, starts, stops)
        train, test = split_by_block(labels, blocks)
        metadata = {
            "session_id": session, "frames": list(map(int, frames)), "region_prefix": region_prefix,
            "units": len(ids), "trials": len(rows), "train_trials": len(train), "test_trials": len(test),
            "train_blocks": np.unique(blocks[train]).tolist(), "test_block": int(blocks[test][0]),
            "regions": dict(Counter(regions)), "window_ms": [30, 250],
            "unit_filter": "valid channel; good quality if present; amplitude_cutoff <= 0.1; presence_ratio >= 0.95; isi_violations <= 0.5",
            "split": "First stimulus blocks train; final complete stimulus block tests.",
            "source": "Allen Brain Observatory Visual Coding Neuropixels",
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
    np.savez_compressed(target / "trials.npz", X=counts, y=labels, blocks=blocks,
                        starts=rows[:, 0], trial_ids=rows[:, 4].astype(int), unit_ids=ids,
                        regions=regions.astype(str), train=train, test=test)
    manifest = json.loads((data / "raw" / f"session_{session}.json").read_text())
    metadata["source_manifest"] = manifest
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return target


def fit_model(X, y, train, test, cells):
    if len(cells) == 0:
        raise ValueError("A decoder needs at least one cell")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))
    model.fit(X[train][:, cells], y[train])
    predictions = model.predict(X[test][:, cells])
    probabilities = model.predict_proba(X[test][:, cells])
    return model, predictions, probabilities


def evaluate(session=DEFAULT_SESSION, data="data", output="results", seeds=5, permutations=20):
    if seeds < 1 or permutations < 1:
        raise ValueError("Use at least one subset seed and one shuffled-label control")
    prepared = Path(data) / "prepared" / str(session)
    metadata = json.loads((prepared / "metadata.json").read_text())
    with np.load(prepared / "trials.npz", allow_pickle=False) as pack:
        X, y, blocks = pack["X"], pack["y"], pack["blocks"]
        unit_ids, regions = pack["unit_ids"], pack["regions"]
        trial_ids, starts = pack["trial_ids"], pack["starts"]
    train, test = split_by_block(y, blocks)
    classes = np.unique(y)
    n_units = X.shape[1]
    budgets = sorted(set(min(v, n_units) for v in [1, 5, 10, 25, 50, 100, 200, n_units]))
    curves, detail = [], {}
    baseline_predictions = None
    for budget in budgets:
        scores = []
        for seed in range(seeds):
            # Nested subsets for each seed. No feature selection uses image labels.
            cells = np.random.default_rng(seed).permutation(n_units)[:budget]
            model, predicted, probabilities = fit_model(X, y, train, test, cells)
            score = float(balanced_accuracy_score(y[test], predicted))
            scores.append(score)
            if seed == 0:
                detail[str(budget)] = {"cells": unit_ids[cells].astype(int).tolist(),
                    "predictions": predicted.astype(int).tolist(),
                    "probabilities": probabilities.round(6).tolist(),
                    "confusion": confusion_matrix(y[test], predicted, labels=classes).tolist(),
                    "accuracy": float(accuracy_score(y[test], predicted)), "balanced_accuracy": score}
                if budget == n_units:
                    baseline_predictions = predicted
        curves.append({"cells": budget, "mean": float(np.mean(scores)),
                       "min": float(min(scores)), "max": float(max(scores)), "scores": scores})
        print(f"{budget:4d} cells: balanced accuracy {np.mean(scores):.3f}", flush=True)
    controls = []
    for seed in range(permutations):
        shuffled = y.copy()
        rng = np.random.default_rng(10_000 + seed)
        for block in np.unique(blocks[train]):
            idx = train[blocks[train] == block]
            shuffled[idx] = rng.permutation(shuffled[idx])
        _, predicted, _ = fit_model(X, shuffled, train, test, np.arange(n_units))
        controls.append(float(balanced_accuracy_score(y[test], predicted)))
    # Compare each removed region against removing the same number of random cells.
    region_results = []
    for region in sorted(set(regions)):
        cells = np.flatnonzero(regions != region)
        if not len(cells):
            continue
        _, predicted, _ = fit_model(X, y, train, test, cells)
        matched = []
        for seed in range(seeds):
            random_cells = np.random.default_rng(20_000 + seed).permutation(n_units)[:len(cells)]
            _, random_pred, _ = fit_model(X, y, train, test, random_cells)
            matched.append(float(balanced_accuracy_score(y[test], random_pred)))
        region_results.append({"region": str(region), "removed": int(n_units-len(cells)),
            "remaining": len(cells), "score": float(balanced_accuracy_score(y[test], predicted)),
            "random_mean": float(np.mean(matched)), "random_min": min(matched), "random_max": max(matched)})
    baseline = float(balanced_accuracy_score(y[test], baseline_predictions))
    result = {"metadata": metadata, "classes": classes.astype(int).tolist(),
              "chance": 1 / len(classes), "seeds": seeds, "curves": curves, "detail": detail,
              "test_labels": y[test].astype(int).tolist(), "test_trial_ids": trial_ids[test].astype(int).tolist(),
              "test_starts": starts[test].tolist(), "test_counts": X[test].astype(int).tolist(),
              "unit_ids": unit_ids.astype(int).tolist(), "regions": regions.tolist(),
              "shuffled_scores": controls, "shuffled_mean": float(np.mean(controls)),
              "permutation_p": (1 + sum(s >= baseline for s in controls)) / (len(controls) + 1),
              "region_results": region_results,
              "method": "StandardScaler + logistic regression, C=1.0, fitted on training blocks only. Each cell budget is retrained. Seed 0 drives the trial viewer.",
              "limits": f"One mouse, one session, {len(classes)} known images. Test trials are new repeats of training images. Scores do not establish unseen-image or cross-mouse decoding. Model probabilities are not calibrated confidence. Removing recorded inputs does not silence cells in the animal.",
              "generated_at": datetime.now(timezone.utc).isoformat()}
    target = Path(output) / str(session)
    target.mkdir(parents=True, exist_ok=True)
    image_dir = target / "images"
    image_dir.mkdir(exist_ok=True)
    for frame in classes:
        with Image.open(Path(data) / "raw" / "images" / f"{frame}.tiff") as image:
            image.convert("RGB").save(image_dir / f"{frame}.png")
    (target / "results.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    summary = {"session_id": session, "units": n_units, "train_trials": len(train),
               "test_trials": len(test), "baseline_balanced_accuracy": baseline,
               "chance": 1 / len(classes), "shuffled_mean": result["shuffled_mean"],
               "curve": curves, "region_results": region_results}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_viewer(target)
    print(json.dumps(summary, indent=2), flush=True)
    return target


def export_viewer(target):
    """Build both a served viewer and a self-contained, offline HTML download."""
    target = Path(target)
    source = Path(__file__).parent / "viewer"
    for name in ["index.html", "app.js", "style.css"]:
        shutil.copyfile(source / name, target / name)
    result = json.loads((target / "results.json").read_text())
    images = {str(frame): "data:image/png;base64," + base64.b64encode(
        (target / "images" / f"{frame}.png").read_bytes()).decode("ascii") for frame in result["classes"]}
    payload = json.dumps(result, allow_nan=False).replace("<", "\\u003c")
    template = (source / "index.html").read_text()
    template = template.replace('<link rel="stylesheet" href="style.css">',
                                "<style>" + (source / "style.css").read_text() + "</style>")
    template = template.replace('<script src="app.js" defer></script>', "")
    scripts = "<script>window.NEURAL_RESULTS=" + payload + ";window.NEURAL_IMAGES=" + json.dumps(images) + ";</script>"
    scripts += "<script>" + (source / "app.js").read_text() + "</script>"
    template = template.replace("</body>", scripts + "</body>")
    (target / "neural-image-decoder.html").write_text(template)
    return target / "neural-image-decoder.html"


def serve(directory, port):
    directory = Path(directory).resolve()
    if not (directory / "results.json").is_file():
        raise ValueError(f"No results.json in {directory}; run evaluate first")
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Viewer: http://127.0.0.1:{port} (SSH tunnel required for remote hosts)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "prepare", "evaluate", "run", "export", "serve"])
    parser.add_argument("--session", type=int, default=DEFAULT_SESSION)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 30, 60, 90])
    parser.add_argument("--region-prefix", default="VIS")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    if len(args.frames) < 2 or len(set(args.frames)) != len(args.frames):
        parser.error("Choose at least two different image frame IDs")
    if args.command in ("fetch", "run"):
        fetch(args.session, args.data, args.frames)
    if args.command in ("prepare", "run"):
        prepare(args.session, args.data, args.frames, args.region_prefix)
    if args.command in ("evaluate", "run"):
        evaluate(args.session, args.data, args.output, args.seeds, args.permutations)
    if args.command == "export":
        print(export_viewer(Path(args.output) / str(args.session)))
    if args.command == "serve":
        serve(Path(args.output) / str(args.session), args.port)


if __name__ == "__main__":
    main()
