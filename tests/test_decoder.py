"""Scientific invariants and a small NWB fixture; no remote data required."""
import json
import h5py
import numpy as np
from PIL import Image
import pytest

from neural_decoder import spike_counts, split_by_block, select_units, stimulus_rows, prepare, evaluate, fit_model


def test_half_open_spike_windows():
    counts = spike_counts([0, .03, .10, .25, .30, .50], [.03, .25], [.25, .50])
    np.testing.assert_array_equal(counts, [2, 2])


def test_reject_unsorted_spikes_and_bad_windows():
    with pytest.raises(ValueError, match="sorted"):
        spike_counts([.2, .1], [0], [1])
    with pytest.raises(ValueError, match="positive"):
        spike_counts([.1], [.2], [.2])


def test_entire_last_block_held_out():
    y = np.tile([0, 1, 0, 1], 3)
    blocks = np.repeat([9, 10, 13], 4)
    train, test = split_by_block(y, blocks)
    assert set(blocks[train]) == {9, 10}
    assert set(blocks[test]) == {13}
    assert not set(train) & set(test)


def test_missing_test_class_fails():
    with pytest.raises(ValueError, match="Every image"):
        split_by_block([0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 2, 2])


def test_no_independent_block_fails():
    with pytest.raises(ValueError, match="at least two"):
        split_by_block([0, 1], [9, 9])


def test_scaler_fits_training_only():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1000, 2000], [2000, 1000]], dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1])
    model, _, _ = fit_model(X, y, np.arange(4), np.arange(4, 6), np.arange(2))
    np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, [.5, .5])


@pytest.fixture
def fixture_data(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "images").mkdir()
    session = 123
    y = np.tile([0, 30, 60, 90], 12)
    starts = np.arange(len(y), dtype=float)
    blocks = np.repeat([9, 10, 13], 16)
    with h5py.File(raw / f"session_{session}.nwb", "w") as f:
        f.create_dataset("identifier", data=str(session))
        g = f.create_group("intervals/natural_scenes_presentations")
        for name, values in {"start_time": starts, "stop_time": starts + .25, "frame": y,
                             "stimulus_block": blocks, "id": np.arange(len(y))}.items():
            g[name] = values
        g["stimulus_name"] = np.repeat(b"natural_scenes", len(y))
        units = f.create_group("units")
        units["id"] = np.arange(4) + 100
        units["peak_channel_id"] = np.arange(4) + 200
        units["quality"] = np.repeat(b"good", 4)
        units["amplitude_cutoff"] = np.repeat(.01, 4)
        units["presence_ratio"] = np.repeat(.99, 4)
        units["isi_violations"] = np.repeat(.01, 4)
        spikes = [np.sort(np.concatenate([starts[y == label] + .05, starts[y == label] + .10])) for label in [0, 30, 60, 90]]
        units["spike_times"] = np.concatenate(spikes)
        units["spike_times_index"] = np.cumsum([len(s) for s in spikes])
        electrodes = f.create_group("general/extracellular_ephys/electrodes")
        electrodes["id"] = np.arange(4) + 200
        electrodes["location"] = np.array([b"VISp", b"VISp", b"VISl", b"VISl"])
        electrodes["valid_data"] = np.ones(4, dtype=bool)
    (raw / f"session_{session}.json").write_text(json.dumps({"session_id": session, "synthetic_test_fixture": True}))
    for frame in [0, 30, 60, 90]:
        Image.new("L", (20, 20), frame).save(raw / "images" / f"{frame}.tiff")
    return tmp_path, session


def test_quality_filters_exclude_bad_units(fixture_data):
    root, session = fixture_data
    with h5py.File(root / "data" / "raw" / f"session_{session}.nwb", "r+") as f:
        f["units/amplitude_cutoff"][0] = .2
        indices, ids, regions = select_units(f)
        assert 100 not in ids
        assert len(indices) == 3


def test_complete_pipeline_and_saved_predictions(fixture_data):
    root, session = fixture_data
    prepare(session, root / "data")
    output = evaluate(session, root / "data", root / "results", seeds=2, permutations=3)
    result = json.loads((output / "results.json").read_text())
    assert result["metadata"]["train_blocks"] == [9, 10]
    assert result["metadata"]["test_block"] == 13
    assert result["detail"]["4"]["accuracy"] == 1
    assert result["chance"] == .25
    assert len(result["shuffled_scores"]) == 3
    for budget, detail in result["detail"].items():
        assert len(detail["cells"]) == int(budget)
        np.testing.assert_allclose(np.sum(detail["probabilities"], axis=1), 1, atol=2e-6)
        assert np.sum(detail["confusion"]) == 16
    assert {r["removed"] for r in result["region_results"]} == {2}
    assert (output / "index.html").exists()
    assert (output / "images/0.png").exists()
    offline = (output / "neural-image-decoder.html").read_text()
    assert "window.NEURAL_RESULTS=" in offline
    assert "data:image/png;base64," in offline
    assert '<script src="app.js"' not in offline
    assert '<link rel="stylesheet"' not in offline


def test_requested_image_must_exist(fixture_data):
    root, session = fixture_data
    with h5py.File(root / "data" / "raw" / f"session_{session}.nwb") as f:
        with pytest.raises(ValueError, match="missing"):
            stimulus_rows(f, [0, 999])
