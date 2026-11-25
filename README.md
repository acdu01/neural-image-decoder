# Neural Image Decoder

Predict which image a mouse saw from recorded brain signals. Test how accuracy changes with fewer recorded cells.

This proof of concept uses **real Allen Brain Observatory Visual Coding Neuropixels data**: one mouse, one recording session, four known natural-scene images, and new repeats held out for testing. It decodes image identity from electrical spike counts, not MRI scans.

## What the viewer shows

- A cell-budget slider with saved models for 1, 5, 10, 25, 50, 100, 200, and all available cells. Budgets larger than the available pool are omitted.
- The true picture, the model's guess, and assigned probabilities for each test trial.
- Spike counts for the selected cells, plus accuracy and a confusion matrix.
- A shuffled-training-label control and a region-removal comparison with matched random removal.

The viewer reads saved results. Moving a slider does not retrain models or download the source data.

For a demo with no server or SSH tunnel, open the generated `results/791319847/neural-image-decoder.html` in any modern browser. This self-contained file includes the real results and four stimulus images. The source recording stays on the compute host.

## Quick start

Use Python 3.10 or newer. A CPU is enough; no GPU or AllenSDK installation is needed. Allow about **3 GB for the source recording**, plus the Python environment. Data stays on the machine where you run these commands.

`requirements.lock.txt` records the exact package versions used for the verified Python 3.12 run. `requirements.txt` allows compatible releases on other supported Python versions.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python neural_decoder.py run
python neural_decoder.py serve
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe neural_decoder.py run
.\.venv\Scripts\python.exe neural_decoder.py serve
```

Open [the local viewer](http://127.0.0.1:8502). The server listens only on `127.0.0.1`. For an HPC cluster, use the [remote instructions](docs/REMOTE.md), not the login node for computation.

To run stages separately:

```bash
python download.py
python neural_decoder.py prepare
python neural_decoder.py evaluate
python neural_decoder.py serve --port 8502
```

After a viewer-only edit, rebuild both viewer formats without retraining:

```bash
python neural_decoder.py export
```

Downloads resume from `.part` files. Completed downloads are reused. The raw manifest records the source URL, byte count, and local SHA-256. This checksum records the downloaded file; it is not an upstream authenticity signature.

## Experiment design

| Setting | Default |
|---|---|
| Session | 791319847 |
| Natural-scene frame IDs | 0, 30, 60, 90 |
| Cells | Visual cortex regions beginning with `VIS` |
| Feature | Spikes from 30–250 ms after image onset |
| Window boundary | Stops at image offset; half-open interval |
| Training | Earlier complete natural-scene blocks |
| Testing | Final complete natural-scene block |
| Model | StandardScaler + logistic regression, C=1 |
| Cell subsets | Five deterministic, nested random sets |
| Control | 20 within-block shuffles of training labels |
| Main score | Balanced accuracy; chance is 25% for four images |

Quality filters require valid electrode data, a named brain location, `good` quality when that field exists, amplitude cutoff ≤ 0.1, presence ratio ≥ 0.95, and ISI violations ≤ 0.5. These are recording-quality filters, not image-based feature selection.

The image IDs are fixed before scoring. Scaling fits only training data. No test label influences the selected cells, feature scaling, or model fitting. The last block remains the same across cell budgets. All repeats of a test presentation stay together; this POC never splits a presentation into train and test windows.

Each cell budget trains a fresh model. It measures performance with fewer available recorded cells, **not** sudden sensor failure in a fixed model. The region experiment similarly retrains after removing inputs. Removing a recorded input does not silence a neuron and is not a causal experiment.

The chart's vertical bars show the score range across sampled cell sets. They are not confidence intervals over mice or trials. Probabilities are uncalibrated model outputs. A shuffled-label score is a sanity check, not proof that all confounds are absent.

## Scope and limits

This is classification among images already seen during training. It does not reconstruct pictures, recognize unseen images, identify individual objects, read thoughts, or show transfer to another mouse. Adjacent stimuli and animal state can still affect responses. The short window avoids counting spikes after the next image starts, but does not remove all effects of preceding images.

One session is enough to test the pipeline and make the demo. Scientific claims would require more sessions, mice, held-out evaluations, and predeclared analyses. Do not pick the best image set or cell seed after inspecting test scores and present it as an unbiased result.

## Files

```text
download.py           Public Allen API queries, resumable downloads
neural_decoder.py     NWB reading, preparation, evaluation, local serving
viewer/               Buildless HTML/CSS/JavaScript viewer
tests/                Small fixtures and scientific-invariant tests
scripts/              Portable Slurm job templates
data/raw/             Ignored source recordings, images, and manifests
data/prepared/        Ignored trial-by-cell count matrix and metadata
results/              Ignored viewer bundles and score summaries
private/              Ignored personal cluster paths and job settings
```

To change the fixed images, choose IDs before scoring, then rerun every stage using `run --frames ...`. Different experiments should use different `--data` and `--output` folders to preserve previous results.

```bash
python neural_decoder.py run --frames 0 30 60 90 --seeds 5 --permutations 20
python -m pytest -q
```

An optional real-browser check verifies the default four-image offline demo, slider, trial navigation, JSON download, and narrow layout. It requires Playwright and its Chromium browser, separate from the runtime dependencies:

```bash
python -m pip install playwright
python -m playwright install chromium
python tests/browser_check.py results/791319847/neural-image-decoder.html --output results/browser-check
```

## Data credit

Source: [Allen Brain Observatory Visual Coding](https://observatory.brain-map.org/visualcoding/) and its [Neuropixels documentation](https://allensdk.readthedocs.io/en/stable/visual_coding_neuropixels.html). The [Allen quick start](https://allensdk.readthedocs.io/en/stable/_static/examples/nb/ecephys_quickstart.html) includes a related image-classification example. This project adds a small browser demo and explicit cell-budget and region-removal comparisons.

Data and stimulus images remain subject to [Allen Institute terms](https://alleninstitute.org/legal/terms-use/). They are downloaded at runtime, not committed to this repository. Review those terms before redistributing a result bundle containing stimulus images.
