# Run on a remote cluster

Keep downloads, preparation, and fitting on a compute node. Run the lightweight viewer there too. The templates assume Slurm and an existing virtual environment in the repository root.

## Prepare results

From the repository root, create a Python environment on an appropriate compute node, then install the requirements. Follow your cluster's policy for package installation and outbound network access.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p logs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION scripts/run.sbatch
```

The run job downloads one recording, prepares the trial matrix, fits the models, and writes the static viewer bundle under `results/791319847/`. Its source download is approximately 2.3 GB. Jobs and logs report real progress; a queued job has not completed the experiment.

## View results

Submit the viewer from the repository root:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION scripts/serve.sbatch
```

Read `logs/viewer-JOB_ID.log` to find the compute hostname. Keep an SSH tunnel open **on your own computer**:

```bash
ssh -J LOGIN_ALIAS -L 8502:127.0.0.1:8502 USER@COMPUTE_HOST -N
```

This uses the login node as a jump host; the tunnel ends on the compute node where the viewer binds to loopback. Follow your cluster's SSH rules. If the alias contains identity or algorithm settings, those may also need to be configured for the compute host; private connection examples belong in ignored `private/` files.

Open [the viewer on your computer](http://127.0.0.1:8502). `localhost` on the cluster is not your computer. Do not bind the viewer to `0.0.0.0` or expose the repository directory.

The viewer job has a four-hour limit. Start it again when needed. Stop an owned job with `scancel JOB_ID`; closing the local tunnel stops access but does not cancel the Slurm job.

## No live server needed

The simplest route is to copy `results/791319847/neural-image-decoder.html` to your computer and open it directly. It contains all scripts, styles, results, and images. It works offline without Python or a server.

You can copy only the generated result folder to your computer. It contains the page, images, and small JSON files; it does not need the 2.3 GB recording.

Serve that folder locally with:

```bash
python3 -m http.server 8502 --bind 127.0.0.1 --directory PATH_TO_RESULT_FOLDER
```

Opening `index.html` as a `file://` URL does not work reliably because browsers restrict loading its JSON. No server process launched by an assistant is promised to persist; use Slurm for remote serving or start the local command yourself.
