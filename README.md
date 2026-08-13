# slar: SIREN for LArTPC photon transport

`slar` trains sinusoidal representation networks (SIRENs) as compact,
continuous surrogates for liquid-argon TPC photon libraries. Given a position
in detector coordinates, a trained model predicts the visibility of that point
to each optical detector.

The package includes the SIREN layers, PhotonLib-backed datasets, visibility
transforms, optimizer and scheduler resume support, checkpoint utilities, and a
command-line training entry point.

## Installation

`slar` requires Python 3.10 or newer and PyTorch. The 0.2.0 release candidate is
pinned to the compatible PhotonLib release commit, so a checkout can be
installed reproducibly with:

```bash
git clone https://github.com/CIDeR-ML/siren-lartpc.git
cd siren-lartpc
python -m pip install .
```

For development:

```bash
python -m pip install -e . pytest
pytest -q
```

The pinned PhotonLib revision is
[`5f734f4`](https://github.com/CIDeR-ML/photonlib/commit/5f734f43218b65951001c7c02cc2f95767056da8).
Replace the direct Git requirement with the final PhotonLib tag once 0.2.0 is
released.

## Training

Training needs a PhotonLib HDF5 file and a YAML configuration. Two example
configurations ship with the package:

```bash
python -c "from slar.utils import list_config; print(list_config())"
python -c "from slar.utils import get_config; print(get_config('icarus_train'))"
```

Start a run with a configuration name or path:

```bash
train-siren.py icarus_train --device=cpu --max_epochs=1
```

Common command-line overrides include `--device`, `--lr`, `--max_epochs`,
`--max_iterations`, `--logdir`, and `--ckpt_file`. To resume both model and
optimizer/scheduler state:

```bash
train-siren.py path/to/train.yaml \
  --resume=True \
  --ckpt_file=path/to/checkpoint.ckpt
```

Checkpoints use PyTorch's restricted `weights_only=True` loader. Keep checkpoint
metadata to tensors and native Python scalar/container types; do not load
untrusted legacy pickle checkpoints.

## Background

Photon libraries replace expensive per-photon Monte Carlo transport with a
lookup table of detection probabilities. A SIREN learns a continuous field
from that table, reducing storage and recovering spatial continuity and useful
input gradients. See the original [SIREN project](https://www.vincentsitzmann.com/siren/)
and the [LArTPC photon-transport study](https://arxiv.org/abs/2211.01505).

API documentation is maintained on
[Read the Docs](https://siren-lartpc.readthedocs.io/). The project is released
under the [MIT License](LICENSE).
