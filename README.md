# W-Seminar: Human and CNN/Fly perception of art

This repository contains an experimental project about emotional responses to
artworks. The CNN workflow recognizes EmoArt's annotated dominant-emotion
categories and organizes artworks in a learned, interactive 3D neighborhood
map. It is an exploratory model of dataset annotations, not a measurement of
any individual viewer's emotions. The FlyBrain implementation remains separate.

## CNN workflow overview

1. Each RGB image passes through convolution and pooling layers.
2. The trained network returns a probability distribution over EmoArt's
   dominant-emotion labels and a learned visual embedding.
3. The visualizer joins the normalized embedding and normalized emotion
   probabilities into one representation. K-nearest neighbors (KNN) create a
   similarity graph between artworks.
4. Spectral embedding projects the neighbor graph into three dimensions, and
   spectral clustering assigns KNN groups. The plotted axes are emergent map
   coordinates; they are not hand-assigned valence/arousal coordinates.
5. A new image is recognized by the CNN and placed near its nearest known
   artworks with distance-weighted KNN interpolation.

EmoArt's labels were produced for the dataset and do not stand in for a new
human participant study. EmoArt's 3D location is a data visualization of model
recognition and neighborhood structure, not a biologically validated 3D
measurement of Damasio's theory.

## Setup

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install the CPU-only PyTorch wheels first, then the project's other Python
dependencies, and download/extract EmoArt-5k:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r cnn-requirements.txt
python -m src.CNN.download_data
```

The PyTorch command deliberately uses the official CPU wheel index. A plain
`pip install torch torchvision` can select a CUDA build and install NVIDIA
runtime libraries even when the machine has no NVIDIA GPU. The CNN code checks
whether CUDA is available and uses the CPU when it is not. The current PyTorch
Linux guidance includes Python 3.14, so Python 3.14.8 is within the documented
range.

The Hugging Face image archive is about 2.6 GB and extraction brings total
storage to roughly 5.2 GB. The raw images, annotations, splits, checkpoints, and
visualization outputs are kept out of Git. To fetch only annotations, run:

```bash
python -m src.CNN.download_data --skip-images
```

Dataset provenance and licensing scope are described in
[`EMOART_DATA_LICENSE.md`](EMOART_DATA_LICENSE.md) and
[`src/CNN/DATASET_SOURCES.md`](src/CNN/DATASET_SOURCES.md). The repository's
MIT license covers project code only, not EmoArt images or annotations.

## Train the emotion-recognition CNN

Train the baseline for 15 epochs using the fixed 70/15/15 stratified split:

```bash
python -m src.CNN.train --device cpu
```

Use `--device nvidia` (or `--device cuda`) to request an NVIDIA GPU, and
`--device auto` to select an available CUDA device or fall back to CPU. The
default is `auto`. A forced NVIDIA/CUDA choice raises an error if CUDA is not
available. The CPU-only PyTorch wheel installed above cannot run CUDA; install
a CUDA-enabled PyTorch build from the official selector if you later train on
an NVIDIA system.

Select learning rate and dropout using a small validation-set grid:

```bash
python -m src.CNN.train --tune --epochs 8 --device cpu
```

The tune grid trains four combinations of learning rate and dropout. Its best
checkpoint is selected by validation macro F1. The test split is evaluated
after selection and does not choose hyperparameters. Training supports these
options:

```bash
python -m src.CNN.train --epochs 20 --batch-size 16 --seed 42
```

The persisted membership file is
`data/splits/emoart5k_70_15_15.csv`. Keep it fixed across experiments; use the
same split when comparing color-space variants. Metrics and model files are
written to `runs/cnn/`, including `best_model.pt`, `labels.json`, and
`training_report.json`. Class-weighted loss helps with the strong emotion-class
imbalance. Report balanced accuracy and macro F1 alongside ordinary accuracy.

The data package includes rare emotion labels with very few examples. Split
creation moves a sample into the training partition when necessary so that
every label is learnable; invalid saved splits raise a clear error. A random
image split can still place works by the same artist or style in more than one
partition, so the score does not establish generalization to unseen artists or
cultures.

This small CNN can run on a CPU. An Intel i7 with 32 GB RAM and NVMe storage is
adequate for a prototype, but training time depends on the exact i7 model and
thread count. Start with `python -m src.CNN.train --epochs 2` to measure one run
before using the four-trial tuning grid. The model and resized batches are
modest in memory; the 5.2 GB dataset footprint is the larger storage item.

## Recognize one artwork

After training, show the top five predicted EmoArt emotion categories:

```bash
python -m src.CNN.predict path/to/artwork.jpg --device cpu
python -m src.CNN.predict path/to/artwork.jpg --top-k 3 --device cpu
```

The printed probabilities represent the network's confidence over dataset
labels. They are not calibrated estimates of a viewer's subjective feelings.

## Create and explore the interactive 3D map

Generate a mouse-rotatable map from known training and validation artworks:

```bash
python -m src.CNN.visualize --device cpu
```

Optionally place a new image into the map using its nearest known artworks:

```bash
python -m src.CNN.visualize --image path/to/artwork.jpg --neighbors 12 --device cpu
```

Open `runs/visualization/emotion_space.html` in a browser. Drag to rotate,
scroll to zoom, and hover over points for the image name, dataset emotion, CNN
recognition, style, split, KNN group, and confidence. The KNN group is shown by
point color. A new query is marked with a diamond and placed by inverse-distance
weighted interpolation from its nearest neighbors. Coordinates and metadata
are also saved to `emotion_space.csv`.

The map uses only training and validation images as known points. The held-out
test set stays out of this exploratory map. KNN and spectral clustering are
unsupervised organization steps over the CNN representation and recognition
probabilities; the 3D axis directions have no intrinsic psychological meaning.

## Code structure

- `src/CNN/data.py`: annotation validation, reproducible split, and image data.
- `src/CNN/model.py`: emotion classifier and learned image representation.
- `src/CNN/train.py`: baseline training, grid tuning, validation selection,
  and final test metrics.
- `src/CNN/predict.py`: emotion-label inference for one artwork.
- `src/CNN/visualize.py`: KNN graph, 3D projection, clustering, and Plotly HTML.
- `src/CNN/download_data.py`: Hugging Face archive download and safe extraction.
- `src/Flybrain/`: separate FlyBrain project files.

All CNN functions use type annotations and parameter/return docstrings. Runtime
edge cases such as missing labels, empty splits, invalid image files, and
invalid KNN sizes raise explicit exceptions.
