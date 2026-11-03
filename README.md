# River segmentation with SAM 3.1 (local)

Gradio UI: upload a GeoTIFF / JPG / PNG → SAM 3.1 text-prompted segmentation → JPG with the river body masked → **Save image**.

> **Not in this repo (downloaded on each machine):**
> - `sam3/`: Meta's SAM 3 source code, cloned from GitHub (~130 MB).
> - The SAM 3.1 weights (`sam3.1_multiplex.pt`, ~3.3 GB), downloaded from Hugging Face the first time you run the app and cached outside the project.
> - `.venv/`: your Python environment.

## Layout
```
River-Erosion-Detection-and-Prediction/
├── app.py              the UI + GeoTIFF loading + segmentation
├── test_sam31.py       command-line smoke test
├── run_app.bat         Windows: double-click to start
├── requirements.txt
├── samples/            test images (Sentinel-2 Amazon as 4-band uint16 GeoTIFF, etc.)
├── outputs/            saved results (<name>_river_<time>.jpg + _river_mask_<time>.png)
├── sam3/               (git-ignored) facebookresearch/sam3 source, you clone it here
└── .venv/              (git-ignored) Python environment
```

---

## Local setup

### 0. Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 – 3.12 | 3.12 |
| Disk | ~10 GB free (torch + weights) | |
| RAM | 16 GB | 32 GB |
| GPU | none (CPU works, but slowly) | NVIDIA GPU with ≥ 8 GB VRAM |
| Tools | `git` | [`uv`](https://docs.astral.sh/uv/) for faster installs (optional) |

You also need a free **Hugging Face account**. The SAM 3.1 weights are gated, so you have to request access once (step 5).

### 1. Clone this repo
```bash
git clone https://github.com/SazidRahman226/River-Erosion-Detection-and-Prediction.git
cd River-Erosion-Detection-and-Prediction
```

### 2. Clone the SAM 3 source into `./sam3`
The folder must be called `sam3` and sit next to `app.py`. Checking out the pinned commit gets you the exact version this project was tested with:
```bash
git clone https://github.com/facebookresearch/sam3.git sam3
git -C sam3 checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
```

### 3. Create a virtual environment

**Windows (PowerShell / cmd)**
```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

**Linux / macOS**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> If you use `uv`, `uv venv .venv --python 3.12` does the same thing. After that, use `uv pip install ...` wherever this README says `pip install ...`.

### 4. Install PyTorch for your hardware, then the project

Pick **one** line from this table and run it inside the activated venv:

| Hardware | Command |
|---|---|
| NVIDIA RTX 50xx (Blackwell), or any recent NVIDIA GPU | `pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128` |
| Older NVIDIA GPU / driver (GTX 10xx – RTX 40xx, driver < 570) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| No NVIDIA GPU (CPU only, Windows / Linux) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| macOS (Apple Silicon / Intel) | `pip install torch torchvision` |

Run `nvidia-smi` to see your GPU and driver. The "CUDA Version" it reports must be at least the version in the wheel name (cu128 = 12.8).

Then install SAM 3 and this app's dependencies:
```bash
pip install "setuptools<81"          # sam3 still uses pkg_resources
pip install -e ./sam3 -r requirements.txt
```

**Triton.** SAM 3 imports `triton` as soon as it loads, even on CPU:
- **Windows:** `requirements.txt` already installs `triton-windows`.
- **Linux + NVIDIA:** the CUDA torch wheel already includes `triton`.
- **Linux CPU-only:** run `pip install triton` too. It only needs to import; it is never actually used on CPU.
- **macOS:** Triton has no macOS build. See [Troubleshooting](#troubleshooting).

Check that it worked:
```bash
python -c "import torch, triton, sam3; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### 5. Get access to the SAM 3.1 weights (one time)
1. Sign in at https://huggingface.co and open https://huggingface.co/facebook/sam3.1. Accept the license / request access, and wait until the page says you have been granted access.
2. Create a **read** token at https://huggingface.co/settings/tokens.
3. Log in from the venv:
   ```bash
   hf auth login           # paste the token
   ```
   On older `huggingface_hub` versions the command is `huggingface-cli login`.

The first run downloads `sam3.1_multiplex.pt` (~3.3 GB) into the Hugging Face cache, **not** into the project:
- Windows: `%USERPROFILE%\.cache\huggingface\hub`
- Linux / macOS: `~/.cache/huggingface/hub`

To put the cache somewhere else (another drive, for example), set `HF_HOME` before running. If you already have the `.pt` file, point to it with `SAM_CHECKPOINT` and nothing gets downloaded (see [Configuration](#configuration)).

### 6. Smoke test (optional)
```bash
python test_sam31.py samples/amazon_s2.jpg water
```
This should print model load time, scores and coverage, and write `outputs/_smoke_test.jpg`.

---

## Run
- **Windows:** double-click `run_app.bat`, or run `.venv\Scripts\python app.py`
- **Linux / macOS:** `source .venv/bin/activate && python app.py`

Open http://127.0.0.1:7860 (it opens by itself). Loading the model takes about 1 minute at startup. After that, each image takes about 0.5 s per prompt on an RTX 5060. On CPU, expect tens of seconds per image.

## Using the UI
- **Text prompt(s)**: comma-separated, and the masks are merged. Default `river, water`.
  `water` is usually more reliable than `river` alone on satellite imagery.
- **Confidence threshold**: lower it (0.2) if the river is missed, raise it if land gets included.
- **Tile size**: 0 = whole image (resized to 1008 px internally). For large scenes use 1024–2048
  so narrow channels stay visible.
- **GeoTIFF bands**: 1-based R,G,B band numbers. Sentinel-2 full-stack true colour = `4,3,2`,
  Landsat 8/9 = `4,3,2`, a 3-band RGB export = `1,2,3`. The percentile stretch handles uint16/float data.
- **Output type**: overlay / binary mask / river-only. **Save image** writes a JPG (and a PNG mask) to `outputs/`
  and offers it for download.

## Configuration
Environment variables, set before `python app.py`. Windows: `set NAME=value`. Linux / macOS: `export NAME=value`.

| Variable | Effect |
|---|---|
| `SAM_CHECKPOINT` | Path to a local `.pt` file. Skips the Hugging Face download completely (good for offline machines). |
| `SAM_VERSION` | `sam3.1` (default) or `sam3`, which uses the original SAM 3 weights (needs access to `facebook/sam3`). |
| `HF_HOME` | Where the Hugging Face cache (and the downloaded weights) is stored. |

To move the weights to an offline machine, copy `sam3.1_multiplex.pt` from the cache folder above (it is under `models--facebook--sam3.1/snapshots/<hash>/`) and set `SAM_CHECKPOINT` to its path.

## Notes
- The SAM 3.1 detector weights are loaded into the SAM 3 *image* model. The 4 "missing keys"
  (`vision_backbone.convs.3.*`) printed at startup are expected: SAM 3.1 uses a 3-scale neck, and the
  image model discards that 4th scale (`scalp=1`).
- The app uses CUDA when it is available and the CPU otherwise. Apple's MPS backend is not used.
- Outputs are plain JPGs with no georeferencing. The saved PNG mask is pixel-aligned with the input raster.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sam3'` | The `sam3/` folder is missing or in the wrong place. Redo step 2 from the project root. |
| `ModuleNotFoundError: No module named 'triton'` | Windows: `pip install "triton-windows>=3.6,<3.7"`. Linux: `pip install triton`. macOS: see below. |
| `No module named 'pkg_resources'` | `pip install "setuptools<81"` |
| `401` / `GatedRepoError` / `Access to model facebook/sam3.1 is restricted` | Access has not been granted yet, or you are not logged in. Redo step 5. |
| `torch.cuda.is_available()` is `False` on an NVIDIA machine | You installed the CPU wheel, or your driver is too old for the wheel. Reinstall torch from the right row in step 4 (`pip install --force-reinstall ...`). |
| `CUDA error: no kernel image is available` (RTX 50xx) | You need the `cu128` wheel (torch ≥ 2.7). |
| CUDA out of memory | Use a smaller **Tile size** (e.g. 1024) or close other GPU apps. |
| **macOS:** `import triton` fails | Triton has no macOS build, and SAM 3 imports it at load time (only the video/tracking code uses it). One workaround is an empty stub package: create `.venv/lib/python3.12/site-packages/triton/__init__.py` and `.venv/lib/python3.12/site-packages/triton/language/__init__.py`, each defining `def jit(f=None, **k): return f if f else (lambda g: g)` and `constexpr = int`. The app should then run on CPU. (This workaround hasn't been tested on a Mac.) |
