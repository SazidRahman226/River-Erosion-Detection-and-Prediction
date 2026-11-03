# River segmentation with SAM 3.1 (local)

Gradio UI: upload a GeoTIFF / JPG / PNG → SAM 3.1 text-prompted segmentation → JPG with the river body masked → **Save image**.

## Layout
```
personal/
├── .venv/              Python 3.12 env (torch 2.10.0+cu128, sam3, gradio, rasterio, triton-windows 3.6)
├── sam3/               facebookresearch/sam3 source (installed editable)
├── app.py              the UI + GeoTIFF loading + segmentation
├── run_app.bat         double-click to start
├── requirements.txt
├── samples/            test images (Sentinel-2 Amazon as 4-band uint16 GeoTIFF, etc.)
└── outputs/            saved results (<name>_river_<time>.jpg + _river_mask_<time>.png)
```

## Run
Double-click `run_app.bat`, or:
```
.venv\Scripts\python app.py
```
Open http://127.0.0.1:7860 . Model loading takes ~1 min at startup; each image takes ~0.5 s per prompt on the RTX 5060.

## Using the UI
- **Text prompt(s)**: comma-separated, masks are merged. Default `river, water`.
  `water` is usually more reliable than `river` alone on satellite imagery.
- **Confidence threshold**: lower (0.2) if the river is missed, higher if land gets included.
- **Tile size**: 0 = whole image (resized to 1008 px internally). For large scenes use 1024–2048
  so narrow channels stay visible.
- **GeoTIFF bands**: 1-based R,G,B band numbers. Sentinel-2 full-stack true colour = `4,3,2`,
  Landsat 8/9 = `4,3,2`; a 3-band RGB export = `1,2,3`. Percentile stretch handles uint16/float data.
- **Output type**: overlay / binary mask / river-only. **Save image** writes a JPG (and a PNG mask) to `outputs/`
  and offers it for download.

## Notes
- Model: `facebook/sam3.1` checkpoint `sam3.1_multiplex.pt` (cached in `%USERPROFILE%\.cache\huggingface`).
  Its detector weights are loaded into the SAM 3 image model. The 4 "missing keys"
  (`vision_backbone.convs.3.*`) printed at startup are expected: SAM 3.1 uses a 3-scale neck and the
  image model discards that 4th scale (`scalp=1`).
- To use the original SAM 3 weights instead: `set SAM_VERSION=sam3` (needs access to `facebook/sam3`),
  or a local file: `set SAM_CHECKPOINT=C:\path\to\ckpt.pt`.
- Outputs are plain JPGs (no georeferencing). The saved PNG mask is pixel-aligned with the input raster.

## Reinstall from scratch
```
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ./sam3 -r requirements.txt
.venv\Scripts\hf auth login
```
