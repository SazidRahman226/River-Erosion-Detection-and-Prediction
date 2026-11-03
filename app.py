"""
River segmentation UI using Meta SAM 3.1 (text-prompted concept segmentation).

Input : GeoTIFF (.tif/.tiff, any band count / dtype) or JPG/PNG.
Output: JPG with the river body highlighted (overlay) or a binary mask JPG.

Run:  .venv\\Scripts\\python app.py   ->  open http://127.0.0.1:7860
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from datetime import datetime
from pathlib import Path

# Make the cloned sam3 repo importable even if `pip install -e` was skipped
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sam3"))

import gradio as gr
import numpy as np
import torch
from PIL import Image

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_VERSION = os.environ.get("SAM_VERSION", "sam3.1")  # "sam3.1" or "sam3"

# ----------------------------------------------------------------------------
# Model (lazy-loaded once)
# ----------------------------------------------------------------------------
_processor = None


def get_processor():
    global _processor
    if _processor is None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

        ckpt = os.environ.get("SAM_CHECKPOINT") or download_ckpt_from_hf(version=SAM_VERSION)
        print(f"[sam] loading {SAM_VERSION} checkpoint: {ckpt}  (device={DEVICE})")
        model = build_sam3_image_model(checkpoint_path=ckpt, device=DEVICE, load_from_HF=False)
        _processor = Sam3Processor(model, device=DEVICE)
    return _processor


# ----------------------------------------------------------------------------
# Image loading (GeoTIFF aware)
# ----------------------------------------------------------------------------
def _stretch_to_uint8(arr, nodata_mask=None, pmin=2, pmax=98):
    """Per-band percentile stretch -> uint8. arr: (H, W, C) float."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = ~nodata_mask if nodata_mask is not None else np.ones(arr.shape[:2], bool)
    for c in range(arr.shape[2]):
        band = arr[..., c].astype(np.float64)
        vals = band[valid & np.isfinite(band)]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [pmin, pmax])
        if hi <= lo:
            hi = lo + 1
        out[..., c] = (np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if nodata_mask is not None:
        out[nodata_mask] = 0
    return out


def parse_bands(text, count):
    """'3,2,1' -> [3,2,1] (1-based). Falls back to sensible defaults."""
    try:
        bands = [int(b) for b in text.replace(" ", "").split(",") if b]
        if bands and all(1 <= b <= count for b in bands):
            return bands[:3]
    except ValueError:
        pass
    return [1, 2, 3] if count >= 3 else [1]


def load_image(path, bands_text="1,2,3", stretch=True):
    """Return (PIL RGB image, info string)."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            bands = parse_bands(bands_text, src.count)
            arr = src.read(bands).transpose(1, 2, 0)  # (H, W, C)
            nodata = None
            if src.nodata is not None:
                nodata = np.all(arr == src.nodata, axis=2)
            info = (
                f"GeoTIFF {src.width}x{src.height}, {src.count} band(s), dtype={src.dtypes[0]}, "
                f"CRS={src.crs}, using bands {bands}"
            )
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 2:
            arr = np.concatenate([arr, arr[..., :1]], axis=2)
        if arr.dtype == np.uint8 and not stretch:
            rgb = arr
        else:
            rgb = _stretch_to_uint8(arr, nodata) if stretch else _minmax_uint8(arr)
        return Image.fromarray(rgb, "RGB"), info
    img = Image.open(path).convert("RGB")
    return img, f"Image {img.width}x{img.height}"


def _minmax_uint8(arr):
    a = arr.astype(np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return ((a - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------
def _hex_to_rgb(h):
    h = h.lstrip("#")
    if h.startswith("rgb"):
        return (0, 120, 255)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


@torch.inference_mode()
def segment_prompts(image_pil, prompts, threshold):
    """Encode the image once, run every text prompt. Returns (union mask, {prompt: scores})."""
    proc = get_processor()
    proc.confidence_threshold = float(threshold)
    union = np.zeros((image_pil.height, image_pil.width), bool)
    scores_by_prompt = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        state = proc.set_image(image_pil)
        for p in prompts:
            proc.reset_all_prompts(state)
            out = proc.set_text_prompt(state=state, prompt=p)
            masks, scores = out["masks"], out["scores"]  # masks: (N, 1, H, W) bool
            if masks is not None and len(masks) > 0:
                union |= masks.any(dim=0).squeeze(0).cpu().numpy()
                scores_by_prompt[p] = [round(float(s), 3) for s in scores.float().cpu()]
            else:
                scores_by_prompt[p] = []
    return union, scores_by_prompt


def segment(image_pil, prompt, threshold):
    mask, s = segment_prompts(image_pil, [prompt], threshold)
    return mask, s[prompt]


def segment_tiled(image_pil, prompts, threshold, tile, overlap=0.15):
    """Run on overlapping tiles (for large scenes) and OR the masks together."""
    W, H = image_pil.size
    if tile <= 0 or (W <= tile and H <= tile):
        return segment_prompts(image_pil, prompts, threshold)
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    mask = np.zeros((H, W), bool)
    scores = {p: [] for p in prompts}
    for y in ys:
        for x in xs:
            box = (x, y, min(x + tile, W), min(y + tile, H))
            m, s = segment_prompts(image_pil.crop(box), prompts, threshold)
            mask[box[1]:box[3], box[0]:box[2]] |= m
            for p in prompts:
                scores[p] += s[p]
    scores = {p: f"{len(v)} instances over {len(xs) * len(ys)} tiles, max={max(v, default=0)}"
              for p, v in scores.items()}
    return mask, scores


def render(image_pil, mask, mode, color_hex, alpha):
    img = np.asarray(image_pil).astype(np.float32)
    if mode == "Binary mask (white river)":
        return Image.fromarray((mask * 255).astype(np.uint8), "L").convert("RGB")
    if mode == "River only (background black)":
        return Image.fromarray((img * mask[..., None]).astype(np.uint8))
    color = np.array(_hex_to_rgb(color_hex), np.float32)
    out = img.copy()
    out[mask] = (1 - alpha) * img[mask] + alpha * color
    return Image.fromarray(out.astype(np.uint8))


def run(file, prompt, threshold, bands, stretch, tile, mode, color, alpha):
    if file is None:
        raise gr.Error("Please upload a GeoTIFF / JPG / PNG image.")
    path = file if isinstance(file, str) else file.name
    image, info = load_image(path, bands, stretch)
    prompts = [p.strip() for p in (prompt or "river").split(",") if p.strip()]
    mask, all_scores = segment_tiled(image, prompts, threshold, int(tile))
    result = render(image, mask, mode, color, alpha)
    coverage = 100 * mask.mean()
    status = (
        f"{info}\nModel: {SAM_VERSION} on {DEVICE}\n"
        f"Instances per prompt (scores): {all_scores}\nRiver pixel coverage: {coverage:.2f}%"
    )
    state = {"result": result, "mask": mask, "name": Path(path).stem}
    return image, result, status, state, None


def save(state, quality):
    if not state:
        raise gr.Error("Run segmentation first.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{state['name']}_river_{ts}.jpg"
    state["result"].save(out_path, "JPEG", quality=int(quality))
    mask_path = OUTPUT_DIR / f"{state['name']}_river_mask_{ts}.png"
    Image.fromarray((state["mask"] * 255).astype(np.uint8)).save(mask_path)
    gr.Info(f"Saved to {out_path}")
    return str(out_path), f"Saved:\n{out_path}\n{mask_path}"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
with gr.Blocks(title="River Segmentation - SAM 3.1") as demo:
    gr.Markdown("## River Body Segmentation with SAM 3.1\nUpload a GeoTIFF or JPG, then click **Segment**.")
    st = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            f_in = gr.File(label="Input image (.tif / .tiff / .jpg / .png)",
                           file_types=[".tif", ".tiff", ".jpg", ".jpeg", ".png"])
            prompt = gr.Textbox(value="river, water", label="Text prompt(s), comma-separated",
                                info="Masks of all prompts are merged. 'water' is usually more reliable than 'river' alone")
            thr = gr.Slider(0.05, 0.95, value=0.3, step=0.05, label="Confidence threshold",
                            info="Lower it if the river is missed; raise it if land is included")
            tile = gr.Slider(0, 4096, value=0, step=256, label="Tile size in px (0 = whole image)",
                             info="For large scenes: e.g. 1024 keeps narrow rivers visible to the model")
            with gr.Accordion("GeoTIFF options", open=False):
                bands = gr.Textbox(value="1,2,3", label="Bands for R,G,B (1-based)",
                                   info="Sentinel-2 full stack true colour: 4,3,2; Landsat 8/9: 4,3,2")
                stretch = gr.Checkbox(value=True, label="Percentile (2-98%) contrast stretch")
            with gr.Accordion("Output style", open=True):
                mode = gr.Radio(["Overlay on image", "Binary mask (white river)",
                                 "River only (background black)"],
                                value="Overlay on image", label="Output type")
                color = gr.ColorPicker(value="#ff1e50", label="Mask colour")
                alpha = gr.Slider(0.1, 1.0, value=0.55, step=0.05, label="Overlay opacity")
            btn = gr.Button("Segment", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                img_in = gr.Image(label="Input (RGB preview)", type="pil", interactive=False)
                img_out = gr.Image(label="Segmented river", type="pil", interactive=False, format="jpeg")
            status = gr.Textbox(label="Info", lines=4)
            with gr.Row():
                q = gr.Slider(50, 100, value=95, step=1, label="JPG quality")
                save_btn = gr.Button("Save image", variant="secondary")
            dl = gr.File(label="Download saved JPG")
            save_msg = gr.Textbox(label="Save status", lines=2)

    btn.click(run, [f_in, prompt, thr, bands, stretch, tile, mode, color, alpha],
              [img_in, img_out, status, st, dl])
    save_btn.click(save, [st, q], [dl, save_msg])

if __name__ == "__main__":
    get_processor()  # load SAM 3.1 once at startup
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
