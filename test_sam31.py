"""Smoke test: download SAM 3.1, load into image model, run one text prompt."""
import sys
import time

import numpy as np
from PIL import Image

img_path = sys.argv[1] if len(sys.argv) > 1 else "sam3/assets/images/test_image.jpg"
prompt = sys.argv[2] if len(sys.argv) > 2 else "person"

import app  # reuses model loading + segmentation code

t = time.time()
proc = app.get_processor()
print(f"model loaded in {time.time() - t:.1f}s")
img, info = app.load_image(img_path)
print(info)
t = time.time()
mask, scores = app.segment(img, prompt, 0.4)
print(f"prompt={prompt!r} scores={scores} coverage={100 * mask.mean():.2f}% time={time.time() - t:.2f}s")
app.render(img, mask, "Overlay on image", "#0078ff", 0.55).save("outputs/_smoke_test.jpg")
print("saved outputs/_smoke_test.jpg")
