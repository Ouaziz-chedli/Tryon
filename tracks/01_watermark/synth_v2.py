"""Synthetic watermark-localisation dataset generator using clean COCO val2017 covers and the exact
embedders (PixelSeal, TrustMark, StegaStamp).

Simulates the 4 edit types from the task spec without creating fake semantic seams on 75% of samples:
  1. Sub-rectangle embedding (outpainting): embed a rectangular crop (aspect <= 2:1 for trustmark,
     <= 16:1 for others) and place back into the clean cover.
  2. Restoration hole (full_watermark=True): embed full image or sub-rect, then restore clean original
     pixels in an irregular polygon/blob or rectangle.
  3. Object transplantation (full_watermark=False): paste watermarked region onto clean cover (mostly
     same cover to force pure watermark learning; 25% cross-cover paste for real boundary robustness).
"""
from pathlib import Path
import cv2
import numpy as np

from wmloc.utils import downsample_mask


def load_clean_covers(cover_dir: str, limit: int = 2000, seed: int = 0) -> list[np.ndarray]:
    paths = sorted(Path(cover_dir).glob("*.jpg"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))[:limit]
    imgs = []
    for i in idx:
        im = cv2.imread(str(paths[i]))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        # resize shorter side >= 512 if smaller, then random 512x512 crop
        if min(h, w) < 512:
            scale = 512.0 / min(h, w)
            im = cv2.resize(im, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]
        y0 = int(rng.integers(0, h - 512 + 1))
        x0 = int(rng.integers(0, w - 512 + 1))
        imgs.append(im[y0:y0 + 512, x0:x0 + 512])
    return imgs


def _rand_rect(rng, max_ar: float, min_side: int = 48):
    for _ in range(30):
        w = int(rng.integers(min_side, 513))
        h = int(rng.integers(min_side, 513))
        if max(w / h, h / w) <= max_ar:
            y0 = int(rng.integers(0, 512 - h + 1))
            x0 = int(rng.integers(0, 512 - w + 1))
            return y0, y0 + h, x0, x0 + w
    return 64, 448, 64, 448


def _rand_blob(rng, size: int = 512) -> np.ndarray:
    """Smooth irregular polygon/ellipse mask in {0, 1} uint8."""
    m = np.zeros((size, size), np.uint8)
    cx = float(rng.uniform(0.15, 0.85) * size)
    cy = float(rng.uniform(0.15, 0.85) * size)
    rx = float(rng.uniform(0.06, 0.42) * size)
    ry = float(rng.uniform(0.06, 0.42) * size)
    n_pts = int(rng.integers(7, 16))
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
    radii = rng.uniform(0.55, 1.25, n_pts)
    pts = np.stack([
        np.clip(cx + rx * radii * np.cos(angles), 0, size - 1),
        np.clip(cy + ry * radii * np.sin(angles), 0, size - 1),
    ], axis=1).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    if rng.random() < 0.35:
        # second blob
        cx2, cy2 = float(rng.uniform(0.2, 0.8) * size), float(rng.uniform(0.2, 0.8) * size)
        r2 = int(rng.uniform(24, 110))
        cv2.circle(m, (int(cx2), int(cy2)), r2, 1, -1)
    return m


def generate_synthetic(wm, method: str, covers: list[np.ndarray], n_samples: int = 1200, seed: int = 42):
    rng = np.random.default_rng(seed)
    max_ar = 2.0 if method == "trustmark" else 16.0
    n_cov = len(covers)
    imgs = np.empty((n_samples, 512, 512, 3), np.uint8)
    masks128 = np.empty((n_samples, 128, 128), np.uint8)
    msgs = np.empty((n_samples, wm.nbits), np.uint8)

    for i in range(n_samples):
        cov = covers[int(rng.integers(0, n_cov))]
        bits = rng.integers(0, 2, wm.nbits, dtype=np.uint8)
        mode = int(rng.integers(0, 4))

        if mode == 0:
            # Outpainting: embed a rectangular sub-region and paste onto clean cover
            y0, y1, x0, x1 = _rand_rect(rng, max_ar, min_side=64 if method == "trustmark" else 48)
            out = cov.copy()
            out[y0:y1, x0:x1] = wm.embed(cov[y0:y1, x0:x1], bits)
            m512 = np.zeros((512, 512), np.uint8)
            m512[y0:y1, x0:x1] = 1
        elif mode == 1:
            # Hole restoration (same image, zero semantic seam): full wm minus irregular clean hole
            wm_full = wm.embed(cov, bits)
            hole = _rand_blob(rng)
            out = wm_full.copy()
            out[hole == 1] = cov[hole == 1]
            m512 = (1 - hole).astype(np.uint8)
        elif mode == 2:
            # Object transplantation (same image, zero semantic seam): watermarked blob on clean cover
            wm_full = wm.embed(cov, bits)
            blob = _rand_blob(rng)
            out = cov.copy()
            out[blob == 1] = wm_full[blob == 1]
            m512 = blob
        else:
            # Cross-cover edit (addition or transplantation with real content change)
            cov2 = covers[int(rng.integers(0, n_cov))]
            blob = _rand_blob(rng)
            if rng.random() < 0.5:
                out = wm.embed(cov, bits).copy()
                out[blob == 1] = cov2[blob == 1]
                m512 = (1 - blob).astype(np.uint8)
            else:
                wm2 = wm.embed(cov2, bits)
                out = cov.copy()
                out[blob == 1] = wm2[blob == 1]
                m512 = blob

        # Light realistic JPEG re-encoding on 30% of samples to narrow synthetic-to-real gap
        if rng.random() < 0.30:
            q = int(rng.integers(78, 96))
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        imgs[i] = out
        masks128[i] = downsample_mask(m512)
        msgs[i] = bits

    return imgs, masks128, msgs
