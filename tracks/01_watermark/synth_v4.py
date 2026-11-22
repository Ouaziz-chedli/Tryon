"""V4 zero-seam synthetic generator: 100% same-image edits using real COCO object contours
(clean_covers/val2017.tar + clean_covers/coco_masks_4k.npz) and U-shaped size distribution.

Zero cross-cover pasting and zero artificial blur seams: the ONLY difference between class 1
and class 0 pixels is the invisible watermark itself, preventing any visual shortcut learning.
"""
import tarfile
import cv2
import numpy as np

from wmloc.utils import downsample_mask


def load_coco_tar_and_masks(tar_path: str, masks_path: str, max_covers: int = 2000, seed: int = 0):
    z = np.load(masks_path)
    mask_names = z["file_names"].tolist()
    packed = z["packed"]

    by_name = {}
    for idx, fn in enumerate(mask_names):
        by_name.setdefault(fn, []).append(idx)

    rng = np.random.default_rng(seed)
    covers, cover_masks = [], []
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")]
        rng.shuffle(members)
        for mem in members:
            bn = mem.name.split("/")[-1]
            raw = tf.extractfile(mem).read()
            im = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            covers.append(cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA))
            cover_masks.append(by_name.get(bn, []))
            if len(covers) >= max_covers:
                break
    return covers, cover_masks, packed


def _unpack(packed_row) -> np.ndarray:
    return np.unpackbits(packed_row)[:512 * 512].reshape(512, 512)


def _rand_rect_u_shape(rng, max_ar: float, small_prob: float = 0.50):
    want_small = rng.random() < small_prob
    for _ in range(40):
        if want_small:
            area_target = float(rng.uniform(0.004, 0.08))
            ar = float(np.exp(rng.uniform(-np.log(max_ar), np.log(max_ar))))
            h = int(np.clip(round(512 * np.sqrt(area_target * ar)), 16, 512))
            w = int(np.clip(round(512 * np.sqrt(area_target / ar)), 16, 512))
        else:
            w = int(rng.integers(64, 513))
            h = int(rng.integers(64, 513))
        if max(w / h, h / w) <= max_ar:
            y0 = int(rng.integers(0, 512 - h + 1))
            x0 = int(rng.integers(0, 512 - w + 1))
            return y0, y0 + h, x0, x0 + w
    return 64, 448, 64, 448


def _sample_object_mask(rng, cover_idx, cover_masks, all_packed, prefer_small: bool):
    pool = cover_masks[cover_idx]
    for _ in range(6):
        if pool and rng.random() < 0.7:
            idx = int(pool[int(rng.integers(0, len(pool)))])
        else:
            idx = int(rng.integers(0, len(all_packed)))
        m = _unpack(all_packed[idx])
        frac = float(m.mean())
        if prefer_small and frac > 0.10:
            s = float(rng.uniform(0.20, 0.55))
            nh, nw = max(16, int(512 * s)), max(16, int(512 * s))
            sm = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
            canvas = np.zeros((512, 512), np.uint8)
            y0, x0 = int(rng.integers(0, 512 - nh + 1)), int(rng.integers(0, 512 - nw + 1))
            canvas[y0:y0 + nh, x0:x0 + nw] = sm
            if canvas.sum() >= 64:
                return canvas
        elif m.sum() >= 64:
            return m
    m = np.zeros((512, 512), np.uint8)
    cv2.ellipse(m, (int(rng.integers(64, 448)), int(rng.integers(64, 448))),
                (int(rng.integers(16, 100)), int(rng.integers(16, 100))), 0, 0, 360, 1, -1)
    return m


def generate_synthetic_v4(wm, method: str, covers, cover_masks, all_packed, n_samples: int = 1800, seed: int = 42):
    rng = np.random.default_rng(seed)
    max_ar = 2.0 if method == "trustmark" else 16.0
    n_cov = len(covers)
    imgs = np.empty((n_samples, 512, 512, 3), np.uint8)
    masks128 = np.empty((n_samples, 128, 128), np.uint8)
    msgs = np.empty((n_samples, wm.nbits), np.uint8)

    for i in range(n_samples):
        ci = int(rng.integers(0, n_cov))
        cov = covers[ci]
        bits = rng.integers(0, 2, wm.nbits, dtype=np.uint8)
        r = rng.random()
        prefer_small = rng.random() < 0.55

        if r < 0.30:
            # Same-image rectangular sub-embedding (simulates outpainting without fake seam)
            y0, y1, x0, x1 = _rand_rect_u_shape(rng, max_ar, small_prob=0.55)
            out = cov.copy()
            out[y0:y1, x0:x1] = wm.embed(cov[y0:y1, x0:x1], bits)
            m512 = np.zeros((512, 512), np.uint8)
            m512[y0:y1, x0:x1] = 1
        elif r < 0.65:
            # Same-image clean hole restoration (full_watermark=True, COCO object hole)
            wm_full = wm.embed(cov, bits).copy()
            obj = _sample_object_mask(rng, ci, cover_masks, all_packed, prefer_small)
            wm_full[obj == 1] = cov[obj == 1]
            out = wm_full
            m512 = (1 - obj).astype(np.uint8)
        else:
            # Same-image watermarked COCO object (full_watermark=False)
            wm_full = wm.embed(cov, bits)
            obj = _sample_object_mask(rng, ci, cover_masks, all_packed, prefer_small)
            out = cov.copy()
            out[obj == 1] = wm_full[obj == 1]
            m512 = obj

        if rng.random() < 0.25:
            q = int(rng.integers(80, 98))
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        imgs[i] = out
        masks128[i] = downsample_mask(m512)
        msgs[i] = bits

    return imgs, masks128, msgs
