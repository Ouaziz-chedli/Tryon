"""V3 synthetic generator using real COCO segmentation masks (clean_covers/val2017.tar +
clean_covers/coco_masks_4k.npz) and matching the U-shaped area distribution of the real dataset
(heavy sampling of <5% small objects, >95% small holes, and up to 16:1 outpainting strips).
"""
import io
import tarfile
import cv2
import numpy as np

from wmloc.utils import downsample_mask


def load_coco_tar_and_masks(tar_path: str, masks_path: str, max_covers: int = 2000, seed: int = 0):
    z = np.load(masks_path)
    mask_names = z["file_names"].tolist()
    packed = z["packed"]

    # Group mask indices by COCO image filename
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
            arr = np.frombuffer(raw, dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im512 = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
            covers.append(im512)
            cover_masks.append(by_name.get(bn, []))
            if len(covers) >= max_covers:
                break

    all_masks_packed = packed
    return covers, cover_masks, all_masks_packed


def _unpack(packed_row) -> np.ndarray:
    return np.unpackbits(packed_row)[:512 * 512].reshape(512, 512)


def _rand_rect_u_shape(rng, max_ar: float, small_prob: float = 0.45):
    """Sample rectangles with heavy representation of small boxes and thin strips (up to max_ar)."""
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
    """Pick a COCO mask (preferring matching-image object when available) or scaled/shifted object."""
    pool = cover_masks[cover_idx]
    for _ in range(6):
        if pool and rng.random() < 0.7:
            idx = int(pool[int(rng.integers(0, len(pool)))])
        else:
            idx = int(rng.integers(0, len(all_packed)))
        m = _unpack(all_packed[idx])
        frac = float(m.mean())
        if prefer_small and frac > 0.12:
            # Downscale object to create small 0.5%..8% targets
            s = float(rng.uniform(0.22, 0.55))
            nh, nw = max(16, int(512 * s)), max(16, int(512 * s))
            sm = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
            canvas = np.zeros((512, 512), np.uint8)
            y0, x0 = int(rng.integers(0, 512 - nh + 1)), int(rng.integers(0, 512 - nw + 1))
            canvas[y0:y0 + nh, x0:x0 + nw] = sm
            if canvas.sum() >= 64:
                return canvas
        elif m.sum() >= 64:
            return m
    # Fallback ellipse
    m = np.zeros((512, 512), np.uint8)
    cx, cy = int(rng.integers(64, 448)), int(rng.integers(64, 448))
    rx, ry = int(rng.integers(16, 100)), int(rng.integers(16, 100))
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return m


def _blend(dst: np.ndarray, src: np.ndarray, mask: np.ndarray, soft: bool) -> np.ndarray:
    if not soft:
        out = dst.copy()
        out[mask == 1] = src[mask == 1]
        return out
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)[..., None]
    return np.clip(src * alpha + dst * (1.0 - alpha), 0, 255).round().astype(np.uint8)


def generate_synthetic_v3(wm, method: str, covers, cover_masks, all_packed, n_samples: int = 1600, seed: int = 42):
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
        mode = int(rng.integers(0, 4))
        prefer_small = rng.random() < 0.50

        if mode == 0:
            # Outpainting (rectangular subset, including small boxes & thin strips)
            y0, y1, x0, x1 = _rand_rect_u_shape(rng, max_ar, small_prob=0.50)
            if rng.random() < 0.45:
                # Outpainted surrounding region from blurred/shifted cover or second cover
                bg = covers[int(rng.integers(0, n_cov))].copy()
            else:
                bg = cov.copy()
            sub_wm = wm.embed(cov[y0:y1, x0:x1], bits)
            out = bg
            out[y0:y1, x0:x1] = sub_wm
            m512 = np.zeros((512, 512), np.uint8)
            m512[y0:y1, x0:x1] = 1
        elif mode == 1:
            # Object Removal / Alteration (full_watermark=True): small or medium COCO object restored/inpainted
            wm_full = wm.embed(cov, bits)
            obj = _sample_object_mask(rng, ci, cover_masks, all_packed, prefer_small)
            if rng.random() < 0.55:
                fill_src = cov
            else:
                # Simulate inpainting/blur removal inside object hole
                fill_src = cv2.GaussianBlur(cov, (15, 15), 4.0)
            out = _blend(wm_full, fill_src, obj, soft=(rng.random() < 0.4))
            m512 = (1 - obj).astype(np.uint8)
        elif mode == 2:
            # Object Addition (full_watermark=True): clean COCO object from cov2 added onto watermarked cov
            ci2 = int(rng.integers(0, n_cov))
            cov2 = covers[ci2]
            wm_full = wm.embed(cov, bits)
            obj2 = _sample_object_mask(rng, ci2, cover_masks, all_packed, prefer_small)
            out = _blend(wm_full, cov2, obj2, soft=(rng.random() < 0.5))
            m512 = (1 - obj2).astype(np.uint8)
        else:
            # Object Transplantation (full_watermark=False): watermarked COCO object pasted onto clean cover
            ci2 = int(rng.integers(0, n_cov))
            cov2 = covers[ci2]
            obj2 = _sample_object_mask(rng, ci2, cover_masks, all_packed, prefer_small)
            wm2 = wm.embed(cov2, bits)
            bg = cov if rng.random() < 0.5 else cov2
            out = _blend(bg, wm2, obj2, soft=(rng.random() < 0.5))
            m512 = obj2

        if rng.random() < 0.30:
            q = int(rng.integers(78, 97))
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        imgs[i] = out
        masks128[i] = downsample_mask(m512)
        msgs[i] = bits

    return imgs, masks128, msgs
