"""Zero-shot Track 1 baseline (~0.48 mIoU) using decoder message-correlation maps, with threshold tuned
on all 510 train images. Runs on CPU or CUDA and optionally submits to the leaderboard.

    python work_bassim/zeroshot_submit.py --device cpu --out output/submission_zeroshot.pt --submit
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"


def up128(m: torch.Tensor) -> np.ndarray:
    return F.interpolate(m[None, None].float(), (128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


@torch.no_grad()
def map_pixelseal(wm, img, bits, device):
    x = to_tensor(img).to(device)
    msg = torch.from_numpy(bits).float().to(device) * 2 - 1
    xr = wm.det.preprocess(x)
    emb = wm.det.pixel_decoder.output_upscaling(wm.det.convnext(xr))
    lin = wm.det.pixel_decoder.linear
    pp = torch.einsum("bchw,oc->bohw", emb, lin.weight)[0] + lin.bias[:, None, None]
    dot16 = (pp[1:] * msg[:, None, None]).mean(0)
    return up128(dot16)


@torch.no_grad()
def map_trustmark(wm, img, bits, device):
    x = to_tensor(img).to(device)
    msg = torch.from_numpy(bits).float().to(device) * 2 - 1
    pp8 = wm.features(x)["per_position_logits"][0]
    dot8 = (pp8 * msg[:, None, None]).mean(0)
    return up128(dot8)


class SSContrib:
    def __init__(self, wm, device):
        import onnx
        import onnxruntime as ort
        self.wm = wm
        m = onnx.load("models/stegastamp/stega_stamp_logits.onnx")
        init = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
        self.W1 = init["gen_decoder/dec_dense/MatMul/ReadVariableOp__75"]
        self.W2 = init["gen_decoder/decoder_out/MatMul/ReadVariableOp__77"]
        for name in ("gen_decoder/dec_conv_7/BiasAdd__443:0", "gen_decoder/dec_dense/BiasAdd:0"):
            m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        prov = (["CUDAExecutionProvider"] if "cuda" in device else []) + ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(m.SerializeToString(), so, providers=prov)

    def __call__(self, img, bits):
        x = self.wm._prep(to_tensor(img))
        _, c7, dense = self.sess.run(
            ["gen_decoder/decoder_out/BiasAdd:0", "gen_decoder/dec_conv_7/BiasAdd__443:0", "gen_decoder/dec_dense/BiasAdd:0"],
            {"image": x, "secret": np.zeros((1, 100), np.float32)},
        )
        msg = bits.astype(np.float32) * 2 - 1
        g = self.W1 @ ((self.W2 @ msg) * (dense[0] > 0))
        contrib = (c7[0] * g.reshape(13, 13, 128)).sum(-1)
        return up128(torch.from_numpy(contrib))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="output/submission_zeroshot.pt")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    if a.device == "cpu":
        torch.set_num_threads(16)

    t0 = time.time()
    pred_test = {}
    all_train_preds, all_train_gts = [], []

    for method in ("pixelseal", "trustmark", "stegastamp"):
        t1 = time.time()
        wm = load_watermark(method, models_dir="models", device=a.device)
        ss = SSContrib(wm, a.device) if method == "stegastamp" else None

        def get_map(it):
            if method == "pixelseal":
                return map_pixelseal(wm, it["image"], it["message"], a.device)
            if method == "trustmark":
                return map_trustmark(wm, it["image"], it["message"], a.device)
            return ss(it["image"], it["message"])

        ds_tr = LocalisationDataset("release", "train", methods=[method])
        tr_maps = [get_map(ds_tr[i]) for i in range(len(ds_tr))]
        tr_gts = [ds_tr[i]["mask128"] for i in range(len(ds_tr))]

        lo, hi = np.percentile(np.concatenate([m.ravel() for m in tr_maps]), [5, 95])
        best_t, best_s = 0.0, 0.0
        for t in np.linspace(lo, hi, 41):
            s = score([(m > t).astype(np.uint8) for m in tr_maps], tr_gts)
            if s > best_s:
                best_s, best_t = float(s), float(t)

        all_train_preds.extend([(m > best_t).astype(np.uint8) for m in tr_maps])
        all_train_gts.extend(tr_gts)
        print(f"[{method}] train mIoU={best_s:.4f} (thr={best_t:.4f}) in {time.time() - t1:.1f}s", flush=True)

        ds_te = LocalisationDataset("release", "test", methods=[method])
        for i in range(len(ds_te)):
            it = ds_te[i]
            pred_test[it["image_name"]] = (get_map(it) > best_t).astype(np.uint8)

    print(f"Overall train mIoU (510 imgs): {score(all_train_preds, all_train_gts):.4f} | total {time.time() - t0:.1f}s", flush=True)

    test_names = [m["image_name"] for m in LocalisationDataset("release", "test").meta]
    masks = torch.from_numpy(np.stack([pred_test[n] for n in test_names])).bool()
    assert masks.shape == (len(test_names), 128, 128) and len(set(test_names)) == len(test_names)
    torch.save({"image_names": test_names, "masks": masks}, a.out)
    size_mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}: {tuple(masks.shape)}, {size_mb:.1f} MB, frac={masks.float().mean():.3f}", flush=True)
    assert size_mb < 20

    if a.submit:
        import requests
        key = os.environ["TRYON_API_KEY"]
        with open(a.out, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": key},
                files={"file": (os.path.basename(a.out), f, "application/octet-stream")},
                timeout=(10, 600),
            )
        print("HTTP", r.status_code, r.text[:500], flush=True)


if __name__ == "__main__":
    main()
