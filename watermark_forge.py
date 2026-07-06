import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent
WMFORGER_DIR = ROOT / "vendor" / "videoseal-main" / "wmforger"
sys.path.insert(0, str(WMFORGER_DIR))
_cwd = os.getcwd()
os.chdir(WMFORGER_DIR)  # build_extractor() loads configs/extractor.yaml via a relative path
from wmforger.models import build_extractor  # noqa: E402
os.chdir(_cwd)

DATASET_DIR = ROOT / "Dataset"
TEMPLATE_DIR = ROOT / "templates_cache"
OUT_DIR = ROOT / "outputs"
TEMPLATE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def _patch_hf_hub_download():
    import huggingface_hub
    import huggingface_hub.file_download as _fd
    if getattr(huggingface_hub.hf_hub_download, "_wm_patched", False):
        return
    _orig = huggingface_hub.hf_hub_download

    def _patched(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        kwargs["token"] = None
        return _orig(*args, **kwargs)
    _patched._wm_patched = True
    huggingface_hub.hf_hub_download = _patched
    _fd.hf_hub_download = _patched


_patch_hf_hub_download()

CATEGORIES = [
    ("WM_1", 1, 25), ("WM_2", 26, 50), ("WM_3", 51, 75), ("WM_4", 76, 100),
    ("WM_5", 101, 125), ("WM_6", 126, 150), ("WM_7", 151, 175), ("WM_8", 176, 200),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BUDGET = 0.022  

MODEL_SIZE = 768
EXTRACT_STEPS = 400  
EXTRACT_LR = 0.05
EXTRACT_MINIBATCH = 5  
TRIM_FRAC = 0.12  

def load_discriminator(ckpt_path=WMFORGER_DIR / "convnext_pref_model.pth"):
    import omegaconf
    state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
    cfg = omegaconf.OmegaConf.load(WMFORGER_DIR / "configs" / "extractor.yaml")["convnext_tiny"]
    model = build_extractor("convnext_tiny", cfg, img_size=256, nbits=0)
    model.load_state_dict(state_dict)
    model = model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def read_rgb01(p):
    arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W in [0,1]


def _extract_chunk(model, chunk01, h, w):
    x = F.interpolate(chunk01, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False).to(DEVICE)
    delta = torch.zeros_like(x, requires_grad=True)
    optim = torch.optim.SGD([delta], lr=EXTRACT_LR)
    for _ in range(EXTRACT_STEPS):
        optim.zero_grad()
        loss = -model((x + delta).clamp(0, 1)).mean()
        loss.backward()
        optim.step()
    with torch.no_grad():
        cleaned = (x + delta).clamp(0, 1)
        cleaned_native = F.interpolate(cleaned, size=(h, w), mode="bilinear", align_corners=False)
        residual_native = (chunk01.to(DEVICE) - cleaned_native) * 255.0  # pixel scale
    return residual_native.permute(0, 2, 3, 1).cpu().numpy()


def extract_pref_template(model, src_paths):
    """Gradient-ascent extraction, averaged (trimmed mean) over the group's 25
    source images. Runs in EXTRACT_MINIBATCH-sized sub-batches (memory)."""
    imgs01 = torch.stack([read_rgb01(p) for p in src_paths])
    n, c, h, w = imgs01.shape
    chunks = []
    for i in range(0, n, EXTRACT_MINIBATCH):
        chunks.append(_extract_chunk(model, imgs01[i:i + EXTRACT_MINIBATCH], h, w))
        torch.cuda.empty_cache()
    res = np.concatenate(chunks, axis=0)  # N,H,W,3
    lo = np.quantile(res, TRIM_FRAC, axis=0)
    hi = np.quantile(res, 1.0 - TRIM_FRAC, axis=0)
    trimmed = np.clip(res, lo, hi).mean(axis=0)
    return trimmed - trimmed.mean(axis=(0, 1), keepdims=True)

def gauss(a, s):
    return a if s <= 0 else gaussian_filter(a, sigma=(s, s, 0), mode="reflect")


def nlm(img):
    u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoisingColored(u8, None, 5, 5, 7, 21).astype(np.float32)


def resize_to(a, h, w):
    if a.shape[0] == h and a.shape[1] == w:
        return a.astype(np.float32)
    u8 = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(u8, "RGB").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32)


def read_rgb255(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def extract_nlm_template(src_paths, plv_p=1.0):
    native = [read_rgb255(p) for p in src_paths]
    h, w = native[0].shape[:2]
    sources = [resize_to(s, h, w) for s in native]

    res = []
    for s in sources:
        r = s - nlm(s)
        res.append(r - r.mean((0, 1), keepdims=True))
    res = np.stack(res, 0)
    Rc = np.fft.fft2(res, axes=(1, 2))
    Ravg = Rc.mean(0)
    plv = np.abs((Rc / (np.abs(Rc) + 1e-6)).mean(0))
    Wf = Ravg * (plv ** plv_p)
    w_ = np.real(np.fft.ifft2(Wf, axes=(0, 1))).astype(np.float32)
    var = res.var(0)
    n = res.shape[0]
    g = (gauss(w_ * w_, 1.0) - 0.3 * gauss(var / n, 1.0)) / (gauss(w_ * w_, 1.0) + 1e-6)
    w_ = w_ * np.clip(g, 0.3, 1.0)
    return w_ - w_.mean((0, 1), keepdims=True)


def unit_std(a):
    return a / (a.std() + 1e-6)


def build_and_cache_templates():
    model = None
    for g, a, b in CATEGORIES:
        pref_path = TEMPLATE_DIR / f"{g}_pref.npy"
        nlm_path = TEMPLATE_DIR / f"{g}_nlm.npy"
        src_dir = DATASET_DIR / "watermarked_sources" / g
        src_paths = sorted(src_dir.glob("*.png"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))

        if not pref_path.exists():
            if model is None:
                model = load_discriminator()
            print(f"[{g}] extracting preference-model template ({EXTRACT_STEPS} steps x 25 images)...")
            np.save(pref_path, extract_pref_template(model, src_paths))
        if not nlm_path.exists():
            print(f"[{g}] estimating NLM+PLV template...")
            np.save(nlm_path, extract_nlm_template(src_paths))


def load_templates(g):
    return np.load(TEMPLATE_DIR / f"{g}_pref.npy"), np.load(TEMPLATE_DIR / f"{g}_nlm.npy")

DEFAULT_PREF_WEIGHT = 0.35


def make_blend(pref_weight):
    def fn(pref, nlm_t):
        return pref_weight * unit_std(pref) + (1 - pref_weight) * unit_std(nlm_t)
    return fn

def rgb2ycc(d):
    r, g, b = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([
        0.299 * r + 0.587 * g + 0.114 * b,
        -0.168736 * r - 0.331264 * g + 0.5 * b,
        0.5 * r - 0.418688 * g - 0.081312 * b,
    ], -1)


def ycc2rgb(d):
    y, cb, cr = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], -1)


def pmask(t):
    g = t.mean(2)
    tex = 0.6 * np.abs(g - gauss(t, 1.0).mean(2)) + 0.4 * np.abs(g - gauss(t, 3.0).mean(2))
    lo, hi = np.percentile(tex, 10), np.percentile(tex, 95)
    tex = np.clip((tex - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.clip(0.55 + 1.0 * tex, 0.55, 1.6)[..., None].astype(np.float32)


def direction(target, w_hat, y_gain=0.85, chroma=1.2):
    d = ycc2rgb(rgb2ycc(w_hat) * np.array([y_gain, chroma, chroma])) * pmask(target)
    return (d / (d.std() + 1e-6)).astype(np.float32)  # unit-std direction


def apply(target, dirn, alpha, cap=26.0):
    delta = np.clip(alpha * dirn, -cap, cap)
    f = target + delta
    return np.clip(np.where(delta >= 0, np.ceil(f), np.floor(f)), 0, 255)


def lpips_dist(lp_fn, a, b):
    def tt(x):
        return (torch.from_numpy(x / 255.0).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1).to(DEVICE)
    with torch.no_grad():
        return float(lp_fn(tt(a), tt(b)).item())


def embed_to_budget(lp_fn, target, dirn, budget, iters=8, max_alpha=12.0):
    lo, hi = 0.0, max_alpha
    for _ in range(iters):
        mid = (lo + hi) / 2
        if lpips_dist(lp_fn, target, apply(target, dirn, mid)) < budget:
            lo = mid
        else:
            hi = mid
    return apply(target, dirn, lo)


def discover_rivagan_message(group, min_agreement=0.8):
    import cv2
    from imwatermark.watermark import WatermarkDecoder
    WatermarkDecoder.loadModel()
    dec = WatermarkDecoder("bits", 32)

    src_dir = DATASET_DIR / "watermarked_sources" / group
    paths = sorted(src_dir.glob("*.png"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    all_bits = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None or im.shape[0] * im.shape[1] < 256 * 256:
            return None  # RivaGAN needs >=256x256
        try:
            all_bits.append(dec.decode(im, "rivaGan"))
        except Exception:
            return None
    all_bits = np.stack(all_bits)
    agreement = np.abs(all_bits.mean(axis=0) - 0.5) * 2  # 0 = split 50/50, 1 = unanimous
    if agreement.mean() < min_agreement:
        return None
    return (all_bits.mean(axis=0) > 0.5).astype(int).tolist()


def discover_trustmark_secret(group, model_type="Q", min_agreement=0.9):
    from trustmark import TrustMark
    from PIL import Image as PILImage
    tm = TrustMark(verbose=False, model_type=model_type)

    src_dir = DATASET_DIR / "watermarked_sources" / group
    paths = sorted(src_dir.glob("*.png"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    results = []
    for p in paths:
        im = PILImage.open(p).convert("RGB")
        secret, present, _ = tm.decode(im)
        results.append(secret if present else None)

    from collections import Counter
    counts = Counter(results)
    top_secret, top_count = counts.most_common(1)[0]
    if top_secret is None or top_count / len(results) < min_agreement:
        return None
    return {"scheme": "trustmark", "model_type": model_type, "secret": top_secret}


def discover_vine_message(group, img_size=256, min_agreement=0.9):
    from watermarklab.watermarks.PGWs import VINE
    from collections import Counter
    model = VINE(img_size=img_size, bits_len=100, model_type="R", device=DEVICE)

    src_dir = DATASET_DIR / "watermarked_sources" / group
    paths = sorted(src_dir.glob("*.png"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    imgs = [np.asarray(Image.open(p).convert("RGB").resize((img_size, img_size), Image.BICUBIC)) for p in paths]
    result = model.extract(imgs)
    msgs = [tuple(bits) for bits in result.ext_bits]
    counts = Counter(msgs)
    top_msg, top_count = counts.most_common(1)[0]
    if top_count / len(msgs) < min_agreement:
        return None
    return {"scheme": "vine", "img_size": img_size, "message": list(top_msg)}

GROUP_OVERRIDES = {
    "WM_2": {"scheme": "rivaGan", "message": [int(b) for b in "00010000101111110011101011101000"]},
    "WM_4": {"scheme": "vine", "img_size": 256, "message": [
        int(b) for b in "1100100101010001111111010110110011110000011111100011100101011111000010000001000010001101100011110001"
    ]},
    "WM_7": {"scheme": "trustmark", "model_type": "Q", "secret": "DLS\x1edKpw|"},
}


def load_override_encoder(override):
    if override["scheme"] == "rivaGan":
        from imwatermark import WatermarkEncoder
        WatermarkEncoder.loadModel()
        enc = WatermarkEncoder()
        enc.set_watermark("bits", override["message"])
        return enc
    elif override["scheme"] == "trustmark":
        from trustmark import TrustMark
        return TrustMark(verbose=False, model_type=override["model_type"])
    elif override["scheme"] == "vine":
        from watermarklab.watermarks.PGWs import VINE
        return VINE(img_size=override["img_size"], bits_len=len(override["message"]), model_type="R", device=DEVICE)
    else:
        raise ValueError(f"unknown override scheme: {override['scheme']}")


def apply_group_override(encoder, override, clean_path):
    if override["scheme"] == "rivaGan":
        import cv2
        im = cv2.imread(str(clean_path))
        wm_im = encoder.encode(im, override["scheme"])
        return cv2.cvtColor(wm_im, cv2.COLOR_BGR2RGB).astype(np.float32)  # H,W,3 RGB, matches read_rgb255
    elif override["scheme"] == "trustmark":
        im = Image.open(clean_path).convert("RGB")
        wm_im = encoder.encode(im, override["secret"])
        return np.asarray(wm_im, dtype=np.float32)  # H,W,3 RGB
    elif override["scheme"] == "vine":
        size = override["img_size"]
        im = np.asarray(Image.open(clean_path).convert("RGB").resize((size, size), Image.BICUBIC))
        wm_im = encoder.embed([im], [override["message"]]).stego_img[0]
        return np.asarray(wm_im, dtype=np.float32)  # H,W,3 RGB, resized to img_size
    else:
        raise ValueError(f"unknown override scheme: {override['scheme']}")

def validate_zip(zp):
    with zipfile.ZipFile(zp, "r") as zf:
        names = zf.namelist()
    assert len(names) == 200, f"Expected 200 images, got {len(names)}"
    assert all("/" not in n for n in names), "No folders/subfolders allowed"
    assert sorted(names, key=lambda s: int(Path(s).stem)) == [f"{i}.png" for i in range(1, 201)]
    print("ZIP OK:", zp)


def build_submission(pref_weight=DEFAULT_PREF_WEIGHT, budget=DEFAULT_BUDGET, tag=None,
                      y_gain=0.85, chroma=1.2, use_overrides=True):
    import lpips as lpips_pkg
    lp_fn = lpips_pkg.LPIPS(net="alex").to(DEVICE).eval()
    direction_fn = make_blend(pref_weight)
    tag = tag or f"w{int(pref_weight * 100):03d}_b{int(budget * 1000):03d}" + ("_ovr" if use_overrides and GROUP_OVERRIDES else "")

    tmp = OUT_DIR / f"tmp_{tag}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    for g, a, b in CATEGORIES:
        override = GROUP_OVERRIDES.get(g) if use_overrides else None
        if override is not None:
            print(f"  [{tag}] {g}: known-scheme override ({override['scheme']})")
            encoder = load_override_encoder(override)
            for num in range(a, b + 1):
                clean_path = DATASET_DIR / "clean_targets" / f"{num}.png"
                cand = apply_group_override(encoder, override, clean_path)
                Image.fromarray(np.clip(np.rint(cand), 0, 255).astype(np.uint8), "RGB").save(
                    tmp / f"{num}.png", compress_level=6
                )
            continue

        pref, nlm_t = load_templates(g)
        w_hat = direction_fn(pref, nlm_t)
        for num in range(a, b + 1):
            target = read_rgb255(DATASET_DIR / "clean_targets" / f"{num}.png")
            cand = embed_to_budget(lp_fn, target, direction(target, w_hat, y_gain=y_gain, chroma=chroma), budget)
            Image.fromarray(np.clip(np.rint(cand), 0, 255).astype(np.uint8), "RGB").save(
                tmp / f"{num}.png", compress_level=6
            )
        print(f"  [{tag}] {g} done")

    zp = OUT_DIR / f"submission_{tag}.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(1, 201):
            z.write(tmp / f"{i}.png", arcname=f"{i}.png")
    validate_zip(zp)
    print(f"[{tag}] ZIP={zp}")
    return zp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref-weight", type=float, default=DEFAULT_PREF_WEIGHT,
                         help="0.0 = pure NLM+PLV, 1.0 = pure preference-model, default 0.35 (best blend ratio found)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--no-overrides", action="store_true",
                         help="disable known-scheme overrides (WM_2 RivaGAN); use the blend for all 8 groups (0.5465 instead of 0.6669)")
    args = parser.parse_args()

    build_and_cache_templates()
    build_submission(pref_weight=args.pref_weight, budget=args.budget, use_overrides=not args.no_overrides)
