# Watermark Forgery Attack - Team XIII

This README explains only how to recreate that result. The recipe combines
four things:

1. 5 unidentified groups (WM_1, WM_3, WM_5, WM_6, WM_8): extract a
   watermark direction two independent ways (a pretrained "preference model"
   discriminator, and NLM-residual + phase-locking-value averaging), blend
   them 35% pref / 65% nlm, and embed the blend into each clean target with
   a YCbCr chroma boost, a texture-adaptive mask, and a binary search that
   holds LPIPS to a fixed budget of 0.022.
2. WM_2 (identified: RivaGAN): decoding WM_2's 25 source images with the
   open-source `imwatermark` library's RivaGAN detector gives the exact same
   32-bit message on 22/25 images (every other group decodes as pure noise -
   25/25 unique). Encoding that recovered message directly onto WM_2's clean
   targets with the real RivaGAN encoder gives ~98.9% mean bit accuracy at
   ~0.016 mean LPIPS.
3. WM_4 (identified: VINE): decoding WM_4's 25 source images with the
   `watermarklab` library's VINE-R detector gives the exact same 100-bit
   message on all 25/25 images. Encoding that message directly onto WM_4's
   clean targets gives 24/25 exact round-trip matches (the 1 miss is still
   99% bit accuracy) at ~0.0075 mean LPIPS.
4. WM_7 (identified: TrustMark): decoding WM_7's 25 source images with
   the `trustmark` library's Q-variant detector gives the exact same secret
   on all 25/25 images. Encoding that secret directly onto WM_7's clean
   targets gives 25/25 exact round-trip matches at ~0.0014 mean LPIPS -
   near-perfect.

All three identified-scheme cases beat residual-estimation by a wide margin,
since they use the actual encoder instead of an estimate.

## 1. Environment

py -3.11 -m pip install torch torchvision numpy scipy opencv-python pillow lpips omegaconf timm einops requests invisible-watermark onnxruntime trustmark watermarklab


Package notes:
- `invisible-watermark` is the pip name for the `imwatermark` module (RivaGAN override).
- `trustmark` is Adobe's Content Credentials watermark library (WM_7 override).
- `watermarklab` bundles the VINE-R model used for the WM_4 override.
- If `pip install trustmark` fails with a `UnicodeDecodeError` (a Windows-only
  encoding bug in its setup.py), retry with: set PYTHONUTF8=1 && py -3.11 -m pip install trustmark

- `watermarklab`'s model downloads call an outdated `hf_hub_download(use_auth_token=True)`
  argument that current `huggingface_hub` rejects (and, worse, interprets as
  "require a real login token" for what is actually a public repo).
  `watermark_forge.py` patches this automatically at import time
  (`_patch_hf_hub_download()`) - no action needed, just noting why it's there.
- The VINE override needs a CUDA GPU - its embed/extract path has a
  device-handling bug in the library that crashes on CPU-only.

Requires a CUDA GPU for the preference-model extraction too (400
gradient-ascent steps per image batch; expect ~1.5 hours total on an 8GB GPU
such as an RTX 3060 Ti, for the 5 non-overridden groups). All three scheme
overrides are otherwise fast (seconds to low minutes, mostly one-time model
downloads on first run).

## 2. Get the dataset

Download `Dataset.zip` (link in the assignment's HuggingFace reference) and
either leave it as `Dataset.zip` in this folder, or extract it here so that
`Dataset/clean_targets/` and `Dataset/watermarked_sources/` exist directly
under this folder.

## 3. Get the preference-model dependency

`watermark_forge.py` calls into Meta's `wmforger` code and pretrained
checkpoint (from "Transferable Black-Box One-Shot Forging of Watermarks via
Image Preference Models", Soucek et al., NeurIPS 2025). It is not vendored
here to keep this folder small - fetch it once:

git clone https://github.com/facebookresearch/videoseal.git vendor/videoseal-main
cd vendor/videoseal-main/wmforger
wget https://dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth
cd ../../..


Resulting layout expected by `watermark_forge.py`:

vendor/videoseal-main/wmforger/convnext_pref_model.pth
vendor/videoseal-main/wmforger/configs/extractor.yaml
vendor/videoseal-main/wmforger/wmforger/            (python package)


## 4. Recreate the best result


py -3.11 watermark_forge.py

With no arguments this builds the winning recipe (35/65 blend for 5 groups +
RivaGAN override for WM_2 + VINE override for WM_4 + TrustMark override for
WM_7, budget 0.022) and writes `outputs/submission_w035_b022_ovr.zip` - the
200-image zip to submit (verified pixel-identical to the actually-submitted
zip that scored 0.7819839470202565). First run also extracts and caches
per-group watermark templates under `templates_cache/` (the slow part, ~1.5
hours for the 5 non-overridden groups); reruns with a different
`--pref-weight` or `--budget` reuse the cache and take under a minute.

Other options (for reference):

py -3.11 watermark_forge.py --no-overrides       # disable all 3 overrides, pure blend for all 8 groups: 0.5465
py -3.11 watermark_forge.py --pref-weight 0      # NLM+PLV only for non-overridden groups: weaker than 0.35
py -3.11 watermark_forge.py --pref-weight 1      # preference-model only for non-overridden groups: weaker than 0.35
py -3.11 watermark_forge.py --budget 0.016       # tighter LPIPS budget (no improvement, tested pre-override)
py -3.11 watermark_forge.py --budget 0.030       # looser LPIPS budget (no improvement, tested pre-override)


To re-derive a group's scheme/message yourself rather than trust the
hardcoded values in `GROUP_OVERRIDES`:
python
watermark_forge.discover_rivagan_message("WM_2")     #the 32-bit message or None
watermark_forge.discover_vine_message("WM_4")        #the 100-bit message dict or None
watermark_forge.discover_trustmark_secret("WM_7", "Q") #the secret dict or None

All three return `None` if a group doesn't show strong enough consensus for
that scheme - which is what happens for the other 5 groups against every
scheme tried so far: DWT-DCT, DwtDctSvd, RivaGAN, TrustMark (Q/C/B), VINE,
StegaStamp, InvisMark, StableSignature, `blind_watermark`, SteganoGAN, and
Meta's entire VideoSeal family (v1.0, v0.0, PixelSeal, ChunkySeal). They
remain unidentified and use the residual-blend estimate. TreeRing and
GaussianShading were not testable - both need the gated
`stabilityai/stable-diffusion-2-1-base` model, which requires a HuggingFace
account that has accepted Stability AI's license.

## 5. Submit


WM_API_KEY=your_team_key py -3.11 submit.py outputs/submission_w035_b022_ovr.zip


Add `--wait` to have it poll every 20s and submit automatically once the
60-minute rate limit clears, instead of failing immediately on a 429.
