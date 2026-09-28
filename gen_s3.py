import sys, os, subprocess, urllib.request, base64, hashlib, json, time
from pathlib import Path

def log(msg):
    print(msg, flush=True)

log("=== QWEN21 VIGGLE-TURBO GEN START (boto3 edition) ===")

# --- S3 via boto3 + temporary STS creds (12h, bucket-scoped) ---
# Creds live ONLY at this random-key public-read S3 object; they auto-expire.
CREDS_URL = ("https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com"
             "/_creds/671d9023ee969715af262becbad68aab.json")
BUCKET = "nishan-qwen-model-cache"
PREFIX = "qwen-turbo-test/"

WORK = Path("/root/qwen-test")
OUT = WORK / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# [1] pip installs. diffusers MUST come from zip: no tagged release has
# QwenImage21Pipeline, and the pod has no `git` binary (git+https fails).
log("[1/7] pip installs...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages",
                "transformers", "accelerate", "pillow", "hf_transfer",
                "sentencepiece", "boto3"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages",
                "https://github.com/huggingface/diffusers/archive/refs/heads/main.zip"],
               check=True)
log("pip done")

# [2] fetch STS creds (public S3 URL, no signing needed) and build boto3 client
log("[2/7] fetching STS creds...")
with urllib.request.urlopen(CREDS_URL, timeout=60) as r:
    creds = json.load(r)
log("creds OK, expires " + str(creds.get("expires")))
import boto3
s3 = boto3.client("s3", region_name="ca-central-1",
                  aws_access_key_id=creds["aws_access_key_id"],
                  aws_secret_access_key=creds["aws_secret_access_key"],
                  aws_session_token=creds["aws_session_token"])

def s3_put(key, data, ctype="application/octet-stream"):
    try:
        s3.put_object(Bucket=BUCKET, Key=PREFIX + key, Body=data,
                      ContentType=ctype)
        log("S3_PUT_OK " + key)
        return True
    except Exception as e:
        log("S3_PUT_FAIL " + key + " " + str(e)[:200])
        return False

# [3] reference image from gist (512px derivative of canonical reference)
log("[3/7] fetching reference...")
urllib.request.urlretrieve(
    "https://gist.githubusercontent.com/foreverbulking/b73fe5e53af32b69a501e5e5680813e7/raw/ref_512.b64",
    str(WORK / "ref.b64"))
raw = base64.b64decode(open(WORK / "ref.b64").read().strip())
open(WORK / "character-reference.png", "wb").write(raw)
h = hashlib.md5(raw).hexdigest()
log("ref md5: " + h)
assert h == "681228db2c60545c22bb870e99f0ef5e", "REF HASH MISMATCH"
from PIL import Image
im = Image.open(WORK / "character-reference.png")
log("ref dims: " + str(im.size) + " " + im.mode)
assert im.size == (512, 512), "REF DIM MISMATCH"
log("REF_OK")

# [4] imports + turbo loader (verbatim from Viggle model card)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
import torch
import diffusers
log("diffusers " + diffusers.__version__)
log("[4/7] loading Qwen/Qwen-Image-2.1 base ...")
t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1",
                                           dtype=torch.bfloat16)
log("base loaded %.1fs" % (time.time() - t0))
log("[5/7] loading Viggle turbo LoRA r256 ...")
t0 = time.time()
pipe.load_lora_weights("Viggle/Qwen-Image-2.1-viggle-turbo",
    weight_name="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors")
log("lora loaded %.1fs" % (time.time() - t0))
log("[6/7] swapping in shipped scheduler ...")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "Viggle/Qwen-Image-2.1-viggle-turbo", subfolder="scheduler")
# 24GB VRAM strategy: CPU offload (33GB base + 1.3GB LoRA cannot sit on 24GB)
pipe.enable_model_cpu_offload()
log("MODEL_LOADED (turbo)")

HOUSE_STYLE = ("flat 2D cartoon illustration, thick dark-brown outlines, simple shapes. "
    "The main character is the colorless hero from the attached reference image and must look "
    "identical to it: completely all-white face, all-white body, all-white hands, white hair "
    "with a prominent side-swept tuft, dot eyes, simple smiling mouth, no beard, no stubble, "
    "teal shirt, dark pants. Everything else in the scene is full color and richly detailed: "
    "environments, backgrounds, objects, secondary characters. Static, animation-friendly "
    "composition, 16:9 landscape framing, one clear focal concept. "
    "No on-image text, no watermark, no logo. Exactly one standalone image, never a grid or collage.")
SCENES = [
    ("001", "Wide establishing shot of a bright car showroom, rows of gleaming sedans under white ceiling lights, the colorless hero standing small near the entrance."),
    ("002", "Close-up detail of the colorless hero shoes squeaking on polished tile, mirrored reflections of cars stretching around him."),
    ("003", "Medium shot of the colorless hero pausing beside a shiny sedan, hands in pockets, studying it with a cautious lean."),
    ("004", "Over-the-shoulder shot from behind the colorless hero toward a smiling full-color salesman in a suit, clipboard tucked under his arm."),
    ("005", "Close-up of a hand placing a car key into the colorless hero open palm, bright showroom lights overhead."),
]
SIGMAS_6 = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]

ref_img = Image.open(WORK / "character-reference.png").convert("RGB")
manifest = []

def gen_scene(num, visual, steps, sigmas, seed):
    prompt = visual + " " + HOUSE_STYLE
    t0 = time.time()
    # turbo: no CFG (true_cfg_scale=1.0), no negative prompt, LoRA scale stays 1.0
    kwargs = dict(prompt=prompt, width=1344, height=768,
                  num_inference_steps=steps, sigmas=sigmas,
                  true_cfg_scale=1.0,
                  generator=torch.Generator("cuda").manual_seed(seed))
    try:
        result = pipe(image=[ref_img], **kwargs)
    except TypeError as e:
        log("scene " + num + " image-list fallback: " + str(e)[:120])
        try:
            result = pipe(image=ref_img, **kwargs)
        except TypeError as e2:
            log("scene " + num + " kwarg fallback: " + str(e2)[:160])
            # drop turbo-specific kwargs one at a time if the pipeline disagrees
            k2 = dict(kwargs)
            for drop in ("sigmas", "true_cfg_scale"):
                k2.pop(drop, None)
            result = pipe(image=ref_img, **k2)
    return result.images[0], time.time() - t0

# [7] generate 5 scenes, upload each PNG immediately (deliverables first)
for i, (num, visual) in enumerate(SCENES):
    image, dt = gen_scene(num, visual, 6, SIGMAS_6, 42 + i)
    path = OUT / ("scene_" + num + ".png")
    image.save(path)
    s3_put("scene_" + num + ".png", path.read_bytes(), "image/png")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    manifest.append({"scene": num, "file": path.name, "seconds": round(dt, 1),
                     "steps": 6, "size": list(image.size),
                     "bytes": path.stat().st_size, "sha256": sha})
    log("SCENE_DONE::%s::%.1fs::6step::%s::%dB" % (num, dt, image.size, path.stat().st_size))
manifest_json = json.dumps(manifest, indent=2)
(OUT / "manifest.json").write_text(manifest_json)
s3_put("manifest.json", manifest_json.encode(), "application/json")

# [8] weight manifest: log every file, then upload all weight files best-effort
log("[8/8] weight manifest...")
hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
log("WEIGHTS_ROOT::" + hf_home)
total = 0
nfiles = 0
wfiles = []
if os.path.isdir(hf_home):
    for root, dirs, files in os.walk(hf_home):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, hf_home)
            total += sz
            nfiles += 1
            wfiles.append((fp, rel, sz))
            log("WEIGHTS::" + rel + "::" + str(sz))
    log("WEIGHTS_END::%d::%d" % (nfiles, total))
else:
    log("WEIGHTS_MISSING::" + hf_home)

log("uploading %d weight files (%.1f GB) best-effort..." % (nfiles, total / 1e9))
wok, wfail, wbytes = 0, 0, 0
for fp, rel, sz in wfiles:
    try:
        with open(fp, "rb") as f:
            s3.put_object(Bucket=BUCKET, Key=PREFIX + "weights/" + rel, Body=f)
        wok += 1
        wbytes += sz
        if wok % 25 == 0:
            log("weights progress %d/%d" % (wok, nfiles))
    except Exception as e:
        wfail += 1
        log("W_PUT_FAIL " + rel + " " + str(e)[:120])
log("WEIGHTS_UPLOADED::%d::%d::%d" % (wok, wfail, wbytes))

log("ALL_DONE")
