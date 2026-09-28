
import sys, os, subprocess, urllib.request, base64, hashlib, json, time
from pathlib import Path

def log(msg):
    print(msg, flush=True)

log("=== QWEN21 VIGGLE-TURBO GEN START ===")
# --- S3 self-reporting (pod has no SSH/logs access from sandbox) ---
import urllib.request as _urlreq
S3_URLS = {'status.json': 'https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com/qwen-turbo-test/status.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVECDXC7QH4MSP3ZW%2F20260928%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260928T021015Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=2e74b58e966312d5ca932ed2d543683a8e5e106809883b1be5af7ecf7282339b', 'scene_001.png': 'https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com/qwen-turbo-test/scene_001.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVECDXC7QH4MSP3ZW%2F20260928%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260928T021017Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=438ea35fcdc7a7b0b27a5e89a2054fb6e024b7197d177acc17f1e765e5157e56', 'scene_002.png': 'https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com/qwen-turbo-test/scene_002.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVECDXC7QH4MSP3ZW%2F20260928%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260928T021018Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6413d3e8d4b2928f48069b0abfd933ac5617034be5db91e8460f97af0bcd5b39', 'scene_003.png': 'https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com/qwen-turbo-test/scene_003.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVECDXC7QH4MSP3ZW%2F20260928%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260928T021019Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=15b14ba19d3e524594585106be10f21164bd172ef49c93e91fb64ffbec9eb77e', 'scene_004.png': 'https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com/qwen-turbo-test/scene_004.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVECDXC7QH4MSP3ZW%2F20260928%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260928T021021Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=467127e9604869c57adb115b293267d69ca1a8c111d849cc02095b1e918ceaf3', 'scene_005.png': 'https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com/qwen-turbo-test/scene_005.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVECDXC7QH4MSP3ZW%2F20260928%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260928T021022Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=1face96c1bea9c72bfbc4b3dce93826d1e92208ab322d663c4d86f9ccb294709', 'manifest.json': 'https://nishan-qwen-model-cache.s3.ca-central-1.amazonaws.com/qwen-turbo-test/manifest.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVECDXC7QH4MSP3ZW%2F20260928%2Fca-central-1%2Fs3%2Faws4_request&X-Amz-Date=20260928T021023Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=3f2d20ea0aafc1dad8609dbf63a4704615d4c07016fc392e77d86802746b13c8'}
def s3_put(key, data, ctype="application/octet-stream"):
    try:
        req = _urlreq.Request(S3_URLS[key], data=data, method="PUT",
                              headers={"Content-Type": ctype})
        with _urlreq.urlopen(req, timeout=180) as r:
            log("S3_PUT_OK " + key)
    except Exception as e:
        log("S3_PUT_FAIL " + key + " " + str(e)[:150])
def s3_status(stage, extra=None):
    d = {"stage": stage, "ts": time.time()}
    if extra: d.update(extra)
    s3_put("status.json", json.dumps(d).encode(), "application/json")
s3_status("started")
# --- end S3 ---

WORK = Path("/root/qwen-test")
OUT = WORK / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# [1] pip installs (diffusers MUST come from git: no tagged release has QwenImage21Pipeline)
log("[1/6] pip installs...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages",
                "transformers", "accelerate", "pillow", "hf_transfer",
                "sentencepiece"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages",
                "git+https://github.com/huggingface/diffusers"], check=True)
log("pip done")
s3_status("pip_done")

# [2] reference image from gist (512px derivative of canonical reference)
log("[2/6] fetching reference...")
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
s3_status("ref_ok")

# [3] imports + turbo loader (verbatim from Viggle model card)
from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
import torch
import diffusers
log("diffusers " + diffusers.__version__)
log("[3/6] loading Qwen/Qwen-Image-2.1 base ...")
t0 = time.time()
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
log("base loaded %.1fs" % (time.time() - t0))
log("[4/6] loading Viggle turbo LoRA r256 ...")
t0 = time.time()
pipe.load_lora_weights("Viggle/Qwen-Image-2.1-viggle-turbo",
    weight_name="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors")
log("lora loaded %.1fs" % (time.time() - t0))
log("[5/6] swapping in shipped scheduler ...")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "Viggle/Qwen-Image-2.1-viggle-turbo", subfolder="scheduler")
# 24GB VRAM strategy: CPU offload (33GB base + 1.3GB LoRA cannot sit on 24GB)
pipe.enable_model_cpu_offload()
log("MODEL_LOADED (turbo)")
s3_status("model_loaded")

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
SIGMAS_8 = [1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125]

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
s3_status("scene_done", {"scene": num, "seconds": round(dt, 1)})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
s3_put("manifest.json", json.dumps(manifest, indent=2).encode(), "application/json")

# [6] emit results as base64 lines for log recovery
log("[6/6] emitting results...")
for m in manifest:
    p = OUT / m["file"]
    name = p.stem
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    log("IMGB64_START::" + name + "::" + sha)
    b64 = base64.b64encode(p.read_bytes()).decode()
    for seq in range(0, len(b64), 60000):
        log("IMGB64::" + name + "::" + str(seq // 60000) + "::" + b64[seq:seq + 60000])
    log("IMGB64_END::" + name)
# [7] weight manifest for S3 cache handoff (path + size, via logs)
log("[7/7] weight manifest...")
hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
log("WEIGHTS_ROOT::" + hf_home)
total = 0
nfiles = 0
if os.path.isdir(hf_home):
    log("WEIGHTS_START")
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
            log("WEIGHTS::" + rel + "::" + str(sz))
    log("WEIGHTS_END::%d::%d" % (nfiles, total))
else:
    log("WEIGHTS_MISSING::" + hf_home)
s3_status("all_done", {"manifest": manifest})
log("ALL_DONE")
