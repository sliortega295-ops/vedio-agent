# Environment and Model Requirements

The package is code-only. The following assets must be supplied separately.

## Validated environment layout

The original H800 run used separate runtime roles:

| Role | Python/runtime | Purpose |
|---|---|---|
| Wan generation | private `wan_gen`-style Conda env | Wan2.1-T2V-1.3B/14B generation, sparse RoPE branches, storyboard extraction, API selector |
| LLaVA evaluation | the validated `gen` environment | official LLaVA-1.6-34B evaluator |
| CV evaluation | `/usr/bin/python` plus the private CV site | spatial, motion, numeracy evaluators |

Do not merge the LLaVA and CV dependency sets blindly. The historical
LLaVA stack pins older Torch/Transformers versions, while Wan generation uses
the newer Wan2.1 stack. On the shared server, use private environments under
the user's workspace and keep one GPU per process.

## Conda setup templates

These are setup templates, not commands run while creating this archive. Put
the environments under a user-owned directory and select the CUDA-compatible
PyTorch wheel for the installed driver:

```bash
conda create --prefix /home/lyy/conda_envs/tts_wan python=3.10 -y
conda run --prefix /home/lyy/conda_envs/tts_wan \
  python -m pip install -r requirements/generation.txt

conda create --prefix /home/lyy/conda_envs/tts_llava34b python=3.10 -y
conda run --prefix /home/lyy/conda_envs/tts_llava34b \
  python -m pip install -r requirements/llava34b.txt
```

The VLM selector can run in `tts_wan` because it uses only the standard
library, NumPy and OpenCV. The CV evaluator should use the already validated
private CV site/environment when available; installing `requirements/cv.txt`
alone does not build GroundingDINO's compiled extension.

## Generation dependencies

See `requirements/generation.txt`. The important packages are:

- Python 3.10+
- PyTorch 2.4+ and matching torchvision
- `opencv-python`, `imageio`, `imageio-ffmpeg`, `numpy<2`
- `diffusers`, `transformers`, `tokenizers`, `accelerate`
- `easydict`, `ftfy`, `tqdm`
- the CUDA-compatible attention/runtime packages required by the local Wan
  installation

## VLM selector dependencies

`select_vlm.py` uses the Python standard library plus:

- `opencv-python`
- `numpy`

It calls an OpenAI-compatible HTTP endpoint and does not require an OpenAI SDK.
The requested endpoint, model and key are in `config/vlm.env`.

## Official evaluator dependencies

- LLaVA-1.6-34B runtime: use the validated `gen` environment and the pinned
  dependencies listed in `requirements/llava34b.txt`.
- CV evaluator runtime: use the private CV environment/site and
  `requirements/cv.txt`; the official workflow previously invoked it with
  `/usr/bin/python` and a CV-only `PYTHONPATH`.
- `ffmpeg`/`ffprobe` must be available for video validation and encoding.

## Required model assets

1. **Generation and model comparisons**
   - `Wan2.1-T2V-1.3B` (the packaged generator's default backend and RoPE
     ablation model).
   - `Wan2.1-T2V-14B` (the same Wan family, also accepted by the dense and
     sparse RoPE generator for the 14B scaling comparison).
   - `HunyuanVideo-1.5-8B` (the 8B scaling comparison model).
   - The Wan checkpoint directories must contain the T5 encoder, VAE, DiT and
     tokenizer files expected by `wan.WanT2V`. Example paths are
     `/gfs/space/jiayu/models/Wan2.1-T2V-1.3B` and
     `/gfs/space/jiayu/models/Wan2.1-T2V-14B`.
   - HunyuanVideo uses its upstream Hunyuan inference environment/backend;
     its weight path should be recorded separately, for example
     `/gfs/space/jiayu/models/HunyuanVideo-1.5-8B`. The extracted sparse RoPE
     runner is intentionally Wan-family-only and does not bundle Hunyuan's
     separate inference backend, agent, or prompt stack.

2. **Official LLaVA metrics**
   - `LLaVA-1.6-34B`
   - Example path: `/gfs/space/jiayu/eval_models/llava-v1.6-34b`.

3. **Official CV metrics**
   - GroundingDINO SwinT-OGC checkpoint:
     `Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth`
   - SAM ViT-H checkpoint:
     `Grounded-Segment-Anything/sam_vit_h_4b8939.pth`
   - Depth Anything ViT-L/14 checkpoint, passed with `--depth-weights`:
     `depth_anything_vitl14.pth`

4. **VLM selection API**
   - Model: `gpt-5.6-luna`
   - Base URL: `https://bmapi.020212.xyz/v1`
   - This is an API service, not a local weight.

Model weights are intentionally not copied into the archive because they are
large and should remain in the shared `/data/models` or `/gfs/space/jiayu/models`
store, with project-local symlinks if needed.
