# Wan Generation Backend

These entry points make the SparseMAC agent self-contained at the source-code
level:

- `generate_sparse.py` runs full-frame warmup followed by key-latent denoising,
  decodes the sparse latent sequence directly, and saves the full warmup latent
  plus metadata for each independent prompt/seed candidate.
- `generate.py` resumes an 81-frame Wan T2V trajectory from that candidate's
  warmup latent and completes full denoising.

Both scripts import the private copy under `generation_backend/wan` by default,
so modifying the agent cannot invalidate the frozen concurrency artifact under
`package/`. Set `SPARSEMAC_WAN_SOURCE_ROOT` to use another compatible Wan source
directory. The full-video entry point can load Wan2.1 T2V 1.3B or 14B, while
the sparse entry point and active agent currently use `t2v-1.3B` to preserve
the recorded experiment contract.

The matching resume support is implemented in
`generation_backend/wan/text2video.py`. A warmup checkpoint is rejected when
its prompt, seed, dimensions, sampling schedule, or latent shape does not match
the requested full generation.
