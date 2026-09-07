# Third-party notices

`package/third_party/wan/` and `agent/generation_backend/wan/` contain separate
minimal Wan2.1 Python source subsets. The first is frozen with the concurrency
artifact; the second supports SparseMAC warmup-latent resume. The agent entry
points under `agent/generation_backend/` are adapted from the same Wan2.1 code.
Wan2.1 is published by the Wan-Video project under Apache-2.0;
the upstream license is included at
`third_party_licenses/Wan2.1-LICENSE.txt` and
both `package/third_party/wan/LICENSE.txt` and
`agent/generation_backend/wan/LICENSE.txt`.

Official upstream: https://github.com/Wan-Video/Wan2.1

The vendored subset came from the user-provided experiment archive rather than
a retained Git checkout. Its exact upstream commit is therefore unknown and is
not fabricated. File-level hashes are provided in `manifests/` so this snapshot
can be compared with a chosen upstream revision.

No model weights are distributed in this repository or its release.
