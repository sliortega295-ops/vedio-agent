# Third-party notices

`package/third_party/wan/` contains a minimal Wan2.1 Python source subset used by
the benchmark. Wan2.1 is published by the Wan-Video project under Apache-2.0;
the upstream license is included at
`third_party_licenses/Wan2.1-LICENSE.txt` and
`package/third_party/wan/LICENSE.txt`.

Official upstream: https://github.com/Wan-Video/Wan2.1

The vendored subset came from the user-provided experiment archive rather than
a retained Git checkout. Its exact upstream commit is therefore unknown and is
not fabricated. File-level hashes are provided in `manifests/` so this snapshot
can be compared with a chosen upstream revision.

No model weights are distributed in this repository or its release.

## CADense / CoFrame

`package/third_party/coframe/` and `package/tests/cadense/` vendor source/tests
from https://github.com/sliortega295-ops/CADense at commit
`ceddd6f08ec4821acac1210da8044d3015f1601b`, under Apache-2.0. The upstream Python
package name is `coframe`. Its original NOTICE and an Apache-2.0 license text are
included. See `package/third_party/coframe/VENDOR.md` and `upstream_sha256.json`
for attribution, exact source hashes, and local integration modifications.
