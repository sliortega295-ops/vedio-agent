# Publication validation

The publication payload was checked before commit and upload.

| Check | Result |
|---|---|
| Frozen archive SHA256 | PASS: `8c0674892d26258215f5e11c7b87da814d753744581aa39f57fc6a4afe2cff6e` |
| zstd integrity | PASS, 319,518,720 decompressed bytes |
| Internal file manifest | PASS, 147/147 files |
| Structured JSON parsing | PASS, 17 files |
| JSONL parsing | PASS, 9 files |
| CSV parsing | PASS, 29 files |
| Python byte compilation | PASS |
| Benchmark and sparse-generator `--help` | PASS |
| Shell syntax (`bash -n`) | PASS |
| High-risk credential scan | PASS, zero matches in 245 text files; 35 binary/large files handled by hash validation |
| Nsys SQLite re-analysis admission | PASS for N9 and N6 |
| Re-analysis key-metric equality | PASS for overlap, idle, top-10 coverage, kernel sum, and busy union |
| Git whitespace check | PASS |

## Not executed locally

No GPU benchmark was rerun during publication. The authoritative GPU evidence is
the frozen remote result tree and its hashes. Model weights are not included and
were not copied to the publication machine.
