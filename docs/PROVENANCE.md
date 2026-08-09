# Provenance and code identity

## Snapshot source

The frozen experiment tree was copied read-only from:

```text
/data/jiayu/yongyan_liu/auto_research/research/sva_candidate_concurrency
```

on host `user-NF5280M6` after the completed formal run and parallel Nsys
captures. The surrounding `auto_research` checkout was on branch `research` at
commit `51cbfb2eacac516090395ea17e82a1a636df983a` with four working-tree entries;
the snapshot does not falsely claim that checkout was clean. The publication snapshot contains 147 files before adding these Git
presentation documents.

## Important hashes

| Object | SHA256 |
|---|---|
| User-provided source archive | `961954a43545f151f62269ab65dfbfbe67e877e6c8b4322218bd66e1cc8f1687` |
| Formal v3 benchmark harness recorded by the run | `5bee0f5516ae7495a65174ca55c6370101685f9668a731a18d81ad95577e781a` |
| Current harness after profile-mode-only extension | `89a0b37fd400a0902fceecb7f19d96eaea5f8d4aeb3c9124d3c0c7860122dc74` |
| Formal v3 launcher | `501597855335d9a94018f218d45adfe0d413bf24c4f6f5aa8502adca1e2fc0e7` |
| Frozen full snapshot archive | `8c0674892d26258215f5e11c7b87da814d753744581aa39f57fc6a4afe2cff6e` |
| Nsys SQLite analyzer | `8639d55435df8a8f761c20c395c56d19e4dd0844c66d7297a360577ac4ff0b00` |

The current harness differs from the formal v3 harness only by the retained
profiling-mode extension (`package/HARNESS_PROFILE_MODE.patch`). The formal
result's own metadata remains the authority for which hash produced the timing
numbers; the old result is not retroactively relabeled.

`manifests/source_extract_benchmark_files.sha256` records hashes at the initial
source extraction stage. Absolute paths embedded in historical manifests are
kept as provenance rather than rewritten.
