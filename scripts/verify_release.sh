#!/usr/bin/env bash
set -euo pipefail
ARTIFACT_DIR=${1:-.}
ARCHIVE="$ARTIFACT_DIR/sva_candidate_concurrency_20260809_full.tar.zst"
ARCHIVE_SUM="$ARTIFACT_DIR/sva_candidate_concurrency_20260809_full.tar.zst.sha256"
FILE_SUM="$ARTIFACT_DIR/sva_candidate_concurrency_20260809_FILES.sha256"
(
  cd "$ARTIFACT_DIR"
  sha256sum -c "$(basename "$ARCHIVE_SUM")"
)
zstd -t "$ARCHIVE"
TMP=$(mktemp -d)
trap 'find "$TMP" -depth -delete' EXIT
tar --zstd -xf "$ARCHIVE" -C "$TMP"
(
  cd "$TMP/sva_candidate_concurrency"
  sha256sum -c "$FILE_SUM"
)
echo "PASS: archive and every frozen snapshot file match their SHA256"
