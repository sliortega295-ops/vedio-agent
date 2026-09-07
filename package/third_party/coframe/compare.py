from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .metrics import reconstruction_metrics, temporal_gradient_relative_l2


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "latents" not in payload:
        raise ValueError(f"{path} is not a CoFrame latent payload")
    return payload


def _probe_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for event in metadata.get("transformer_events", []):
        probes.extend(event.get("probes", []))

    keys = (
        "spearman",
        "pearson",
        "prior_spearman",
        "defect_spearman",
        "spearman_gain_over_rhyme_prior",
        "anchor_context_error_mean",
        "anchor_context_delta_relative_l2",
        "block_delta_normalized_mse",
        "block_delta_relative_l2",
        "non_anchor_block_delta_relative_l2",
        "block_delta_frame_p95",
        "block_delta_frame_cvar10",
        "mesh_current_relative_l2",
        "mesh_rhyme_relative_l2",
        "mesh_fixed_relative_l2",
        "mesh_oracle_relative_l2",
        "mesh_current_nmse",
        "mesh_rhyme_nmse",
        "mesh_oracle_nmse",
        "mesh_headroom_recovery",
        "mesh_nmse_improvement_over_rhyme",
        "mesh_current_oracle_nmse_regret",
        "swap_prior_spearman",
        "swap_causal_spearman",
        "swap_post_observation_spearman",
        "swap_post_observation_gain_recovery",
        "swap_post_observation_regret",
        "swap_post_observation_normalized_regret",
        "swap_post_observation_top1_exact",
        "propagated_relative_l2_h1",
        "propagated_relative_l2_h3",
        "propagated_frame_cvar10_h1",
        "propagated_frame_cvar10_h3",
    )
    if not probes:
        return {"probe_count": 0, **{f"probe_{key}_mean": None for key in keys}}

    def mean_present(key: str) -> float | None:
        values = [float(item[key]) for item in probes if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "probe_count": len(probes),
        **{f"probe_{key}_mean": mean_present(key) for key in keys},
        # Compatibility aliases for result files produced by the first prototype.
        "probe_spearman_mean": mean_present("spearman"),
        "probe_pearson_mean": mean_present("pearson"),
        "probe_prior_spearman_mean": mean_present("prior_spearman"),
        "probe_defect_spearman_mean": mean_present("defect_spearman"),
        "probe_spearman_gain_mean": mean_present("spearman_gain_over_rhyme_prior"),
        "probe_anchor_context_error_mean": mean_present("anchor_context_error_mean"),
    }


def compare_payload(reference: dict[str, Any], candidate: dict[str, Any], name: str) -> dict[str, Any]:
    dense = reference["latents"].float()
    sparse = candidate["latents"].float()
    if dense.shape != sparse.shape or dense.ndim != 5:
        raise ValueError(f"Shape mismatch for {name}: {tuple(dense.shape)} vs {tuple(sparse.shape)}")

    dense_frames = dense.permute(0, 2, 1, 3, 4)
    sparse_frames = sparse.permute(0, 2, 1, 3, 4)
    endpoint = reconstruction_metrics(dense_frames, sparse_frames)
    cosine = F.cosine_similarity(dense.flatten(), sparse.flatten(), dim=0)
    dense_time = float(reference.get("metadata", {}).get("denoise_time_sec", 0.0))
    sparse_time = float(candidate.get("metadata", {}).get("denoise_time_sec", 0.0))

    result = {
        "method": name,
        "normalized_mse": endpoint["normalized_mse"],
        "relative_l2": endpoint["relative_l2"],
        "cosine": float(cosine.item()),
        "temporal_gradient_relative_l2": temporal_gradient_relative_l2(dense, sparse, frame_dim=2),
        "frame_error_mean": endpoint["frame_error_mean"],
        "frame_error_p95": endpoint["non_anchor_frame_error_p95"],
        "frame_error_cvar10": endpoint["non_anchor_frame_error_cvar10"],
        "frame_error_max": endpoint["frame_error_max"],
        "frame_error": endpoint["per_frame_global_normalized_rms"],
        "denoise_time_sec": sparse_time or None,
        "speedup_vs_dense": dense_time / sparse_time if dense_time > 0 and sparse_time > 0 else None,
        "initial_anchors": candidate.get("metadata", {}).get("initial_anchors"),
        "final_anchors": candidate.get("metadata", {}).get("final_anchors"),
        **_probe_summary(candidate.get("metadata", {})),
    }
    controller = candidate.get("metadata", {}).get("controller")
    result["refresh_count"] = len(controller.get("refresh_history", [])) if controller else 0
    result["accepted_refresh_count"] = (
        sum(1 for event in controller.get("refresh_history", []) if event.get("gain", 0.0) > 0)
        if controller
        else 0
    )
    return result


def _format(value: float | None, pattern: str) -> str:
    return "-" if value is None else format(value, pattern)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["method", "speedup", "end-L2", "motion-L2", "frame-tail", "mesh-NMSE", "recovery", "swap-rho", "prop-h1"]
    print("  ".join(f"{header:>12}" for header in headers))
    for row in rows:
        speedup = row["speedup_vs_dense"]
        values = [
            row["method"],
            "-" if speedup is None else f"{speedup:.3f}x",
            f"{row['relative_l2']:.5f}",
            f"{row['temporal_gradient_relative_l2']:.5f}",
            f"{row['frame_error_cvar10']:.5f}",
            _format(row.get("probe_mesh_current_nmse_mean"), ".6f"),
            _format(row.get("probe_mesh_headroom_recovery_mean"), ".3f"),
            _format(row.get("probe_swap_post_observation_spearman_mean"), ".3f"),
            _format(row.get("probe_propagated_relative_l2_h1_mean"), ".5f"),
        ]
        print("  ".join(f"{value:>12}" for value in values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare CoFrame latent runs against a dense Wan reference")
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "LATENTS_PT"), required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = _load(args.dense)
    rows = [compare_payload(reference, _load(Path(path)), name) for name, path in args.candidate]
    _print_table(rows)
    payload = {"dense": str(args.dense), "results": rows}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
