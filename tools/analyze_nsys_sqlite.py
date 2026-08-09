#!/usr/bin/env python3
"""Fail-closed, timeline-only analysis for Phase-2 E016 Nsight exports.

The script deliberately makes no Tensor-Core/SM/DRAM SOL claim.  CUDA work is
attributed through the process-scoped chain

    NVTX range -> CUDA API globalTid -> (globalPid, correlationId) -> GPU op.

Only CUDA launches whose complete CPU API interval is nested in one of the
closed request step ranges 20..24 are analysed.  The kernels selected through
those process-scoped launch correlations keep their complete GPU intervals:
CUDA is asynchronous, so a selected kernel is allowed to finish after its CPU
NVTX step has popped.  A wider registered NVTX capture trigger may include
guard steps 19/25; their launches and kernels are excluded here.  For a
two-request trace, ``request_overlap_fraction`` means the intersection of the
two *request-attributed, selected-step* kernel-busy unions divided by the
selected-step kernel-busy union.  It is not generic same-request/auxiliary-
stream concurrency.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MASK64 = (1 << 64) - 1
GLOBAL_PID_MASK = 0xFFFFFFFFFF000000
DEFAULT_FIRST_STEP = 20
DEFAULT_LAST_STEP = 24
REQUEST_RE = re.compile(r"(?:^|/)request=([^/]+)(?:/|$)")
STEP_RE = re.compile(
    r"(?:^|/)request=([^/]+)/step=([0-9]+)/(conditional|unconditional)/dit(?:/|$)"
)
OPERATOR_RE = re.compile(
    r"(?:^|/)request=([^/]+)/(?:block_group=([0-9]+)/block=([0-9]+)/)?"
    r"(self_attention|cross_attention|mlp)(?:/|$)"
)
LAUNCH_RE = re.compile(r"^(?:cudaLaunch|cuLaunch)")
SYNC_RE = re.compile(r"(?:Synchronize|WaitEvent|StreamWait)", re.I)
BLOCKING_MEMCPY_RE = re.compile(r"^(?:cuda|cu)Memcpy(?!Async)(?:_v[0-9]+)?$")


class AdmissionError(RuntimeError):
    """The trace cannot support the requested mechanism inference."""


def _tables(db: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _cols(db: sqlite3.Connection, table: str) -> set[str]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", table):
        raise ValueError(f"unsafe table name {table!r}")
    return {str(row[1]) for row in db.execute(f'PRAGMA table_info("{table}")')}


def _u64(value: int) -> int:
    return int(value) & MASK64


def _pid_from_global_tid(value: int) -> int:
    return (_u64(value) >> 24) & 0x00FFFFFF


def _global_pid_from_tid(value: int) -> int:
    return _u64(value) & GLOBAL_PID_MASK


def _normalize_global_pid(value: int) -> int:
    return _u64(value) & GLOBAL_PID_MASK


def _percentile(values: Sequence[float | int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    if not ordered:
        return []
    out = [ordered[0]]
    for start, end in ordered[1:]:
        old_start, old_end = out[-1]
        if start <= old_end:
            out[-1] = (old_start, max(old_end, end))
        else:
            out.append((start, end))
    return out


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(intervals))


def _intersection_ns(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> int:
    a, b = _merge(left), _merge(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _active_at_least_ns(intervals: Iterable[tuple[int, int]], threshold: int) -> int:
    events: list[tuple[int, int]] = []
    for start, end in intervals:
        if end > start:
            # End before start at equal timestamps: touching intervals do not overlap.
            events.extend(((int(start), 1), (int(end), -1)))
    events.sort(key=lambda item: (item[0], item[1]))
    active = total = 0
    last: int | None = None
    for timestamp, delta in events:
        if last is not None and active >= threshold:
            total += timestamp - last
        active += delta
        last = timestamp
    return total


def _gaps(intervals: Iterable[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    clipped = _merge((max(start, s), min(end, e)) for s, e in intervals)
    cursor = start
    result: list[tuple[int, int]] = []
    for left, right in clipped:
        if left > cursor:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return result


def _strings(db: sqlite3.Connection, table_set: set[str]) -> dict[int, str]:
    if "StringIds" not in table_set:
        return {}
    columns = _cols(db, "StringIds")
    value_col = "value" if "value" in columns else "str" if "str" in columns else None
    if value_col is None:
        return {}
    return {
        int(identifier): str(value)
        for identifier, value in db.execute(f'SELECT id,"{value_col}" FROM StringIds')
    }


def _resolved_name(raw: Any, string_map: Mapping[int, str]) -> str:
    if isinstance(raw, int):
        return str(string_map.get(raw, raw))
    return str(raw or "<unknown>")


@dataclass(frozen=True)
class NvtxRange:
    rowid: int
    start: int
    end: int
    global_tid: int
    global_pid: int
    pid: int
    text: str
    request: str | None


def _load_nvtx(
    db: sqlite3.Connection, table_set: set[str], string_map: Mapping[int, str]
) -> list[NvtxRange]:
    if "NVTX_EVENTS" not in table_set:
        raise AdmissionError("trace has no NVTX_EVENTS table")
    columns = _cols(db, "NVTX_EVENTS")
    required = {"start", "end", "globalTid"}
    if not required.issubset(columns):
        raise AdmissionError(f"NVTX_EVENTS lacks {sorted(required - columns)}")
    selected = ["rowid", "start", "end", "globalTid"]
    selected.extend(name for name in ("textId", "text") if name in columns)
    ranges: list[NvtxRange] = []
    query = f'SELECT {",".join(selected)} FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start'
    for row in db.execute(query):
        item = dict(row)
        raw = item.get("textId")
        text = _resolved_name(raw, string_map) if raw is not None else str(item.get("text") or "")
        # Some exporters populate text even when textId is NULL.
        if (not text or text == "<unknown>") and item.get("text") is not None:
            text = str(item["text"])
        global_tid = _u64(int(item["globalTid"]))
        match = REQUEST_RE.search(text)
        ranges.append(
            NvtxRange(
                rowid=int(item["rowid"]),
                start=int(item["start"]),
                end=int(item["end"]),
                global_tid=global_tid,
                global_pid=_global_pid_from_tid(global_tid),
                pid=_pid_from_global_tid(global_tid),
                text=text,
                request=match.group(1) if match else None,
            )
        )
    return ranges


def _load_api_table(
    db: sqlite3.Connection,
    table_set: set[str],
    string_map: Mapping[int, str],
    table: str,
    source: str,
) -> list[dict[str, Any]]:
    if table not in table_set:
        return []
    columns = _cols(db, table)
    required = {"start", "end", "correlationId", "globalTid"}
    if not required.issubset(columns):
        return []
    name_col = "nameId" if "nameId" in columns else "name" if "name" in columns else None
    selected = ["rowid", "start", "end", "correlationId", "globalTid"]
    if name_col:
        selected.append(name_col)
    records: list[dict[str, Any]] = []
    for row in db.execute(f'SELECT {",".join(selected)} FROM "{table}"'):
        item = dict(row)
        global_tid = _u64(int(item["globalTid"]))
        raw = item.get(name_col) if name_col else None
        records.append(
            {
                "rowid": int(item["rowid"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
                "correlation_id": int(item["correlationId"]),
                "global_tid": global_tid,
                "global_pid": _global_pid_from_tid(global_tid),
                "pid": _pid_from_global_tid(global_tid),
                "name": _resolved_name(raw, string_map),
                "source": source,
            }
        )
    return records


def _load_kernels(
    db: sqlite3.Connection, table_set: set[str], string_map: Mapping[int, str]
) -> list[dict[str, Any]]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in table_set:
        raise AdmissionError("trace has no CUDA kernel table")
    columns = _cols(db, table)
    required = {"start", "end", "correlationId", "contextId", "streamId", "globalPid"}
    if not required.issubset(columns):
        raise AdmissionError(f"kernel table lacks {sorted(required - columns)}")
    name_col = next(
        (name for name in ("demangledName", "shortName", "mangledName", "name") if name in columns),
        None,
    )
    selected = [
        "rowid",
        "start",
        "end",
        "correlationId",
        "contextId",
        "streamId",
        "globalPid",
    ]
    selected.extend(name for name in ("deviceId", "greenContextId") if name in columns)
    if name_col:
        selected.append(name_col)
    result: list[dict[str, Any]] = []
    for row in db.execute(f'SELECT {",".join(selected)} FROM "{table}"'):
        item = dict(row)
        raw = item.get(name_col) if name_col else None
        result.append(
            {
                **item,
                "rowid": int(item["rowid"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
                "duration_ns": int(item["end"]) - int(item["start"]),
                "correlation_id": int(item["correlationId"]),
                "global_pid": _normalize_global_pid(int(item["globalPid"])),
                "name": _resolved_name(raw, string_map),
            }
        )
    return result


def _load_gpu_activity(
    db: sqlite3.Connection, table_set: set[str], table: str
) -> list[tuple[int, int]]:
    if table not in table_set:
        return []
    columns = _cols(db, table)
    if not {"start", "end"}.issubset(columns):
        return []
    return [
        (int(start), int(end))
        for start, end in db.execute(f'SELECT start,end FROM "{table}" WHERE end > start')
    ]


def _containing_request(
    api: Mapping[str, Any], request_ranges_by_tid: Mapping[int, Sequence[NvtxRange]]
) -> tuple[str | None, str | None, list[str]]:
    candidates = [
        item
        for item in request_ranges_by_tid.get(int(api["global_tid"]), ())
        if item.start <= int(api["start"]) and item.end >= int(api["end"])
    ]
    requests = sorted({item.request for item in candidates if item.request is not None})
    if len(requests) > 1:
        return None, None, requests
    if not requests:
        return None, None, []
    request = requests[0]
    tightest = min(
        (item for item in candidates if item.request == request),
        key=lambda item: (item.end - item.start, item.rowid),
    )
    return request, tightest.text, []


def _containing_selected_step(
    api: Mapping[str, Any],
    selected_steps_by_tid: Mapping[int, Sequence[NvtxRange]],
) -> tuple[NvtxRange | None, list[NvtxRange]]:
    """Return the unique selected step fully containing a CUDA API interval.

    Selection is deliberately based on the CPU launch interval, not on the GPU
    kernel interval.  A kernel may legally extend beyond the enclosing NVTX
    range because CUDA launches are asynchronous.
    """

    candidates = [
        item
        for item in selected_steps_by_tid.get(int(api["global_tid"]), ())
        if item.start <= int(api["start"]) and item.end >= int(api["end"])
    ]
    if len(candidates) == 1:
        return candidates[0], []
    return None, candidates


def _interval_summary(intervals: Sequence[tuple[int, int]]) -> dict[str, Any]:
    durations = [end - start for start, end in intervals]
    return {
        "count": len(intervals),
        "sum_ns": sum(durations),
        "union_ns": _union_ns(intervals),
        "median_ns": statistics.median(durations) if durations else None,
        "p95_ns": _percentile(durations, 0.95),
    }


def _name_summary(kernels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    total = sum(int(item["duration_ns"]) for item in kernels)
    for item in kernels:
        grouped[str(item["name"])].append(item)
    result = []
    for name, values in grouped.items():
        durations = [int(item["duration_ns"]) for item in values]
        value_sum = sum(durations)
        result.append(
            {
                "name": name,
                "calls": len(values),
                "total_ns": value_sum,
                "median_ns": statistics.median(durations),
                "p95_ns": _percentile(durations, 0.95),
                "share_of_kernel_sum": value_sum / total if total else None,
            }
        )
    return sorted(result, key=lambda item: (-int(item["total_ns"]), str(item["name"])))


def analyze_trace(
    path: str | Path,
    *,
    expected_request_count: int,
    first_step: int = DEFAULT_FIRST_STEP,
    last_step: int = DEFAULT_LAST_STEP,
    capture_name_pattern: str = r"^wan_(?:denoise_steps_20_24|capture_guard_steps_19_25)$",
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    db = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        table_set = _tables(db)
        string_map = _strings(db, table_set)
        nvtx = _load_nvtx(db, table_set, string_map)
        capture_re = re.compile(capture_name_pattern)
        captures = [item for item in nvtx if capture_re.fullmatch(item.text)]
        issues: list[str] = []
        if len(captures) != 1:
            issues.append(f"expected exactly one closed capture trigger range, found {len(captures)}")
        capture = captures[0] if len(captures) == 1 else None

        step_ranges: dict[tuple[str, int, str], list[NvtxRange]] = defaultdict(list)
        all_request_ranges: list[NvtxRange] = []
        for item in nvtx:
            if item.request is not None:
                all_request_ranges.append(item)
            match = STEP_RE.search(item.text)
            if match:
                request, step, pass_name = match.groups()
                step_ranges[(request, int(step), pass_name)].append(item)
        requests = sorted(
            {
                request
                for request, step, _pass in step_ranges
                if first_step <= step <= last_step
            }
        )
        if len(requests) != expected_request_count:
            issues.append(
                f"expected {expected_request_count} request IDs in analysed steps, found {requests}"
            )
        for request in requests:
            for step in range(first_step, last_step + 1):
                for pass_name in ("conditional", "unconditional"):
                    count = len(step_ranges.get((request, step, pass_name), ()))
                    if count != 1:
                        issues.append(
                            f"{request} step={step} pass={pass_name}: expected one closed DiT range, found {count}"
                        )

        selected_steps = [
            values[0]
            for (request, step, _pass), values in step_ranges.items()
            if request in requests and first_step <= step <= last_step and len(values) == 1
        ]
        if not selected_steps:
            issues.append("no complete selected step ranges")
            cpu_step_start = cpu_step_end = 0
        else:
            cpu_step_start = min(item.start for item in selected_steps)
            cpu_step_end = max(item.end for item in selected_steps)
            if capture and not (
                capture.start <= cpu_step_start and capture.end >= cpu_step_end
            ):
                issues.append("selected CPU step envelope is censored by capture trigger range")

        # This map contains *only* the selected 20..24 DiT ranges.  In
        # particular, request-labelled guard steps 19/25 and wider request
        # ranges are not eligible to select launches.
        selected_steps_by_tid: dict[int, list[NvtxRange]] = defaultdict(list)
        for item in selected_steps:
            selected_steps_by_tid[item.global_tid].append(item)

        request_ranges_by_tid: dict[int, list[NvtxRange]] = defaultdict(list)
        for item in all_request_ranges:
            request_ranges_by_tid[item.global_tid].append(item)

        runtime = _load_api_table(
            db,
            table_set,
            string_map,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "runtime",
        )
        driver = _load_api_table(
            db,
            table_set,
            string_map,
            "CUPTI_ACTIVITY_KIND_DRIVER",
            "driver",
        )
        if not runtime:
            issues.append("no CUDA runtime API records")
        apis = runtime + driver
        ambiguous_api_count = 0
        selected_step_api_ambiguous_count = 0
        for api in apis:
            request, range_text, conflicts = _containing_request(api, request_ranges_by_tid)
            api["request"] = request
            api["nvtx_range"] = range_text
            if conflicts:
                ambiguous_api_count += 1
                api["request_conflicts"] = conflicts
            selected_step, selected_conflicts = _containing_selected_step(
                api, selected_steps_by_tid
            )
            api["_selected_step"] = selected_step
            if selected_step is not None:
                # Step attribution is authoritative for admission/selection.
                # The tighter all-range attribution above is retained for
                # attention/MLP breakdowns.
                api["selected_step_request"] = selected_step.request
                api["selected_step_text"] = selected_step.text
            elif selected_conflicts:
                selected_step_api_ambiguous_count += 1
                api["selected_step_conflicts"] = [
                    item.text for item in selected_conflicts
                ]

        # If one process contains exactly one observed request, safely attribute
        # its between-range scheduler APIs to that request.  Never do this for B5,
        # where both requests share a PID.
        pid_requests: dict[int, set[str]] = defaultdict(set)
        for item in all_request_ranges:
            if item.request:
                pid_requests[item.pid].add(item.request)
        for api in apis:
            if api["request"] is None and len(pid_requests.get(int(api["pid"]), set())) == 1:
                api["request"] = next(iter(pid_requests[int(api["pid"])]))
                api["nvtx_range"] = "PID_UNIQUE_REQUEST_FALLBACK"

        kernels = _load_kernels(db, table_set, string_map)
        by_join: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for api in apis:
            if LAUNCH_RE.search(str(api["name"])):
                by_join[(int(api["global_pid"]), int(api["correlation_id"]))].append(api)
        join_missing = join_ambiguous = selected_join_ambiguous = 0
        for kernel in kernels:
            candidates = by_join.get(
                (int(kernel["global_pid"]), int(kernel["correlation_id"])), []
            )
            # Prefer the API closest to the kernel while preserving signed queue
            # delay.  Multiple equally-close records remain an admission issue.
            if candidates:
                best_distance = min(abs(int(kernel["start"]) - int(api["end"])) for api in candidates)
                best = [
                    api
                    for api in candidates
                    if abs(int(kernel["start"]) - int(api["end"])) == best_distance
                ]
            else:
                best = []
            if len(best) == 1:
                api = best[0]
                kernel["api_rowid"] = api["rowid"]
                kernel["api_name"] = api["name"]
                kernel["api_source"] = api["source"]
                kernel["request"] = api["request"]
                kernel["nvtx_range"] = api["nvtx_range"]
                kernel["queue_delay_signed_ns"] = int(kernel["start"]) - int(api["end"])
                selected_step = api.get("_selected_step")
                kernel["_selected_step"] = selected_step
                kernel["_selected_phase_candidate"] = selected_step is not None
                if selected_step is not None:
                    # The full enclosing step is the authoritative request
                    # attribution.  Preserve the tight nested range text for
                    # operator summaries, but never let a PID fallback select a
                    # guard/out-of-window launch.
                    kernel["request"] = selected_step.request
                    kernel["selected_step_text"] = selected_step.text
            elif not best:
                join_missing += 1
                kernel["request"] = None
                kernel["_selected_phase_candidate"] = False
            else:
                join_ambiguous += 1
                kernel["request"] = None
                # An equally-close collision cannot be silently dropped if one
                # of the candidate launches belongs to the selected phase.
                # Retain it in the denominator, leave request unattributed, and
                # fail admission through the >=90% attribution rule below.
                selected_best = [api for api in best if api.get("_selected_step") is not None]
                kernel["_selected_phase_candidate"] = bool(selected_best)
                if selected_best:
                    selected_join_ambiguous += 1

        # Select by the *launch API's* complete nesting in a selected NVTX
        # range, then keep the correlated GPU kernel's complete interval.  Do
        # not intersect/clip with the CPU NVTX envelope: asynchronous tails are
        # expected and are exactly why an interval-based CPU filter is wrong.
        selected_kernels = [
            item for item in kernels if item.get("_selected_phase_candidate")
        ]
        if not selected_kernels:
            issues.append("no kernels attributed to launches nested in selected steps")

        if selected_kernels:
            analysis_start = min(int(item["start"]) for item in selected_kernels)
            analysis_end = max(int(item["end"]) for item in selected_kernels)
        else:
            analysis_start = analysis_end = 0

        async_tail_kernels = [
            item
            for item in selected_kernels
            if item.get("_selected_step") is not None
            and int(item["end"]) > int(item["_selected_step"].end)
        ]
        outside_cpu_step_envelope = [
            item
            for item in selected_kernels
            if int(item["start"]) < cpu_step_start or int(item["end"]) > cpu_step_end
        ] if cpu_step_end > cpu_step_start else []

        kernel_intervals = [
            (int(item["start"]), int(item["end"])) for item in selected_kernels
        ]
        busy_ns = _union_ns(kernel_intervals)
        kernel_sum_ns = sum(int(item["duration_ns"]) for item in selected_kernels)
        any_concurrency_ns = _active_at_least_ns(kernel_intervals, 2)
        request_intervals = {
            request: [
                (int(item["start"]), int(item["end"]))
                for item in selected_kernels
                if item.get("request") == request
            ]
            for request in requests
        }
        request_overlap_ns: int | None = None
        if len(requests) == 2:
            request_overlap_ns = _intersection_ns(
                request_intervals[requests[0]], request_intervals[requests[1]]
            )
        elif len(requests) == 1:
            request_overlap_ns = 0
        attributed_sum = sum(
            int(item["duration_ns"])
            for item in selected_kernels
            if item.get("request") in requests
        )
        attribution_fraction = attributed_sum / kernel_sum_ns if kernel_sum_ns else None
        if attribution_fraction is None or attribution_fraction < 0.90:
            issues.append(
                f"request attribution covers only {attribution_fraction!r} of kernel-duration sum"
            )
        selected_api_request_ambiguous_count = sum(
            1
            for api in apis
            if api.get("_selected_step") is not None and api.get("request_conflicts")
        )
        if (
            selected_api_request_ambiguous_count
            or selected_step_api_ambiguous_count
            or selected_join_ambiguous
        ):
            issues.append(
                "ambiguous selected-step attribution: "
                f"API_request={selected_api_request_ambiguous_count}, "
                f"API_step={selected_step_api_ambiguous_count}, "
                f"kernel_join={selected_join_ambiguous}"
            )

        # Two notions are deliberately kept separate.  ``selected_kernels``
        # define the fixed step-20..24 comparison set, while *all* kernels
        # executing inside that GPU envelope determine whether the physical
        # device was actually idle.  A guard-step kernel can run between two
        # selected kernels when requests are skewed; counting that interval as
        # GPU idle would inflate the launch/sync Amdahl bound.
        selected_phase_gaps = (
            _gaps(kernel_intervals, analysis_start, analysis_end)
            if analysis_end > analysis_start
            else []
        )
        all_gpu_intervals_in_envelope = [
            (max(analysis_start, int(item["start"])), min(analysis_end, int(item["end"])))
            for item in kernels
            if int(item["end"]) > analysis_start and int(item["start"]) < analysis_end
        ]
        selected_kernel_rowids = {int(item["rowid"]) for item in selected_kernels}
        out_of_phase_gpu_intervals_in_envelope = [
            (max(analysis_start, int(item["start"])), min(analysis_end, int(item["end"])))
            for item in kernels
            if int(item["rowid"]) not in selected_kernel_rowids
            and int(item["end"]) > analysis_start
            and int(item["start"]) < analysis_end
        ]
        physical_gpu_gaps = (
            _gaps(all_gpu_intervals_in_envelope, analysis_start, analysis_end)
            if analysis_end > analysis_start
            else []
        )
        window_ns = analysis_end - analysis_start
        selected_phase_unoccupied_ns = sum(
            end - start for start, end in selected_phase_gaps
        )
        physical_gpu_idle_ns = sum(end - start for start, end in physical_gpu_gaps)
        all_gpu_busy_ns = _union_ns(all_gpu_intervals_in_envelope)
        out_of_phase_busy_ns = _union_ns(out_of_phase_gpu_intervals_in_envelope)
        additional_out_of_phase_busy_ns = max(0, all_gpu_busy_ns - busy_ns)
        long_gaps = sorted(
            (
                {"start_ns": start, "end_ns": end, "duration_ns": end - start}
                for start, end in physical_gpu_gaps
            ),
            key=lambda item: -int(item["duration_ns"]),
        )[:20]

        # CUDA API metrics follow the same full-nesting phase selection as the
        # kernel set.  A wall-clock/GPU-envelope filter would accidentally mix
        # launches from async guard step 25 or the other request.
        selected_apis = [api for api in apis if api.get("_selected_step") is not None]
        launch_apis = [api for api in selected_apis if LAUNCH_RE.search(str(api["name"]))]
        sync_apis = [api for api in selected_apis if SYNC_RE.search(str(api["name"]))]
        blocking_apis = [
            api for api in selected_apis if BLOCKING_MEMCPY_RE.search(str(api["name"]))
        ]
        selected_launch_step_keys: Counter[tuple[str, int, str]] = Counter()
        for api in launch_apis:
            selected_step = api.get("_selected_step")
            if selected_step is None:
                continue
            match = STEP_RE.search(selected_step.text)
            if match:
                request, step_text, pass_name = match.groups()
                selected_launch_step_keys[(request, int(step_text), pass_name)] += 1
        for request in requests:
            for step in range(first_step, last_step + 1):
                for pass_name in ("conditional", "unconditional"):
                    if selected_launch_step_keys[(request, step, pass_name)] == 0:
                        issues.append(
                            f"{request} step={step} pass={pass_name}: no fully nested CUDA launch API"
                        )
        queue_delays = [
            int(item["queue_delay_signed_ns"])
            for item in selected_kernels
            if "queue_delay_signed_ns" in item
        ]
        queue_nonnegative = [max(0, value) for value in queue_delays]

        top = _name_summary(selected_kernels)
        by_request: dict[str, dict[str, Any]] = {}
        for request in requests:
            values = [item for item in selected_kernels if item.get("request") == request]
            intervals = [(int(item["start"]), int(item["end"])) for item in values]
            if not values:
                issues.append(f"{request}: no request-attributed selected kernel")
            by_request[request] = {
                "kernel_count": len(values),
                "kernel_duration_sum_ns": sum(int(item["duration_ns"]) for item in values),
                "kernel_busy_union_ns": _union_ns(intervals),
            }

        by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in selected_kernels:
            lane = (
                int(item["global_pid"]),
                item.get("contextId"),
                item.get("streamId"),
                item.get("request"),
            )
            by_lane[str(lane)].append(item)
        lanes = {
            key: {
                "calls": len(values),
                "kernel_duration_sum_ns": sum(int(item["duration_ns"]) for item in values),
                "kernel_busy_union_ns": _union_ns(
                    (int(item["start"]), int(item["end"])) for item in values
                ),
            }
            for key, values in sorted(by_lane.items())
        }

        operator_sums: dict[str, Counter[str]] = defaultdict(Counter)
        for item in selected_kernels:
            text = str(item.get("nvtx_range") or "")
            match = OPERATOR_RE.search(text)
            if match:
                request, _group, _block, operator = match.groups()
                operator_sums[request][operator] += int(item["duration_ns"])

        hot_overlap: list[dict[str, Any]] = []
        if len(requests) == 2:
            for summary in top[:10]:
                name = str(summary["name"])
                left = [
                    (int(item["start"]), int(item["end"]))
                    for item in selected_kernels
                    if item.get("request") == requests[0] and item["name"] == name
                ]
                right = [
                    (int(item["start"]), int(item["end"]))
                    for item in selected_kernels
                    if item.get("request") == requests[1] and item["name"] == name
                ]
                overlap = _intersection_ns(left, right)
                hot_overlap.append(
                    {
                        "name": name,
                        "request0_calls": len(left),
                        "request1_calls": len(right),
                        "same_name_cross_request_overlap_ns": overlap,
                        "overlaps": overlap > 0,
                    }
                )

        memcpy_intervals = _load_gpu_activity(
            db, table_set, "CUPTI_ACTIVITY_KIND_MEMCPY"
        )
        memset_intervals = _load_gpu_activity(
            db, table_set, "CUPTI_ACTIVITY_KIND_MEMSET"
        )
        sync_activity_intervals = _load_gpu_activity(
            db, table_set, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
        )
        capture_payload = (
            {
                "name": capture.text,
                "start_ns": capture.start,
                "end_ns": capture.end,
                "duration_ns": capture.end - capture.start,
                "pid": capture.pid,
            }
            if capture
            else None
        )
        result: dict[str, Any] = {
            "sqlite": str(resolved),
            "classification_limits": (
                "Nsight Systems timeline only; no Tensor/SM/DRAM SOL, occupancy, "
                "register/shared-memory, or tail-wave claim"
            ),
            "admission": {
                "pass": not issues,
                "issues": issues,
                "expected_request_count": expected_request_count,
                "observed_requests": requests,
                "selected_steps": [first_step, last_step],
                "capture_trigger": capture_payload,
                "closed_selected_step_range_count": len(selected_steps),
                "async_tail_kernel_count": len(async_tail_kernels),
                "selected_kernel_outside_cpu_step_envelope_count": len(
                    outside_cpu_step_envelope
                ),
                "ambiguous_api_request_count_all_trace": ambiguous_api_count,
                "ambiguous_api_request_count_selected_steps": selected_api_request_ambiguous_count,
                "ambiguous_selected_step_api_count": selected_step_api_ambiguous_count,
                "kernel_api_join_missing_count_all_trace": join_missing,
                "kernel_api_join_ambiguous_count_all_trace": join_ambiguous,
                "kernel_api_join_ambiguous_count_selected_steps": selected_join_ambiguous,
                "request_attribution_kernel_sum_fraction": attribution_fraction,
                "fully_nested_selected_launch_api_count": len(launch_apis),
                "selected_step_passes_with_launch_count": sum(
                    1 for value in selected_launch_step_keys.values() if value > 0
                ),
            },
            "selected_cpu_step_window": {
                "start_ns": cpu_step_start,
                "end_ns": cpu_step_end,
                "duration_ns": cpu_step_end - cpu_step_start,
                "definition": (
                    f"envelope of closed CPU NVTX DiT ranges for steps "
                    f"{first_step}..{last_step}; used only to select fully nested APIs"
                ),
            },
            "analysis_window": {
                "start_ns": analysis_start,
                "end_ns": analysis_end,
                "duration_ns": window_ns,
                "definition": (
                    f"GPU envelope (min start/max end) of complete kernels whose launch APIs "
                    f"are fully nested in selected steps {first_step}..{last_step}"
                ),
            },
            "kernel_count": len(selected_kernels),
            "kernel_duration_sum_ns": kernel_sum_ns,
            "gpu_kernel_busy_union_ns": busy_ns,
            "selected_phase_gpu_kernel_busy_union_ns": busy_ns,
            "all_gpu_kernel_busy_union_inside_analysis_window_ns": all_gpu_busy_ns,
            "out_of_phase_gpu_busy_union_inside_analysis_window_ns": out_of_phase_busy_ns,
            "additional_out_of_phase_gpu_busy_not_covered_by_selected_phase_ns": (
                additional_out_of_phase_busy_ns
            ),
            "any_kernel_concurrency_ns": any_concurrency_ns,
            "any_kernel_concurrency_fraction": any_concurrency_ns / busy_ns if busy_ns else None,
            "request_overlap_union_ns": request_overlap_ns,
            "request_overlap_fraction": request_overlap_ns / busy_ns if busy_ns and request_overlap_ns is not None else None,
            "gpu_idle_inside_analysis_window_ns": physical_gpu_idle_ns,
            "gpu_idle_fraction_inside_analysis_window": physical_gpu_idle_ns / window_ns if window_ns else None,
            "gpu_idle_definition": (
                "physical gaps in the union of every recorded GPU kernel inside the "
                "selected-kernel GPU envelope; guard/out-of-step work is not mislabeled idle"
            ),
            "largest_gpu_idle_gaps": long_gaps,
            "selected_phase_unoccupied_inside_analysis_window_ns": selected_phase_unoccupied_ns,
            "selected_phase_unoccupied_fraction_inside_analysis_window": (
                selected_phase_unoccupied_ns / window_ns if window_ns else None
            ),
            "selected_phase_unoccupied_definition": (
                "gaps in only the fixed step-20..24 selected-kernel union; this is "
                "reported diagnostically and is not a physical GPU-idle/Amdahl bound"
            ),
            "launch_sync_idle_amdahl": {
                "max_reclaimable_gpu_idle_fraction": physical_gpu_idle_ns / window_ns if window_ns else None,
                "warning": (
                    "Physical GPU-idle fraction is an optimistic wall upper bound. "
                    "CUDA API sums may "
                    "overlap GPU work/each other and are not added to this bound."
                ),
            },
            "per_request": by_request,
            "per_context_stream_request_lane": lanes,
            "operator_kernel_duration_sum_ns_by_request": {
                request: dict(values) for request, values in sorted(operator_sums.items())
            },
            "cuda_launch_api": _interval_summary(
                [(int(item["start"]), int(item["end"])) for item in launch_apis]
            ),
            "launch_to_kernel_queue_delay_ns": {
                "count": len(queue_delays),
                "signed_median": statistics.median(queue_delays) if queue_delays else None,
                "signed_p95": _percentile(queue_delays, 0.95),
                "negative_count": sum(value < 0 for value in queue_delays),
                "nonnegative_median": statistics.median(queue_nonnegative) if queue_nonnegative else None,
                "nonnegative_p95": _percentile(queue_nonnegative, 0.95),
            },
            "synchronization_api": _interval_summary(
                [(int(item["start"]), int(item["end"])) for item in sync_apis]
            ),
            "blocking_memcpy_api": _interval_summary(
                [(int(item["start"]), int(item["end"])) for item in blocking_apis]
            ),
            "gpu_memcpy_activity": _interval_summary(
                [(max(analysis_start, s), min(analysis_end, e)) for s, e in memcpy_intervals if e > analysis_start and s < analysis_end]
            ),
            "gpu_memset_activity": _interval_summary(
                [(max(analysis_start, s), min(analysis_end, e)) for s, e in memset_intervals if e > analysis_start and s < analysis_end]
            ),
            "synchronization_activity": _interval_summary(
                [(max(analysis_start, s), min(analysis_end, e)) for s, e in sync_activity_intervals if e > analysis_start and s < analysis_end]
            ),
            "top_kernels": top[:10],
            "top10_kernel_time_coverage": (
                sum(int(item["total_ns"]) for item in top[:10]) / kernel_sum_ns
                if kernel_sum_ns
                else None
            ),
            "all_kernel_medians": {
                str(item["name"]): {
                    "median_ns": item["median_ns"],
                    "calls": item["calls"],
                    "total_ns": item["total_ns"],
                }
                for item in top
            },
            "hot_kernel_cross_request_overlap": hot_overlap,
        }
        return result
    finally:
        db.close()


def _cross_trace_analysis(traces: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "P50_SINGLE" in traces and "DUAL_P50" in traces:
        solo = traces["P50_SINGLE"].get("all_kernel_medians", {})
        dual = traces["DUAL_P50"].get("all_kernel_medians", {})
        common: list[dict[str, Any]] = []
        for name in set(solo) & set(dual):
            solo_median = float(solo[name]["median_ns"])
            dual_median = float(dual[name]["median_ns"])
            if solo_median <= 0:
                continue
            common.append(
                {
                    "name": name,
                    "single_p50_calls": int(solo[name]["calls"]),
                    "dual_p50_calls": int(dual[name]["calls"]),
                    "single_p50_median_ns": solo_median,
                    "dual_p50_median_ns": dual_median,
                    "kernel_inflation": dual_median / solo_median,
                    "dual_p50_total_ns": int(dual[name]["total_ns"]),
                }
            )
        result["kernel_inflation_p50_to_dual"] = sorted(
            common, key=lambda item: -int(item["dual_p50_total_ns"])
        )

    dual = traces.get("DUAL_P50")
    if dual and dual.get("admission", {}).get("pass"):
        overlap = dual.get("request_overlap_fraction")
        idle = dual.get("gpu_idle_fraction_inside_analysis_window")
        top10 = dual.get("top10_kernel_time_coverage")
        inflation_rows = result.get("kernel_inflation_p50_to_dual", [])[:10]
        weighted_den = sum(int(row["dual_p50_total_ns"]) for row in inflation_rows)
        weighted_inflation = (
            sum(
                float(row["kernel_inflation"]) * int(row["dual_p50_total_ns"])
                for row in inflation_rows
            )
            / weighted_den
            if weighted_den
            else None
        )
        rules = {
            "NO_REAL_OVERLAP": "request overlap <1%",
            "OVERLAP_WITH_KERNEL_INFLATION": (
                "request overlap >=1%, top10 coverage >=80%, weighted top10 inflation >=1.10"
            ),
            "GPU_IDLE_OR_HOST_GAP": "GPU idle fraction >=5%",
            "MIXED": "none of the preceding timeline-only rules dominates",
        }
        if overlap is not None and float(overlap) < 0.01:
            classification = "NO_REAL_OVERLAP"
        elif idle is not None and float(idle) >= 0.05:
            classification = "GPU_IDLE_OR_HOST_GAP"
        elif (
            overlap is not None
            and float(overlap) >= 0.01
            and top10 is not None
            and float(top10) >= 0.80
            and weighted_inflation is not None
            and weighted_inflation >= 1.10
        ):
            classification = "OVERLAP_WITH_KERNEL_INFLATION"
        else:
            classification = "MIXED"
        result["timeline_classification"] = {
            "value": classification,
            "rules": rules,
            "weighted_top10_kernel_inflation": weighted_inflation,
            "limits": "timeline-only diagnostic, not a hardware-resource-bound claim",
        }
    elif dual:
        result["timeline_classification"] = {
            "value": "INVALID",
            "reason": "DUAL_P50 trace failed admission",
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", nargs="+")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument(
        "--expected-request-counts",
        nargs="*",
        type=int,
        help="One count per SQLite. Defaults to 1 for labels containing SINGLE, else 2.",
    )
    parser.add_argument("--first-step", type=int, default=DEFAULT_FIRST_STEP)
    parser.add_argument("--last-step", type=int, default=DEFAULT_LAST_STEP)
    parser.add_argument("--capture-name-pattern", default=r"^wan_(?:denoise_steps_20_24|capture_guard_steps_19_25)$")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = args.labels or [Path(item).stem for item in args.sqlite]
    if len(labels) != len(args.sqlite):
        parser.error("--labels count must match SQLite inputs")
    counts = args.expected_request_counts
    if counts is None or not counts:
        counts = [1 if "SINGLE" in label.upper() else 2 for label in labels]
    if len(counts) != len(args.sqlite) or any(value not in (1, 2) for value in counts):
        parser.error("--expected-request-counts must contain one 1/2 value per SQLite")
    traces: dict[str, Any] = {}
    for label, path, count in zip(labels, args.sqlite, counts):
        traces[label] = analyze_trace(
            path,
            expected_request_count=count,
            first_step=args.first_step,
            last_step=args.last_step,
            capture_name_pattern=args.capture_name_pattern,
        )
    result = {
        "schema_version": 3,
        "analysis_type": "E016_NSYS_TIMELINE_ONLY",
        "selection_semantics": (
            "full CUDA launch API nested in selected CPU NVTX step; process-scoped "
            "launch correlation selects complete asynchronous GPU kernel interval"
        ),
        "traces": traces,
        "cross_trace": _cross_trace_analysis(traces),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if all(value["admission"]["pass"] for value in traces.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
