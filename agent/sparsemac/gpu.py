from __future__ import annotations

import csv
import io
import subprocess


def detect_idle_gpus(max_count: int, memory_limit_mib: int = 2048, util_limit: int = 10) -> list[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        return [0]
    if proc.returncode != 0:
        return [0]
    gpus: list[int] = []
    for row in csv.reader(io.StringIO(proc.stdout)):
        if len(row) < 3:
            continue
        idx = int(row[0].strip())
        mem = int(row[1].strip())
        util = int(row[2].strip())
        if mem <= memory_limit_mib and util <= util_limit:
            gpus.append(idx)
    return gpus[:max_count] or [0]

