from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass


def get_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def resolve_device(config_device: str = "auto") -> str:
    if config_device != "auto":
        return config_device
    torch = get_torch()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clear_gpu_memory() -> None:
    gc.collect()
    torch = get_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_memory_mb() -> float | None:
    torch = get_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**2), 2)


def reset_gpu_peak() -> None:
    torch = get_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@dataclass
class RunMetrics:
    runtime_seconds: float
    gpu_memory_mb: float | None


@contextmanager
def measured_run():
    reset_gpu_peak()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        measured_run.last = RunMetrics(round(elapsed, 4), gpu_memory_mb())


measured_run.last = RunMetrics(0.0, None)
