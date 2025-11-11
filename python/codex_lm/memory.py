"""Utilities for analyzing CUDA memory usage of model components."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List

import torch


@dataclass
class MemoryRecord:
    """Represents the memory usage of a profiled section."""

    name: str
    peak_bytes: int = 0
    retained_bytes: int = 0
    children: List["MemoryRecord"] = field(default_factory=list)

    def add_child(self, child: "MemoryRecord") -> None:
        self.children.append(child)


class MemoryAnalyzer:
    """Hierarchical CUDA memory profiler using manual instrumentation."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("MemoryAnalyzer requires a CUDA device for accurate metrics")
        if not torch.cuda.is_available():  # pragma: no cover - depends on environment
            raise RuntimeError("CUDA is not available; cannot collect memory statistics")
        self._stack: list[MemoryRecord] = []
        self._records: list[MemoryRecord] = []

    @property
    def records(self) -> List[MemoryRecord]:
        return self._records

    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        """Context manager that records memory deltas for a code region."""

        torch.cuda.synchronize(self.device)
        start_alloc = torch.cuda.memory_allocated(self.device)
        start_peak = torch.cuda.max_memory_allocated(self.device)
        record = MemoryRecord(name=name)
        if self._stack:
            self._stack[-1].add_child(record)
        else:
            self._records.append(record)
        self._stack.append(record)
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device)
            end_alloc = torch.cuda.memory_allocated(self.device)
            end_peak = torch.cuda.max_memory_allocated(self.device)
            record.peak_bytes = max(0, int(end_peak - start_peak))
            record.retained_bytes = max(0, int(end_alloc - start_alloc))
            self._stack.pop()

    def format_report(self) -> str:
        """Return a formatted string summarizing recorded sections."""

        lines = ["Section".ljust(48) + "Peak (MiB)".rjust(12) + "Retained (MiB)".rjust(16)]

        def _format(record: MemoryRecord, indent: int) -> None:
            indent_str = "  " * indent
            peak_mib = record.peak_bytes / (1024 ** 2)
            retained_mib = record.retained_bytes / (1024 ** 2)
            lines.append(
                f"{indent_str}{record.name}".ljust(48)
                + f"{peak_mib:>11.2f}"
                + f"{retained_mib:>15.2f}"
            )
            for child in record.children:
                _format(child, indent + 1)

        for rec in self._records:
            _format(rec, 0)
        return "\n".join(lines)


__all__ = ["MemoryAnalyzer", "MemoryRecord"]
