"""Resource helpers for sandbox ``run_experiment`` OOM retries."""

from __future__ import annotations

import dataclasses
import re

import flyte

RESOURCE_FLOOR = flyte.Resources(cpu=2, memory="2Gi")
RESOURCE_CEILING = flyte.Resources(cpu=16, memory="32Gi")
_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?\s*$")


def _memory_to_mib(memory: str | None) -> int:
    if not memory:
        return 2048
    match = _MEM_RE.match(memory)
    if not match:
        return 2048
    value = float(match.group(1))
    unit = (match.group(2) or "Mi").lower()
    if unit in ("gi", "g", "gb"):
        return int(value * 1024)
    if unit in ("mi", "m", "mb"):
        return int(value)
    if unit in ("ki", "k", "kb"):
        return max(1, int(value // 1024))
    return int(value)


def _mib_to_memory(mib: int) -> str:
    if mib >= 1024 and mib % 1024 == 0:
        return f"{mib // 1024}Gi"
    return f"{mib}Mi"


def _cap_resources(resources: flyte.Resources) -> flyte.Resources:
    floor_cpu = int(RESOURCE_FLOOR.cpu or 2)
    ceil_cpu = int(RESOURCE_CEILING.cpu or 16)
    cpu = int(resources.cpu or floor_cpu)
    cpu = max(floor_cpu, min(ceil_cpu, cpu))

    floor_mib = _memory_to_mib(
        RESOURCE_FLOOR.memory if isinstance(RESOURCE_FLOOR.memory, str) else "2Gi"
    )
    ceil_mib = _memory_to_mib(
        RESOURCE_CEILING.memory if isinstance(RESOURCE_CEILING.memory, str) else "32Gi"
    )
    mem_mib = _memory_to_mib(resources.memory if isinstance(resources.memory, str) else None)
    mem_mib = max(floor_mib, min(ceil_mib, mem_mib))
    return flyte.Resources(cpu=cpu, memory=_mib_to_memory(mem_mib))


def _ensure_oom_increase(resources: flyte.Resources, previous: flyte.Resources) -> flyte.Resources:
    """If memory did not grow after OOM, bump deterministically up to the ceiling."""
    prev_mib = _memory_to_mib(previous.memory if isinstance(previous.memory, str) else None)
    new_mib = _memory_to_mib(resources.memory if isinstance(resources.memory, str) else None)
    if new_mib <= prev_mib:
        ceil_mib = _memory_to_mib(
            RESOURCE_CEILING.memory if isinstance(RESOURCE_CEILING.memory, str) else "32Gi"
        )
        new_mib = min(ceil_mib, max(prev_mib * 2, prev_mib + 2048))
        resources = dataclasses.replace(resources, memory=_mib_to_memory(new_mib))
    prev_cpu = int(previous.cpu or RESOURCE_FLOOR.cpu or 2)
    new_cpu = int(resources.cpu or prev_cpu)
    if new_cpu < prev_cpu:
        resources = dataclasses.replace(resources, cpu=prev_cpu)
    return _cap_resources(resources)


def bump_memory(resources: flyte.Resources) -> flyte.Resources:
    """Deterministic memory bump after OOM."""
    return _ensure_oom_increase(resources, resources)
