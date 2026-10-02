"""Deterministic checks. Each check module appends its `check` to REGISTRY."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from callcheck.model import Finding, FlowSpec, Thread


@dataclass
class Context:
    spec: FlowSpec
    thread: Thread
    alignment: Any = None
    judgment: Any = None


REGISTRY: list[Callable[[Context], list[Finding]]] = []
