"""Portable hardware and per-user service interfaces; imports never probe hardware."""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Memory:
    total: int
    available: int
    pressure: str = "normal"
    # Missing sensors stay unknown, rather than looking like zero usage. Keep
    # these keyword-only so Snapshot's positional RAM/VRAM contract is stable.
    swap_used: int | None = field(default=None, kw_only=True)
    compressed: int | None = field(default=None, kw_only=True)
    swap_out: int | None = field(default=None, kw_only=True)
    psi: float | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class Accel:
    kind: str
    name: str
    memory_total: int = 0
    memory_free: int = 0
    identifier: str = ""


LOW_BATTERY_FALLBACK = 0.10  # Where macOS and Windows warn by default.


@dataclass(frozen=True)
class Power:
    ac: bool | None = None
    low: bool = False
    seconds: float | None = None
    # Charge fraction, set only when the OS gives a percentage but no
    # low-battery level of its own; then the usual OS warning point applies.
    fraction: float | None = None

    def constrained(self, expected_seconds=0):
        return self.ac is False and (
            self.low
            or (self.fraction is not None and self.fraction <= LOW_BATTERY_FALLBACK)
            or (self.seconds is not None and 0 <= self.seconds < expected_seconds)
        )


def power_status(platform):
    """Absent power telemetry must not prevent work on desktops/servers."""
    try:
        return platform.power() or Power()
    except (AttributeError, OSError, ValueError, TypeError, subprocess.SubprocessError):
        return Power()


class Handle(Protocol):
    def close(self) -> None: ...


class Service(Protocol):
    def install(self, argv: list[str], env: dict[str, str]): ...
    def write_definition(self, argv: list[str], env: dict[str, str]): ...
    def restart(self, expected: dict[str, str]): ...
    def uninstall(self): ...
    def status(self): ...


class Platform(Protocol):
    name: str

    def data_dir(self) -> Path: ...
    def idle_seconds(self) -> float | None: ...
    def on_ac_power(self) -> bool | None: ...
    def power(self) -> Power: ...
    def thermal(self) -> int | None: ...
    def memory(self) -> Memory: ...
    def accelerators(self) -> list[Accel]: ...
    def keep_awake(self, pid: int) -> Handle | None: ...
    def service(self) -> Service: ...
    # (path, plain name) of the disk swap grows on, or None when it cannot grow.
    def swap_volume(self) -> tuple[Path, str] | None: ...


def current() -> Platform:
    if sys.platform == "darwin":
        from .darwin import Darwin

        return Darwin()
    if sys.platform == "win32":
        from .windows import Windows

        return Windows()
    from .linux import Linux

    return Linux()


def data_dir(root=None):
    selected = (
        root
        if root is not None
        else (
            Path(os.environ["SONGE_DATA_DIR"])
            if "SONGE_DATA_DIR" in os.environ
            else current().data_dir()
        )
    )
    return Path(selected).expanduser().resolve()


class Rooted:
    """Use an explicit Songe data root while delegating live OS measurements."""

    def __init__(self, platform, root):
        self.platform, self.root = platform, Path(root)
        self.name = getattr(platform, "name", sys.platform)

    def data_dir(self):
        return self.root

    def __getattr__(self, name):
        return getattr(self.platform, name)
