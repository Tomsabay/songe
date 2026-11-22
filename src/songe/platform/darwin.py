import ctypes
import os
import re
import subprocess
from pathlib import Path

from .. import NAME
from . import Accel, Memory, Power
from .common import ChildHandle, read


def pressure(level):
    """The kernel exports bit values 1 normal, 2 warning, 4 critical."""
    return {1: "normal", 2: "warn", 4: "critical"}.get(level, "warn")


_RUSAGE_INFO_V2 = 2
_PHYS_FOOTPRINT_OFFSET = 72  # uuid[16] + 7 uint64 fields in rusage_info_v2


def phys_footprint(pid):
    """Activity Monitor's "Memory" for a process, or None if unreadable."""
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        buffer = ctypes.create_string_buffer(512)
        if libproc.proc_pid_rusage(int(pid), _RUSAGE_INFO_V2, buffer) != 0:
            return None
        return int.from_bytes(
            buffer.raw[_PHYS_FOOTPRINT_OFFSET : _PHYS_FOOTPRINT_OFFSET + 8], "little"
        )
    except (OSError, AttributeError, ValueError):
        return None


def process_info_value(selector, result=ctypes.c_long):
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    send_obj = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    send_value = ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    instance = send_obj(
        objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
    )
    method = objc.sel_registerName(selector)
    responds = ctypes.CFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )(("objc_msgSend", objc))
    if not responds(instance, objc.sel_registerName(b"respondsToSelector:"), method):
        return None
    return send_value(instance, method)


def thermal_state():
    return process_info_value(b"thermalState")


class Darwin:
    name = "darwin"

    def __init__(self, home=None):
        self.home = Path(home) if home is not None else Path.home()

    def data_dir(self):
        return self.home / f"Library/Application Support/{NAME}"

    def swap_volume(self):
        """Where macOS adds swap files as memory runs short: the startup
        disk's container (its VM volume shares the free space)."""
        return Path("/private/var/vm"), "the startup disk"

    def idle_seconds(self):
        try:
            idle = re.search(
                r'"HIDIdleTime"\s*=\s*(\d+)',
                read("/usr/sbin/ioreg", "-c", "IOHIDSystem"),
            )
            return int(idle[1]) / 1e9 if idle else None
        except (OSError, subprocess.SubprocessError):
            return None

    def on_ac_power(self):
        return self.power().ac

    def power(self):
        ac, low, seconds = None, False, None
        try:
            iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
            iokit.IOPSGetTimeRemainingEstimate.argtypes = []
            iokit.IOPSGetTimeRemainingEstimate.restype = ctypes.c_double
            remaining = iokit.IOPSGetTimeRemainingEstimate()
            # -2 is unlimited (AC/no battery); -1 is battery, still estimating.
            if remaining == -2:
                ac = True
            elif remaining == -1 or remaining >= 0:
                ac = False
                seconds = remaining if remaining >= 0 else None
            iokit.IOPSGetBatteryWarningLevel.argtypes = []
            iokit.IOPSGetBatteryWarningLevel.restype = ctypes.c_uint
            low = iokit.IOPSGetBatteryWarningLevel() in (2, 3)  # early/final
        except (OSError, AttributeError):
            pass
        try:
            low = (
                bool(process_info_value(b"isLowPowerModeEnabled", ctypes.c_bool)) or low
            )
        except (OSError, AttributeError):
            pass
        return Power(ac, low, seconds)

    def thermal(self):
        try:
            return thermal_state()
        except (OSError, AttributeError):
            return None

    def memory(self):
        total = int(read("/usr/sbin/sysctl", "-n", "hw.memsize"))
        vm = read("/usr/bin/vm_stat")
        page = int(re.search(r"page size of (\d+) bytes", vm)[1])
        counts = {k: int(v) for k, v in re.findall(r"(Pages [^:]+):\s*(\d+)", vm)}
        free = (
            sum(
                counts.get(k, 0)
                for k in ("Pages free", "Pages inactive", "Pages speculative")
            )
            * page
        )
        level = int(
            read("/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level")
        )
        swap_used = None
        try:
            usage = read("/usr/sbin/sysctl", "-n", "vm.swapusage")
            used = re.search(r"used\s*=\s*([\d.]+)([KMGT]?)", usage)
            if used:
                swap_used = int(
                    float(used[1]) * 1024 ** (" KMGT".index(used[2]) if used[2] else 0)
                )
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            pass
        swapouts = re.search(r"^Swapouts:\s*(\d+)", vm, re.M)
        return Memory(
            total,
            free,
            pressure(level),
            swap_used=swap_used,
            compressed=(
                counts["Pages occupied by compressor"] * page
                if "Pages occupied by compressor" in counts
                else None
            ),
            swap_out=int(swapouts[1]) * page if swapouts else None,
        )

    def process_bytes(self, pid):
        """Physical footprint: unlike RSS it includes Metal (MLX) buffers."""
        footprint = phys_footprint(pid)
        if footprint is not None:
            return footprint
        try:
            return int(read("/bin/ps", "-o", "rss=", "-p", str(pid)).strip()) * 1024
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def accelerators(self):
        import platform

        if platform.machine() == "arm64":
            try:
                chip = read(
                    "/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"
                ).strip()
            except (OSError, subprocess.SubprocessError):
                chip = "Apple Silicon"
            try:
                memory = self.memory()
            except (OSError, ValueError, subprocess.SubprocessError):
                memory = Memory(0, 0, "warn")
            return [
                Accel(
                    "metal",
                    chip or "Apple Silicon",
                    memory.total,
                    memory.available,
                )
            ]
        return [Accel("cpu", platform.processor() or "CPU")]

    def keep_awake(self, pid):
        return ChildHandle(
            subprocess.Popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    def service(self):
        from .services import Launchd

        return Launchd(
            self.home / f"Library/LaunchAgents/ai.souslabs.{NAME}.plist",
            uid=os.getuid(),
        )
