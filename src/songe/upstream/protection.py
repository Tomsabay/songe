"""Admission and lifetime protection for model instances requested by Songe.

External runtimes are never signalled. Only the exact instances this object
requested may be unloaded; a reused user instance never becomes ours.
"""

import asyncio
import contextlib
from functools import wraps
from pathlib import Path

from ..memory import MemoryPressure, escalated, paging_grew, snapshot, stopped_sample
from ..memory import admit as admit_memory
from ..platform import current


def weight_bytes(path):
    path = Path(path)
    return (
        sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        if path.is_dir()
        else path.stat().st_size
        if path.is_file()
        else 0
    )


LOAD_BYTES_PER_SECOND = 50e6  # a slow external disk


def load_seconds(config, path=None):
    """How long a model may take to load: the configured wait, or longer for
    a model too large to read from a slow disk in that time."""
    try:
        size = weight_bytes(path or config.model)
    except OSError:
        size = 0
    return max(config.upstream_timeout, size / LOAD_BYTES_PER_SECOND)


def admit(platform, required=0, device_required=None):
    return admit_memory(platform, required, device_required)


async def admit_load(upstream, version):
    # A load the user's own request needs is never refused for memory: the
    # server would load it without Songe, so Songe must not answer with 502.
    if version.get("serving"):
        return
    await asyncio.to_thread(
        admit, upstream.memory_platform, weight_bytes(version["artifact"])
    )


async def unload_within_grace(upstream, version):
    # Not asyncio.wait_for: on Python 3.11 it returns normally when the task
    # running it is cancelled just as the inner call finishes, so the memory
    # watcher could swallow its own cancellation and never stop.
    async with asyncio.timeout(getattr(upstream.config, "guard_grace_seconds", 3.0)):
        await upstream.unload(version)


def guarded_load(method):
    @wraps(method)
    async def protected(self, version):
        if not hasattr(self, "owned"):
            self.owned, self.memory_platform = {}, current()
            self.allocations = set()
            self.memory_alarm = None
            self.alarm_event = asyncio.Event()
            self.memory_watcher = None
        if hasattr(self, "recover"):
            await self.recover()
        # Verified resident reuse allocates no second weights and stays unowned.
        reuse = getattr(self, "reuse", getattr(self, "resident", None))
        resident = bool(reuse and await reuse(version))
        if resident and not self.owned:
            return
        if version.get("serving") and not resident:
            # Serving loads behave as the server would without Songe: no
            # admission and no memory watch that could cancel them midway.
            return await method(self, version)
        baseline = None
        # A load the user's own request needs behaves as the server would
        # without Songe; admission guards only loads Songe starts on its own.
        if not resident and not version.get("serving"):
            baseline = await asyncio.to_thread(
                admit, self.memory_platform, weight_bytes(version["artifact"])
            )
        # Preserve the earliest baseline while claims/loads are live. A second
        # request cannot normalize a deterioration caused by the first one.
        if self.memory_watcher is None or self.memory_watcher.done():
            self.memory_baseline = baseline or await asyncio.to_thread(
                snapshot, self.memory_platform
            )
        operation = None if resident else asyncio.create_task(method(self, version))
        if operation:
            self.allocations.add(operation)

            def finished(task):
                self.allocations.discard(task)
                # An alarm can return to the caller before the external runtime
                # confirms cancellation. The watcher still owns that operation.
                if not task.cancelled():
                    task.exception()

            operation.add_done_callback(finished)

        def unfinished():
            return [o for o in self.owned.values() if o.get("pending")]

        async def watch():
            # Loaded instances serve the user and are never unloaded for
            # memory; the watcher only guards loads that are still in flight.
            while (
                self.allocations
                or unfinished()
                or any(o.get("reclaim") for o in self.owned.values())
            ):
                try:
                    for owned in list(self.owned.values()):
                        if owned.get("reclaim"):
                            await unload_within_grace(self, owned)
                    sample = await asyncio.to_thread(admit, self.memory_platform)
                    if escalated(self.memory_baseline, sample) or paging_grew(
                        self.memory_baseline, sample
                    ):
                        raise MemoryPressure(
                            stopped_sample(
                                sample,
                                "pressure_escalation"
                                if escalated(self.memory_baseline, sample)
                                else "upstream_paging",
                                baseline=self.memory_baseline,
                            )
                        )
                except Exception as exc:
                    self.memory_alarm = exc
                    self.alarm_event.set()
                    for task in list(self.allocations):
                        if not task.cancelling():
                            task.cancel()
                    # Do not wait for server cancellation before releasing other
                    # instances or sampling again. Pending claims remain guarded.
                    for owned in unfinished():
                        with contextlib.suppress(Exception):
                            await unload_within_grace(self, owned)
                    if not unfinished() and not self.allocations:
                        return
                await asyncio.sleep(getattr(self.config, "guard_min_interval", 0.1))

        if self.memory_watcher is None or self.memory_watcher.done():
            self.memory_alarm = None
            self.alarm_event.clear()
            self.memory_watcher = asyncio.create_task(watch())
        alarm = asyncio.create_task(self.alarm_event.wait())
        try:
            if operation:
                await asyncio.wait(
                    (operation, alarm), return_when=asyncio.FIRST_COMPLETED
                )
                if self.memory_alarm is not None:
                    raise self.memory_alarm
                return await operation
        except BaseException:
            if operation and not operation.done() and not operation.cancelling():
                operation.cancel()
            if version["identifier"] in self.owned:
                with contextlib.suppress(Exception):
                    await self.unload(version)
            raise
        finally:
            alarm.cancel()
            await asyncio.gather(alarm, return_exceptions=True)

    return protected


def claim(upstream, version):
    upstream.owned[version["identifier"]] = dict(version)


async def live_claims(upstream):
    for identifier, version in list(getattr(upstream, "owned", {}).items()):
        try:
            valid = await upstream.resident(version)
        except RuntimeError:
            valid = False
        if not valid:
            upstream.owned.pop(identifier, None)
    return list(getattr(upstream, "owned", {}).values())


async def close_owned(upstream):
    for task in list(getattr(upstream, "allocations", ())):
        if not task.cancelling():
            task.cancel()
    for version in list(getattr(upstream, "owned", {}).values()):
        await upstream.unload(version)
    watcher = getattr(upstream, "memory_watcher", None)
    if watcher and not getattr(upstream, "owned", {}):
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
