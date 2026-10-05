"""What the pyroki planner says and compiles while it builds and plans, for the timing report.

Collects, timestamped and tagged with the thread and scenario command:
- log lines: the planner's `logging` lines (resolver compile/cache lines, urdf.py gripper mesh fallback)
  and the `loguru` lines of pyroki and jaxls ("Building optimization problem ...");
- cloud messages the resolver sends through its dora node (e.g. "IK solver not ready yet - waiting
  for background compilation ...");
- JAX compile and persistent-cache events (jax.monitoring): tracing, lowering and backend compile
  time, cache hits and misses. They are summed per timed planner call / build phase (the calling
  thread's "slot") or per thread, so a cache hit shows as misses == 0.

The planner is built once and reused across scenarios (and editor runs), so the lines and events of
one run are collected in a CaptureWindow open only during that run, from the thread that opened it and
the planner's background threads (not other threads of the process, e.g. editor requests). Without
loguru or with an older JAX lacking some monitoring hooks, the matching part is simply not captured.
"""

import logging
import re
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

# JAX monitoring duration events -> compile stats key they add to (seconds).
_DURATION_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "trace_s",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower_s",
    "/jax/core/compile/backend_compile_duration": "backend_s",
    "/jax/compilation_cache/cache_retrieval_time_sec": "cache_read_s",
    "/jax/compilation_cache/compile_time_saved_sec": "compile_saved_s",
}
# JAX monitoring count events -> compile stats key they count in.
_COUNT_EVENTS = {"/jax/compilation_cache/cache_hits": "cache_hits", "/jax/compilation_cache/cache_misses": "cache_misses"}
# Compile stats of nothing compiled yet (copied for every slot, thread and window).
ZERO_STATS = {"trace_s": 0.0, "lower_s": 0.0, "backend_s": 0.0, "cache_read_s": 0.0, "compile_saved_s": 0.0,
              "cache_hits": 0, "cache_misses": 0}  # fmt: skip
# Log lines kept per window: a replay logs a few lines per batch, and the report only lists them.
MAX_LINES = 500
# Lines worth highlighting in the report: compiles, the JAX cache, waits and the gripper mesh fallback.
NOTABLE = re.compile(r"compil|cache|warm-up|not ready|waiting|fall(ing)? ?back|not found", re.IGNORECASE)
# Name prefixes of the planner's own background threads (resolver build/warm-ups, transit warm-up): their
# lines and events go in every window; other threads only in the windows they opened.
PLANNER_THREADS = ("pyroki-", "poc-")


class CaptureWindow:
    """Log lines, messages and compile stats of one capture window (a planner build or one run)."""

    def __init__(self, t0: float):
        """t0: Window start, epoch seconds; entry times are relative to it."""
        self.t0 = t0
        self.thread = threading.current_thread().name  # the thread that plans in it (see wants)
        self.log: list[dict] = []  # {"t" (epoch s), "thread", "src", "level", "message", "command"}
        self.messages: list[dict] = []  # cloud messages: {"t", "thread", "level", "message", "command"}
        self.dropped = 0  # log lines past MAX_LINES, not kept
        self.stats = dict(ZERO_STATS)  # compile stats of the window's events (its thread and planner threads)

    def result(self) -> dict:
        """The window's content for a results file.

        Returns: {"log", "messages": entries with "t" in seconds since t0, "log_dropped": lines not kept,
            "compile": compile stats (see ZERO_STATS) plus "compile_s" (trace + lower + backend)}.
        """
        rel = [[dict(e, t=round(e["t"] - self.t0, 4)) for e in entries] for entries in (self.log, self.messages)]
        return {"log": rel[0], "messages": rel[1], "log_dropped": self.dropped, "compile": with_compile_s(self.stats)}

    def wants(self, thread: str) -> bool:
        """Whether a line or event belongs here: from the window's own thread or a planner background thread,
        not e.g. an editor request thread logging while a run plans.

        thread: Name of the thread it came from. Returns: True to collect it.
        """
        return thread == self.thread or thread.startswith(PLANNER_THREADS)


def with_compile_s(stats: dict) -> dict:
    """Rounded copy of compile stats with their total compile time.

    stats: Compile stats (see ZERO_STATS).
    Returns: The same keys rounded to ms, plus "compile_s" = trace_s + lower_s + backend_s.
    """
    out = {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}
    out["compile_s"] = round(stats["trace_s"] + stats["lower_s"] + stats["backend_s"], 4)
    return out


class _RootHandler(logging.Handler):
    """Hands every root-logger record to the capture; prints warnings while logging is not configured."""

    def __init__(self, capture: "PlannerCapture"):
        """capture: Receives the records."""
        super().__init__(level=logging.INFO)
        self._capture = capture

    def emit(self, record: logging.LogRecord) -> None:
        """Captures one record. record: The log record."""
        try:
            self._capture.add_line(record.created, record.threadName, f"logging:{record.name}", record.levelname,
                                   record.getMessage())  # fmt: skip
        except Exception:  # a capture bug must never break the planner's logging
            pass
        # With only capture handlers on the root logger (no basicConfig, e.g. sweep.py), Python's own fallback
        # (lastResort) never runs: print warnings and errors like it would, once (from the first handler).
        handlers = logging.getLogger().handlers
        unconfigured = all(isinstance(h, _RootHandler) for h in handlers) and handlers[:1] == [self]
        if unconfigured and logging.lastResort is not None and record.levelno >= logging.lastResort.level:
            logging.lastResort.handle(record)


class PlannerCapture:
    """The process-wide capture: JAX listeners, root log handler and loguru sink, feeding open windows."""

    def __init__(self):
        """Creates it idle; install() hooks it into JAX, logging and loguru."""
        self._lock = threading.Lock()  # guards the windows, slots and per-thread stats
        self._windows: list[CaptureWindow] = []
        self._slots: dict[int, dict] = {}  # thread ident -> compile stats of the call/phase that thread runs
        self.thread_stats: dict[str, dict] = {}  # thread name -> compile stats of its events outside slots
        self.thread_span: dict[str, list[float]] = {}  # thread name -> [first, last] epoch s of its lines/events
        self.tracking = False  # fill thread_stats / thread_span (only while the planner builds; they are unbounded)
        self.command: Callable[[], int] = lambda: -1  # the scenario command running now, for the entries
        self._handler: _RootHandler | None = None
        self._jax_hooked = False

    def install(self, command: Callable[[], int]) -> None:
        """Hooks into JAX monitoring, the root logger and loguru (once; later calls only set command).

        command: Returns the scenario command running now (-1 outside runs), stored with each entry.
        """
        self.command = command
        if self._handler is None:
            self._handler = _RootHandler(self)
            logging.getLogger().addHandler(self._handler)
            try:
                from loguru import logger  # noqa: PLC0415 (pyroki and jaxls log through loguru)

                logger.add(self._on_loguru, level="INFO", format="{message}")
            except ImportError:
                pass
        if not self._jax_hooked:
            self._jax_hooked = True
            try:
                from jax import monitoring  # noqa: PLC0415 (the resolver must pick the platform first)

                monitoring.register_event_listener(self._on_event)
                monitoring.register_event_duration_secs_listener(self._on_duration)
            except (ImportError, AttributeError):
                pass  # an older JAX without these hooks: no compile/cache stats

    @contextmanager
    def import_lines(self) -> Iterator[CaptureWindow]:
        """Captures the log lines of an import (urdf.py warns about the gripper mesh at import time).

        The handler is removed again afterwards, so a later logging.basicConfig still configures logging.
        Yields: The window collecting the lines.
        """
        handler = _RootHandler(self)
        logging.getLogger().addHandler(handler)
        window = self.open_window(time.time())
        try:
            yield window
        finally:
            logging.getLogger().removeHandler(handler)
            self.close_window(window)

    def open_window(self, t0: float) -> CaptureWindow:
        """Starts collecting into a new window. t0: Its start, epoch seconds. Returns: The window."""
        window = CaptureWindow(t0)
        with self._lock:
            self._windows.append(window)
        return window

    def close_window(self, window: CaptureWindow) -> None:
        """Stops collecting into a window. window: From open_window."""
        with self._lock:
            if window in self._windows:
                self._windows.remove(window)

    @contextmanager
    def slot(self) -> Iterator[dict]:
        """Sums the calling thread's compile events into fresh stats while the block runs (nestable).

        Yields: The stats (see ZERO_STATS), filled in as the block compiles.
        """
        ident, stats = threading.get_ident(), dict(ZERO_STATS)
        with self._lock:
            outer = self._slots.get(ident)
            self._slots[ident] = stats
        try:
            yield stats
        finally:
            with self._lock:
                if outer is None:
                    self._slots.pop(ident, None)
                else:
                    self._slots[ident] = outer
                    for k, v in stats.items():
                        outer[k] += v

    def add_line(self, t: float, thread: str, src: str, level: str, message: str) -> None:
        """Adds a log line to the open windows that want its thread (see CaptureWindow.wants).

        t: Epoch seconds. thread: Thread name. src: "logging:<logger>" or "loguru:<module>".
        level: Level name. message: The text.
        """
        entry = {"t": t, "thread": thread, "src": src, "level": level, "message": message, "command": self.command()}
        with self._lock:
            if self.tracking:
                self._touch(thread, t)
            for window in (w for w in self._windows if w.wants(thread)):
                if len(window.log) < MAX_LINES:
                    window.log.append(entry)
                else:
                    window.dropped += 1

    def add_message(self, level: str, message: str) -> None:
        """Adds a cloud message the resolver sent (to the operator's app on the robot) to the open windows
        that want the calling thread.

        level: Its level ("INFO", ...). message: Its text.
        """
        entry = {"t": time.time(), "thread": threading.current_thread().name, "level": level, "message": message,
                 "command": self.command()}  # fmt: skip
        with self._lock:
            for window in (w for w in self._windows if w.wants(entry["thread"])):
                window.messages.append(entry)

    def _touch(self, thread: str, t: float) -> None:
        """Extends a thread's activity span (call with _lock held). thread: Its name. t: Epoch seconds."""
        span = self.thread_span.setdefault(thread, [t, t])
        span[0], span[1] = min(span[0], t), max(span[1], t)

    def _add(self, key: str, value: float) -> None:
        """Adds one JAX event to the open windows that want the thread and to its slot (or, while tracking,
        to its per-thread stats).

        key: Compile stats key (see ZERO_STATS). value: Seconds, or 1 for a count.
        """
        thread = threading.current_thread()
        with self._lock:
            targets = [w.stats for w in self._windows if w.wants(thread.name)]
            slot = self._slots.get(thread.ident)
            if slot is None and self.tracking:
                slot = self.thread_stats.setdefault(thread.name, dict(ZERO_STATS))
                self._touch(thread.name, time.time())
            if slot is not None:
                targets.append(slot)
            for stats in targets:
                stats[key] += value

    def _on_event(self, event: str, **kwargs) -> None:
        """jax.monitoring event listener. event: Event name. kwargs: Event tags (unused)."""
        key = _COUNT_EVENTS.get(event)
        if key is not None:
            self._add(key, 1)

    def _on_duration(self, event: str, duration: float, **kwargs) -> None:
        """jax.monitoring duration listener. event: Event name. duration: Seconds. kwargs: Tags (unused)."""
        key = _DURATION_EVENTS.get(event)
        if key is not None:
            self._add(key, float(duration))

    def _on_loguru(self, message) -> None:
        """loguru sink. message: The formatted message; its .record holds time, thread, level and module."""
        record = message.record
        self.add_line(record["time"].timestamp(), record["thread"].name, f"loguru:{record['name']}",
                      record["level"].name, record["message"])  # fmt: skip


capture = PlannerCapture()  # the one capture of this process (JAX listeners and log hooks are global)
