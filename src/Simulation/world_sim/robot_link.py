"""Connected robots: a planner worker kept running on a robot, so Runs there skip the sync and the warm-up.

connect(host) syncs the code like robot.py, then starts plan_server.py in the robot's billie container over
one ssh connection that stays open. The worker builds the planner once (the warm-up a cold robot run
pays every time) and then plans each scenario sent to it. runs.py uses a ready connection instead of
robot.py's sync / run / fetch. The worker ends with the connection: disconnect(), the editor exiting, a
dropped ssh, or its idle timeout (plan_server.IDLE_TIMEOUT_S).
"""

import itertools
import json
import subprocess
import threading
import time
import traceback
from typing import Callable

from Simulation.world_sim import robot

# Worker log lines kept per connection (the page shows the tail).
_LOG_LINES = 300


class RobotConnection:
    """One robot's planner worker and its state."""

    def __init__(self, host: str, robot_checkout: str):
        """host: ssh destination. robot_checkout: billie-onboard checkout on the robot for the planner code ("" = local)."""
        self.host = host
        self.robot_checkout = robot_checkout
        self.state = "connecting"  # connecting -> syncing -> building -> ready; or error / disconnected
        self.detail = ""  # the error, or what the worker said when ready
        self.since = time.time()  # when the state last changed
        self.log: list[str] = []
        self.process: subprocess.Popen | None = None
        self._ids = itertools.count(1)
        self._job_lock = threading.Lock()  # one job at a time
        self._pending: dict[str, dict] = {}  # job id -> {"event": threading.Event, "reply": dict | None}
        self._job_log: Callable[[str], None] | None = None  # the running job's log
        threading.Thread(target=self._start, daemon=True).start()

    def _set(self, state: str, detail: str = "") -> None:
        """Changes the state. state: The new state. detail: Its note (error text, ready info)."""
        self.state, self.detail, self.since = state, detail, time.time()
        self._line(f"--- {state}{': ' + detail if detail else ''}")

    def _line(self, line: str) -> None:
        """Keeps one log line and passes it to the running job. line: The text."""
        self.log = (self.log + [line])[-_LOG_LINES:]
        if self._job_log is not None:
            self._job_log(line)

    def _start(self) -> None:
        """Syncs the code, starts the worker and follows its output until it ends."""
        try:
            self._set("syncing")
            robot.sync(self.host, [], self.robot_checkout, log=self._line)
            self._set("building", "the planner warms up once (about 1 minute, several after a geometry change)")
            self.process = subprocess.Popen(robot.worker_command(self.host), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)  # fmt: skip
            for line in self.process.stdout:
                self._on_output(line.rstrip("\n"))
            code = self.process.wait()
            if self.state != "disconnected":
                self._set("error", f"the worker ended (exit code {code}); see its log")
        except Exception as e:  # shown on the page; the robot can still run the cold way
            self._line(traceback.format_exc())
            self._set("error", f"{type(e).__name__}: {e}")
        for job in self._pending.values():  # nothing more will answer them
            job["event"].set()

    def _on_output(self, line: str) -> None:
        """Handles one output line of the worker: a protocol event, or a log line.

        line: The line, without its newline.
        """
        if not line.startswith(robot.WORKER_PREFIX):
            self._line(line)
            return
        message = json.loads(line[len(robot.WORKER_PREFIX):])
        if message["event"] == "ready":
            self._set("ready", f"planner built in {message['build_s']}s on {message['host']} {', '.join(message['devices'])}")
        elif message.get("id") in self._pending:
            self._pending[message["id"]]["reply"] = message
            self._pending[message["id"]]["event"].set()

    def plan(self, scenario: dict, flags: dict, log: Callable[[str], None]) -> dict:
        """Plans a scenario on the worker (its recordings and room map are synced first).

        scenario: The scenario JSON. flags: plan.plan_scenario's flags. log: Receives the worker's lines meanwhile.
        Returns: The plan results, like plan.py's.
        Raises: RuntimeError if the connection is not ready, ends, or the worker reports an error.
        """
        with self._job_lock:
            if self.state != "ready":
                raise RuntimeError(f"{self.host} is not connected ({self.state})")
            robot.sync_scenario_data(self.host, scenario, log)
            job_id = str(next(self._ids))
            self._pending[job_id] = {"event": threading.Event(), "reply": None}
            self._job_log = log
            try:
                self.process.stdin.write(json.dumps({"id": job_id, "scenario": scenario, "flags": flags}) + "\n")
                self.process.stdin.flush()
                self._pending[job_id]["event"].wait()
            finally:
                self._job_log = None
            reply = self._pending.pop(job_id)["reply"]
        if reply is None:
            raise RuntimeError(f"The connection to {self.host} ended during the run ({self.detail})")
        if reply["event"] == "error":
            raise RuntimeError(f"The worker on {self.host} failed: {reply['error']}")
        return reply["results"]

    def disconnect(self) -> None:
        """Stops the worker (it ends when its input closes) and the ssh connection."""
        self._set("disconnected")
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()

    def status(self) -> dict:
        """The connection's state for the page.

        Returns: {"host", "state", "detail", "seconds" (in this state), "busy" (planning a job), "log" (tail)}.
        """
        return {"host": self.host, "state": self.state, "detail": self.detail, "seconds": round(time.time() - self.since),
                "busy": self._job_lock.locked(), "log": self.log[-40:]}  # fmt: skip


_connections: dict[str, RobotConnection] = {}  # host -> its connection
_lock = threading.Lock()


def connect(host: str, robot_checkout: str) -> RobotConnection:
    """Connects to a robot (again, if it was connected before).

    host: ssh destination. robot_checkout: See RobotConnection.
    Returns: The new connection (it syncs and builds in the background).
    """
    with _lock:
        old = _connections.get(host)
        if old is not None:
            old.disconnect()
        _connections[host] = RobotConnection(host, robot_checkout)
        return _connections[host]


def disconnect(host: str) -> None:
    """Disconnects a robot; nothing happens if it is not connected. host: ssh destination."""
    with _lock:
        connection = _connections.pop(host, None)
    if connection is not None:
        connection.disconnect()


def ready(host: str) -> RobotConnection | None:
    """The robot's connection if it can take a job now (busy ones queue). host: ssh destination. Returns: It, or None."""
    connection = _connections.get(host)
    return connection if connection is not None and connection.state == "ready" else None


def statuses() -> list[dict]:
    """Every connection's status. Returns: RobotConnection.status() per robot."""
    return [c.status() for c in list(_connections.values())]


def disconnect_all() -> None:
    """Disconnects every robot (the editor is exiting)."""
    for host in list(_connections):
        disconnect(host)
