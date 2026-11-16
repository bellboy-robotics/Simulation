"""The timeline of one flow run from a robot's logs: every action's start and duration, the planner time inside it,
and the warnings, refusals and aborts it caused.

    python -m Simulation.flow_fix.room_timing --brain log_brain.txt --planner log_pyroki_planner.txt \
        --start 10:32:52 --end 10:43:07

The brain marks each area ("# Area:"), item ("## Item:") and action (a "<details>" line, then "<summary>name</summary>").
An action lasts until the next one starts (the last one until --end). The planner's own log gives the time of every
single-move plan, replay plan, look and batch IK solve.
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

# A log line's timestamp (the logs are in UTC).
_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
# Brain lines worth showing under an action: refusals, failures, aborts, reroutes, looks, retries.
_EVENT = re.compile(r"\[(WARNING|ERROR)\]|Aborting|skipped|did not finish|Re-running|refused|blocked|Rerouted|"
                    r"shortened|Looking first|Look before|Waiting up to|Playback:")  # fmt: skip
# Brain warnings that say nothing about the run (printed on every replay).
_NOISE = ("Collision avoidance is temporarily", "deprecated", "torchcodec")


@dataclass
class Action:
    """One action of the flow."""

    start: datetime  # when the brain started it
    area: str  # its area's name
    item: str  # its item's name
    name: str  # the command, e.g. "joints"
    end: datetime | None = None  # when the next action started
    events: list[str] = field(default_factory=list)  # notable brain lines while it ran
    planner: dict[str, tuple[float, int]] = field(default_factory=dict)  # kind -> (seconds, count)

    @property
    def seconds(self) -> float:
        """How long it ran. Returns: Seconds (0 before its end is known)."""
        return (self.end - self.start).total_seconds() if self.end else 0.0


def _stamp(line: str) -> datetime | None:
    """A log line's time. line: The line. Returns: Its timestamp, or None for a continuation line."""
    m = _TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


def parse_actions(brain_log: str, start: str, end: str) -> list[Action]:
    """The actions of the run between two times, with their notable brain lines.

    brain_log: Path of log_brain.txt. start, end: "HH:MM:SS" (UTC) of the run's first action and its end.
    Returns: The actions in order, each with its end set.
    """
    lines = open(brain_log, encoding="utf-8", errors="replace").read().splitlines()
    actions: list[Action] = []
    area = item = ""
    last: datetime | None = None
    for i, line in enumerate(lines):
        t = _stamp(line)
        if t is None or not start <= t.strftime("%H:%M:%S") <= end:
            continue
        last = t
        if "[INFO] # Area:" in line:
            area = line.split("# Area:")[1].strip()
        elif "[INFO] ## Item:" in line:
            item = line.split("## Item:")[1].strip()
        elif "<details>" in line and i + 1 < len(lines) and "<summary>" in lines[i + 1]:
            actions.append(Action(t, area, item, re.sub(r"</?summary>", "", lines[i + 1]).strip()))
        elif actions and _EVENT.search(line) and not any(n in line for n in _NOISE):
            actions[-1].events.append(f"{t:%H:%M:%S} " + re.sub(r".*\[(INFO|WARNING|ERROR)\]\s*", "", line)[:120])
    for action, following in zip(actions, actions[1:] + [None]):
        action.end = following.start if following else last
    return actions


def add_planner_times(actions: list[Action], planner_log: str | None) -> None:
    """Adds the planner's time inside each action: single moves, replays, looks and batch IK.

    actions: The run's actions (changed in place). planner_log: Path of log_pyroki_planner.txt, or None.
    """
    if not planner_log:
        return
    plans: list[tuple[datetime, str, float]] = []
    for line in open(planner_log, encoding="utf-8", errors="replace"):
        t = _stamp(line)
        kind = re.search(r"(Move|Replay|Look) planned in ([\d.]+)s", line)
        batch = re.search(r"Batch IK solver time: ([\d.]+)ms", line)
        if t and kind:
            plans.append((t, kind.group(1).lower(), float(kind.group(2))))
        elif t and batch:
            plans.append((t, "ik", float(batch.group(1)) / 1000))
    for action in actions:
        inside = [p for p in plans if action.start <= p[0] < (action.end or action.start)]
        for kind in ("move", "replay", "look", "ik"):
            of_kind = [s for _, k, s in inside if k == kind]
            if of_kind:
                action.planner[kind] = (sum(of_kind), len(of_kind))


def print_timeline(actions: list[Action]) -> None:
    """Prints the run grouped by area and item, then the totals. actions: The run's actions (with planner times)."""
    if not actions:
        print("no actions in that time range")
        return
    total = (actions[-1].end - actions[0].start).total_seconds() if actions[-1].end else 0.0
    print(f"run {actions[0].start:%H:%M:%S} - {actions[-1].end:%H:%M:%S} UTC: {total / 60:.1f} min, {len(actions)} actions")
    where = None
    for a in actions:
        if (a.area, a.item) != where:
            where = (a.area, a.item)
            print(f"\n## {a.area[:40]} > {a.item}")
        planned = " ".join(f"{k} {s:.1f}s/{n}" for k, (s, n) in a.planner.items())
        print(f"  {a.start:%H:%M:%S} {a.seconds:6.1f}s  {a.name:22} {planned}")
        for event in a.events[:4]:
            print(f"      · {event}")
    planner = Counter()
    by_type = Counter()
    for a in actions:
        by_type[a.name] += a.seconds
        for kind, (seconds, _) in a.planner.items():
            planner[kind] += seconds
    print("\nplanner time:", {k: round(v, 1) for k, v in planner.items()}, "total", round(sum(planner.values()), 1), "s")
    print("time by action type:", {k: round(v) for k, v in by_type.most_common()})


def main() -> None:
    """Parses the arguments and prints the run's timeline."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--brain", required=True, help="log_brain.txt")
    parser.add_argument("--planner", help="log_pyroki_planner.txt (optional: no planner times without it)")
    parser.add_argument("--start", required=True, help="HH:MM:SS UTC of the run's start ('Starting flow' line)")
    parser.add_argument("--end", default="23:59:59", help="HH:MM:SS UTC of its end ('Finished execution' line)")
    args = parser.parse_args()
    actions = parse_actions(args.brain, args.start, args.end)
    add_planner_times(actions, args.planner)
    print_timeline(actions)


if __name__ == "__main__":
    main()
