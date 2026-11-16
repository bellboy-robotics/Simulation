"""Compares two runs of the same flow action by action: the reference (no mapping) against the mapped run.

    python -m Simulation.flow_fix.compare_runs \
        --a-brain ref/log_brain.txt --a-planner ref/log_pyroki_planner.txt --a-start 10:32:52 --a-end 10:43:07 \
        --b-brain map/log_brain.txt --b-planner map/log_pyroki_planner.txt --b-start 09:58:40 --b-end 10:18:39

Actions are matched by area, item, command and occurrence (the 2nd "joints" of an item matches the 2nd). Prints the
matched actions that differ by more than --min-diff seconds, the totals per command, and what ran in only one run
(manual recoveries, retries, the scan).
"""

import argparse
from collections import Counter, defaultdict

from .room_timing import Action, add_planner_times, parse_actions

# A matched action's key: (area, item, command, occurrence).
Key = tuple[str, str, str, int]


def keyed(actions: list[Action]) -> dict[Key, Action]:
    """The actions by their match key. actions: One run's actions. Returns: key -> action (first of each key)."""
    seen: Counter = Counter()
    out: dict[Key, Action] = {}
    for a in actions:
        base = (a.area[:30], a.item[:24], a.name)
        seen[base] += 1
        out.setdefault((*base, seen[base]), a)
    return out


def load(brain: str, planner: str | None, start: str, end: str) -> dict[Key, Action]:
    """One run's actions with planner times, keyed. brain, planner: Log paths. start, end: HH:MM:SS. Returns: Keyed."""
    actions = parse_actions(brain, start, end)
    add_planner_times(actions, planner)
    return keyed(actions)


def compare(a: dict[Key, Action], b: dict[Key, Action], min_diff: float) -> None:
    """Prints the differences between two runs. a: The reference. b: The run compared. min_diff: Seconds to show."""
    both = [k for k in b if k in a]
    print(f"{'area > item > command':60} {'ref':>7} {'run':>7}   planner ref / run")
    for k in both:
        x, y = a[k], b[k]
        if abs(x.seconds - y.seconds) > min_diff:
            px, py = sum(s for s, _ in x.planner.values()), sum(s for s, _ in y.planner.values())
            print(f"{' > '.join(map(str, k[:3])):60} {x.seconds:6.1f}s {y.seconds:6.1f}s   {px:5.1f}s / {py:5.1f}s")
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for k in both:
        t = totals[k[2]]
        t[0] += a[k].seconds
        t[1] += b[k].seconds
        t[2] += sum(s for s, _ in a[k].planner.values())
        t[3] += sum(s for s, _ in b[k].planner.values())
    print("\nmatched actions by command (ref / run, planner ref / run):")
    for name, (x, y, px, py) in sorted(totals.items(), key=lambda item: -(item[1][1] - item[1][0])):
        print(f"  {name:22} {x:6.0f}s / {y:6.0f}s   planner {px:5.0f}s / {py:5.0f}s")
    print(f"  all {len(both)} matched     {sum(a[k].seconds for k in both):6.0f}s / {sum(b[k].seconds for k in both):6.0f}s")
    for label, run, other in (("only in the reference", a, b), ("only in the run", b, a)):
        extra = [k for k in run if k not in other]
        names = Counter(k[2] for k in extra)
        print(f"{label}: {sum(run[k].seconds for k in extra):.0f}s {dict(names)}")


def main() -> None:
    """Parses the arguments and prints the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for side, what in (("a", "the reference run (no mapping)"), ("b", "the run compared (mapped)")):
        parser.add_argument(f"--{side}-brain", required=True, help=f"log_brain.txt of {what}")
        parser.add_argument(f"--{side}-planner", help=f"log_pyroki_planner.txt of {what}")
        parser.add_argument(f"--{side}-start", required=True, help="HH:MM:SS UTC")
        parser.add_argument(f"--{side}-end", default="23:59:59", help="HH:MM:SS UTC")
    parser.add_argument("--min-diff", type=float, default=3.0, help="seconds: show matched actions differing more")
    args = parser.parse_args()
    reference = load(args.a_brain, args.a_planner, args.a_start, args.a_end)
    run = load(args.b_brain, args.b_planner, args.b_start, args.b_end)
    compare(reference, run, args.min_diff)


if __name__ == "__main__":
    main()
