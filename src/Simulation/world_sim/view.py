"""View step: the HTML report and the PyBullet playback of a plan step's results.

    python -m Simulation.world_sim.view output/world_sim/table_block.json [--no-gui] [--speed 2]
"""

import argparse
import os

from Simulation.world_sim.analysis import load_run
from Simulation.world_sim.report import write_report


def main() -> None:
    """Command line: write (and open) the report next to the results file, then play the run in PyBullet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="results JSON written by Simulation.world_sim.plan")
    parser.add_argument("--no-gui", action="store_true", help="only write the HTML report")
    parser.add_argument("--no-browser", action="store_true", help="do not open the report")
    parser.add_argument("--speed", type=float, default=1.0, help="PyBullet playback speed")
    parser.add_argument("--loop", action="store_true", help="replay the PyBullet animation until closed")
    args = parser.parse_args()

    run = load_run(args.results)
    report_path = os.path.abspath(os.path.splitext(args.results)[0] + ".html")
    write_report(run, report_path, open_browser=not args.no_browser)
    print(f"Report: {report_path}")
    if not args.no_gui:
        from Simulation.world_sim.viewer import play

        play(run, speed=args.speed, loop=args.loop)


if __name__ == "__main__":
    main()
