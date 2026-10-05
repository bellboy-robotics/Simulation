# world_sim — brain arm commands against virtual objects

Runs `joints`, `pose` and `replay_policy` (`poses` / `joints` modes) through the real pyroki planner
(`billie-onboard`, branch `arm_awareness`, world-collision POC) with objects placed around the arm,
then shows what the arm did.

## Run

```bash
# 1. plan: headless, needs only the planner (first run compiles ~30s on a Mac CPU, then the JAX cache)
python -m Simulation.world_sim.plan src/Simulation/world_sim/scenarios/post_across_path.json --keep-going
# 2. view: HTML report + PyBullet playback (no JAX needed)
python -m Simulation.world_sim.view output/world_sim/post_across_path.json
```

`plan` flags: `--keep-going` runs the remaining commands after one fails (a brain script would stop),
`--continue-on-block` plays moves the arm-move guard refuses, marked with an X, to see the whole path.

## Editor

```bash
python -m Simulation.world_sim.editor [post_across_path.json] [--robot bellboy@billie-29.bellboy ...]
```

A browser page to build scenarios and run them:
- **Billie on the map**: its pose `[x mm, y mm, yaw°]`. Type the values, or click Billie (or "select")
  and drag it in the view.
- **Objects**: add a table, wall, post or ball in front of the arm, then drag it (W = move,
  E = rotate) or type its fields. The view draws the capsules the planner gets. The header shows the
  count against the 32 slots. For a table or wall, "spacing" sets the distance between neighbouring
  capsules (empty = the robot's default). Fewer capsules save slots; the form shows the holes that
  leaves. "🎲 Random object" adds a table, wall, post or ball within the arm's reach, never touching
  Billie at the start joints.
- **Arm**: Billie is drawn from the planner's merged URDF (`/tmp/urdf/<XARM_SN>-with-tcp.urdf`: base,
  xArm, gripper and TCP) with its meshes, and its joints are set by the URDF itself. A link turns
  orange within the planner's 2cm margin of an avoid object and magenta inside one. "Planner
  capsules" (C) overlays the link capsules the planner checks. The sliders edit whatever is
  selected: the start joints, a `joints` command, or a free preview.
- **Commands**: add `joints` (the slider joints), `pose` (the current TCP; drag its target in the
  view) or `replay_policy`. "Solve IK" shows the planner's solution for a pose. "🎲 Random" adds N
  `joints` or `pose` commands to valid configurations: inside the joint limits, clear of the avoid
  objects, the floor and Billie's own body. A pose is the TCP of such a configuration, so it is reachable.
  The toast shows the seed; type it in the seed field to draw the same again.
- **Run on**: tick this Mac and/or robots. Robots come from `--robot` (default billie-29) or the
  "add a robot" field. All ticked machines plan the same scenario at once. A robot runs
  `robot.py`'s sync → plan in its `billie` container → fetch, with the planner code from "robot code"
  (a checkout on the robot; empty = this Mac's billie-onboard). Each machine shows its stage and log
  line live. When they finish you get ▶ play, its report and a per-command table: planner seconds /
  robot time estimate, plus the planner build.
- **History**: every Run is kept in `output/world_sim/runs/<time>_<file>/` with the scenario as it
  ran, then per machine its results (`local.json`, `billie-29.json`, ...) and report. A run also keeps
  the note typed before Run. Click a run to see its machines and timing, play one (this loads that
  run's scenario), reopen its scenario or delete it. `runs/index.html` ("all runs") lists every
  run and opens straight from disk too.
- **Save** writes the scenario to `scenarios/`, ready for `plan` and `robot.py`.

The planner builds in the background. Editing works before it is ready, using the arm model cached in
`output/world_sim/arm_model.json`. Use `--no-planner` to edit without JAX (needs that cache).

## Scenario file

```json
{
  "name": "...",
  "start_joints_deg": [j1, j2, j3, j4, j5, j6],
  "objects": [
    {"name": "post",  "role": "avoid",     "type": "capsule", "start_mm": [x,y,z], "end_mm": [x,y,z], "radius_mm": 40},
    {"name": "ball",  "role": "avoid",     "type": "sphere",  "center_mm": [x,y,z], "radius_mm": 50},
    {"name": "table", "role": "avoid",     "type": "table",   "center_xy_mm": [x,y], "top_height_mm": 750, "size_mm": [length, width, thickness], "yaw_deg": 0},
    {"name": "wall",  "role": "avoid",     "type": "wall",    "start_xy_mm": [x,y], "end_xy_mm": [x,y], "height_mm": 850, "thickness_mm": 10},
    {"name": "drawer","role": "eef_touch", "type": "table",   "...": "..."}
  ],
  "commands": [
    {"cmd": "joints", "joints": [...6 deg]},
    {"cmd": "pose", "pose": [x_mm, y_mm, z_mm, rx, ry, rz]},
    {"cmd": "replay_policy", "repo_id": "bellboy-robotics/...", "speed": 1.0, "transform": "poses"}
  ]
}
```

- Coordinates are in the xArm base frame, in millimeters, as the arm status shows them. The pose
  rotation is a rotation vector in radians.
- An object with `"frame": "map"` is placed in the map instead: points are `[x, y, z above the floor]`
  and `yaw_deg` is from the map +X. Such objects need
  `"billie": {"pose": [x_mm, y_mm, yaw_deg], "arm_to_base": [x_mm, y_mm, yaw_deg]}`
  (`arm_to_base` = `ARM_TO_BASE_CALIBRATION`). They are converted to the arm frame with
  `map_to_arm_frame`, so moving Billie moves them relative to the arm. The editor writes this format.
  Objects without `frame` stay in the arm frame and move with Billie. `top_height_mm` / `height_mm` are above the floor; the
  floor is 440.5mm below the arm base.
- `avoid` objects go to the planner, the arm-move guard, detours and reroutes. Together they must
  fit in the planner's 32 capsule slots. A table uses about `width / thickness` capsules.
- A table or wall may set `"spacing_mm"`: the distance between neighbouring capsule centers (radius =
  thickness / 2). Without it the robot's builders choose: a wall every 1.2 radii (overlapping), a
  table every 2 radii (touching). A larger spacing uses fewer capsules but leaves holes of
  `spacing - thickness`.
- `eef_touch` objects are only drawn (green) and measured: EEF vs rest-of-arm distance in the
  report. The planner does not know them yet; see WORLD_COLLISION_DISCUSSION.md §3.

## What is simulated

| Robot | Here |
|---|---|
| pyroki planner node (`plan`, `batch_plan`, `poc_plan_transit`, obstacles) | `planner.py`, the same `IKResolver` and transit planner in-process |
| brain `set_state_with_arm_joints` + detour, `pose`, `joints`, `replay_policy`, BufferingPlayer, reroute | `commands.py`, `reroute.py` (ports, keep in sync) |
| xarm_writer queue + arm-move guard | `sim_arm.py`, the real `arm_motion_guard_poc` |

The arm reaches each target instantly (no controller dynamics, no timing). Gripper, `thing`, brain
commands inside recordings, and the `map` / `pointcloud` / `gripper` replay transforms are not simulated.

## Timing on a robot (Jetson GPU)

```bash
python -m Simulation.world_sim.robot src/Simulation/world_sim/scenarios/*.json --keep-going
python -m Simulation.world_sim.view output/world_sim/billie-29/post_across_path.json
```

`robot.py` runs three steps:
1. **sync** copies files to `/home/bellboy/billie/world_sim`, which is mounted into the `billie` container:
   - the planner code (`pyroki_planner`, `billie_utils`, `node_ticks.py`, vendored pyroki) from the
     robot's own checkout `--robot-checkout` (default `~/users/ronit/billie-onboard`). With `""` it
     uses the local billie-onboard instead. It prints both commits.
   - this package, the scenarios and their recordings (exported to JSON, since the robot venv has no
     lerobot).
   - the arm's URDF from a local `~/releases/env/*/urdf`. It is used only on a robot with no URDF of its own (no physical arm, e.g. billie-29). Every xArm6 URDF has the same links and joints, so timings are unaffected; clearances differ by about a millimeter.
2. **run** starts `robot_run_plan.sh` inside the container. It uses the live `pyroki-planner`
   process's environment (arm, flags, GPU), the planner venv, the synced code ahead of `/app`, and its
   own JAX cache. The robot's `/app`, planner process and cache are not touched.
3. **fetch** copies the results to `output/world_sim/<robot>/` and writes their reports.

The report's "Planner timing" table shows, for each command:
- single IK, detour, transit and replay batch times;
- a robot time estimate. It feeds the replay batch times into the 50Hz BufferingPlayer, so a stall
  shows as wall time above the playback time.

The first run on a robot includes the JAX compile. Later runs load it from the cache.
