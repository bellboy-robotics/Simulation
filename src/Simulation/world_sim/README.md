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
`--compare-no-obstacles` also plans every command without the avoid objects. The result goes in
`results["no_obstacles"]`, and the report adds a "without obstacles" column.

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
  Billie at the start joints. "+ Door" and "+ Handle" add objects that turn about a hinge when the arm
  pushes them (see [Doors and handles](#doors-and-handles)). A new handle is mounted on the last door,
  on the arm's side. Faint copies show each one at its min and max angles. Moving a door moves its
  handle too.
- **Arm init position**: Billie is drawn from the planner's merged URDF (`/tmp/urdf/<XARM_SN>-with-tcp.urdf`:
  base, xArm, gripper and TCP) with its meshes, and its joints are set by the URDF itself.
  - A link turns orange within the planner's 2cm margin of an avoid object and magenta inside one.
    The color takes the worse of the planner's capsule and the real mesh (see
    [Real geometry check](#real-geometry-check)), and the text shows both when they disagree.
  - "Planner capsules" (C) overlays the capsules the planner checks. In playback these are the run's
    own capsules, for example the robot's.
  - The sliders edit whatever is selected: the init position ("Edit init position"), a `joints`
    command, or a free preview. "Jump to init position" shows the arm there without changing anything.
- **Commands**: add `joints` ("+ joints (sliders)"), `pose` ("+ pose (TCP)": the sliders' TCP; drag its
  target in the view) or `replay_policy`. "Solve IK" shows the planner's solution for a pose. "🎲 Random" adds N
  `joints` or `pose` commands to valid configurations: inside the joint limits, clear of the avoid
  objects, the floor and Billie's own body. A pose is the TCP of such a configuration, so it is reachable.
  The toast shows the seed; type it in the seed field to draw the same again.
- **Run on**: tick this Mac and/or robots. Robots come from `--robot` (default billie-29) or the
  "add a robot" field. All ticked machines plan the same scenario at once. A robot runs
  `robot.py`'s sync → plan in its `billie` container → fetch, with the planner code from "robot code"
  (a checkout on the robot; empty = this Mac's billie-onboard). Each machine shows its stage and log
  line live. When they finish you get ▶ play, its report and a per-command table: planner seconds /
  robot time estimate, plus the planner build.
  **Connect** (next to a robot) keeps a warm planner there.
  - It syncs once, then starts `plan_server.py` in the robot's `billie` container over one ssh
    connection (`robot_link.py`). The worker builds the planner once (about 45s on billie-29), and a
    Run on that robot then goes straight to it: no sync, no warm-up, a few seconds per run.
  - Only the recordings a scenario replays are synced per run. After changing code, reconnect.
  - The worker holds GPU memory on the robot, so it ends on Disconnect, when the editor exits or is
    killed (its input closes), or after an hour without a job (`WORLD_SIM_IDLE_S`).
  - The hover text shows its log.
  "Also plan without obstacles" (on by default) plans every command a second time with the avoid
  objects removed. It uses the same planner in the same process, so there is no second build. The
  table shows those times in grey under each cell. The difference is what the obstacles cost:
  detours, transits, reroutes and the solver's extra work. A command that fails with obstacles
  stops early, so compare its row with care.
  **Reset** puts the arm back at the start joints and clears the last run's results from the page.
  History keeps them. Run also clears the old results before it starts.
  On the play bar, a numbered marker sits where each command's last target is reached. Failed
  commands are red, a command that failed before sending any target shows "N✗", and clicking a marker
  jumps there. The playback label also shows door and handle angles, and blocked pushes.
- **History**: every Run is kept in `output/world_sim/runs/<time>_<file>/` with the scenario as it
  ran, then per machine its results (`local.json`, `billie-29.json`, ...) and report. A run also keeps
  the note typed before Run. Click a run to see its machines and timing, play one (this loads that
  run's scenario), reopen its scenario or delete it. **Clean** deletes every finished run (it asks
  first). Runs that are still going on are kept; a run whose editor died shows as "failed
  (interrupted)". `runs/index.html` ("all runs") lists every run and opens straight from disk too.
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
- A command may set `"touch": true` (or `false`): its moves let the fingers touch the room and the
  objects, 5cm deep for the finger spheres (`billie_utils.arm_touch`, `FINGER_TOUCH_DEPTH_M`). Without it, a
  `replay_policy` touches (its move to the first frame too) and `joints` / `pose` never
  touch, as on the robot (the AI and replay_policy touch, stage 1).
- An object with `"frame": "map"` is placed in the map instead: points are `[x, y, z above the floor]`
  and `yaw_deg` is from the map +X. Such objects need
  `"billie": {"pose": [x_mm, y_mm, yaw_deg], "arm_to_base": [x_mm, y_mm, yaw_deg]}`
  (`arm_to_base` = `ARM_TO_BASE_CALIBRATION`). They are converted to the arm frame with
  `map_to_arm_frame`, so moving Billie moves them relative to the arm. The editor writes this format.
  Objects without `frame` stay in the arm frame and move with Billie. `top_height_mm` / `height_mm` are above the floor; the
  floor is 440.5mm below the arm base.
- `avoid` objects go to the planner, the arm-move guard, detours and reroutes. Together they must
  fit in the planner's capsule slots (`MAX_WORLD_CAPSULES`: 32 on `arm_planner`, 8 on `arm_plan_esdf`).
  A table uses about `width / thickness` capsules.
- `"map"` (`arm_plan_esdf` only): the mapper's baked ESDF the planner plans against as the room, e.g.
  `"output/world_sim/maps/office-2-20261006-free12/esdf_grid.esdfbin"` (path from the repo root; needs
  `"billie"`). The arm is a set of spheres fitted to its collision meshes (`arm_spheres.py`); objects stay
  capsules next to it (moving obstacles: a drawer, a door). The editor draws the map's surface voxels
  around Billie (room map field under Billie, colored by height) and counts the room in the arm's
  clearance colors; robot runs copy the map file to the robot.
- A table or wall may set `"spacing_mm"`: the distance between neighbouring capsule centers (radius =
  thickness / 2). Without it the robot's builders choose: a wall every 1.2 radii (overlapping), a
  table every 2 radii (touching). A larger spacing uses fewer capsules but leaves holes of
  `spacing - thickness`.
- `eef_touch` objects are only drawn (green) and measured: EEF vs rest-of-arm distance in the
  report. The planner does not know them yet; see WORLD_COLLISION_DISCUSSION.md §3.
- `push` objects are doors and handles (the role is always `push`). They are not sent to the
  planner; the simulation turns them out of the arm's way:
  ```json
  {"name": "door 1",   "role": "push", "type": "door",   "hinge_xy_mm": [x,y], "yaw_deg": 0, "width_mm": 800, "height_mm": 2000,
   "bottom_mm": 10, "thickness_mm": 40, "direction": "ccw", "min_deg": 0, "max_deg": 90},
  {"name": "handle 1", "role": "push", "type": "handle", "start_mm": [x,y,z], "end_mm": [x,y,z], "radius_mm": 12,
   "axis": [ax,ay,az], "direction": "cw", "min_deg": 0, "max_deg": 45, "mounted_on": "door 1"}
  ```
  - A door's hinge is vertical, at `hinge_xy_mm`. `yaw_deg` points from the hinge to the free edge
    at 0°.
  - A handle's hinge passes through `start_mm` along `axis`. `end_mm` is its free end at 0°.
  - `direction` is the positive turn: `ccw` or `cw`, looking down the hinge axis from its tip (for a
    door: seen from above).
  - `min_deg` ≤ 0 ≤ `max_deg` are limits from the initial pose.
  - `mounted_on` makes a handle turn with its door. `scenarios/door_push.json` is an example.

## What is simulated

| Robot | Here |
|---|---|
| pyroki planner node (`plan`, `batch_plan`, `plan_move`, `plan_replay`, obstacles) | `planner.py`, the same `IKResolver` and transit planner in-process |
| brain `set_state_with_arm_joints` + detour, `pose`, `joints`, `replay_policy`, BufferingPlayer, reroute | `commands.py`, `reroute.py` (ports, keep in sync) |
| xarm_writer queue + arm-move guard | `sim_arm.py`, the real `arm_motion_guard_poc` |

## Real geometry check

The planner checks one capsule per link, and the capsules don't cover every link. Even fitted to
the real mesh, the gripper/camera sticks out of its capsule by up to 22mm; the link6 flange
by 35mm.

`mesh_check.py` samples every link's URDF collision mesh, posed with the run's own link frames
(so a robot run uses its calibration). It measures the real distance to the avoid objects:
- The report adds a "real mesh" column and a red box listing every move where the real mesh enters
  an object while the planner model reads clear, with the link and depth.
- A box warns when the planner used the gripper's bounding-box fallback (its gripper mesh was
  missing), because that box misses the camera bracket.

## Did it reach the target, and did Billie say so

Billie refuses moves in the brain: the arm-move guard, "no detour found", "no route found" in a
replay reroute, and the replay's final-position check. The IK solver itself always returns its best
joints. Nothing on the robot checks:
- a `pose` command's IK result, against its target;
- the replay frames on the way;
- a pose replay's last frame below `final_position_threshold` (50mm).

A target inside the robot base silently leaves the arm where it is.

So every command in a run gets a check (`analysis.target_checks`):

| Verdict | Meaning |
|---|---|
| reached | Every pose target within 5mm / 2° |
| failed | The command failed with an error: Billie reported it |
| **silent miss** | It finished, but a target was missed by more than 5mm / 2°. Billie reports success. For replays: how many frames, the worst, and the last |
| **stayed put** | A pose inside the robot base: the arm did not move, and Billie reports success |
| not checked | No pose target (`joints`, replays in joints mode) |

The report puts silent misses in a red box at the top and in the result column. The editor marks
them ⚠ in the commands list and the run panel, and counts them per machine in the run view. The sim
writes three notes the robot never sends (target inside the base, the IK error, a replay's large
motion). They are greyed and marked "simulation note" and never count as Billie having reported
anything.

## Close to collision

"Safe" is the planner's own clearance, 20mm (`WORLD_COL_MARGIN_M`). `near_collision.py` finds every
stretch of the path, within one command, where some link comes closer than that to an avoid
object, by the planner's capsule or the real mesh, whichever is closer.
- **Report:** a "Close to collision" table (command, targets, closest distance, which model, link,
  object, both minima). Orange (closer than safe) or magenta (inside) bands on the Timeline. A
  marker in the 3D scene from the closest arm point to the object.
- **Editor playback:** the same bands under the play bar. The TCP path is colored blue / orange /
  magenta. A marker sits at each zone's closest point, and clicking it, or the zone in the run panel,
  jumps there.

## Doors and handles

`pushables.py` turns every door and handle along the planned path, at 0.5° joint steps, so a link
can't jump through a door between samples. The model is kinematic:
- When an arm link would enter the object, the object turns just enough to stay clear. It only
  turns the way that takes it out of the arm.
- It stays where it was left: no spring, no inertia, no friction.
- At a limit with the arm still inside, it is **blocked**. That is a collision: the report and the
  editor show the depth and the pushing link.
- Pushing with a link other than the end effector is flagged.

The report's "Doors and handles" section lists every push: object, command, targets, angles, free
or blocked, and the links that pushed. Its view plots each angle along the path, with the limits
dashed. The PyBullet viewer draws doors and handles at 0°.

The arm reaches each target instantly (no controller dynamics, no timing). Gripper, `thing`, brain
commands inside recordings, and the `map` / `pointcloud` / `gripper` replay transforms are not simulated.

## A robot's live occupancy map

```bash
python -m Simulation.world_sim.occupancy_view --host bellboy@billie-07.bellboy   # or --file <occupancy.bin.gz>
```

Copies the mapper's live `occupancy.bin.gz` and camera path from the robot (read only) to
`output/world_sim/occupancy/<robot>/`, prints the cell count and the floor height (it should be near z = 0), and
opens a 3D page: occupied cells colored by height above the floor, the camera path in red. `--no-floor` hides the
floor, `--zmax` hides what is higher.

### The planner's room field

```bash
python -m Simulation.world_sim.esdf_view --host bellboy@billie-07.bellboy        # a local page: Update + history
python -m Simulation.world_sim.esdf_view --file <occupancy.bin.gz> --at x,y
python -m Simulation.world_sim.esdf_view --esdf <esdf_grid.esdfbin> --at x,y      # a baked map
```

With `--host` it serves a local page (http://127.0.0.1:8765/, `--port`) with an **Update** button: each update
fetches the robot's live occupancy and `robot.json` (read only) and is kept in
`output/world_sim/occupancy/<robot>/history/<time>/`. The page lists the updates, newest first, each with
`pose:[x, y, z mm, rx, ry, rz rad]` (the TCP as the brain's `pose` command takes it; from the arm's own reading,
or the planner's model when the mapper does not send it), `joints:[...]` and `base:[x, y, theta]`, each with a Copy
button; clicking one shows its 3D map, and "slices" opens its slice page. `--static` writes the files instead.

Builds the crop around the arm base the way spatial_awareness does for the planner (occupancy_field.build_field:
2.5cm, holes filled, box rule capped at 20cm; Billie's base not cleared, no AutoXing walls), or cuts a baked map's
crop, and opens `esdf_3d.html`: the room's cells as cubes colored by height, with the 5cm and 10cm keep-out shells and
the never-observed cells switchable in the legend, and Billie itself: its base, arm and gripper from the planner's
own URDF (base + the robot's arm, from `~/releases/env/*/urdf/<XARM_SN>.urdf`) at the robot's joints, placed as the
planner places the arm (AutoXing pose, `ARM_TO_BASE_CALIBRATION`, the arm base height). It also prints where the map's
floor is around Billie and what arm base height a flat floor at its wheels would mean; `--on-floor` draws Billie
standing on that floor, `--height` sets the height. Billie needs `robot.json`, which the mapper writes from
arm_plan_esdf on (with the last frame's pose and joints). `esdf_slices.html` has horizontal slices with a height slider and
vertical x-z slices with a y slider: black is occupied, the color is the distance (cm), grey is never observed (the
planner counts it as free). The red x is the arm base (default: the robot's last camera position; `--at` sets it, `--height` its height, `--uncapped` drops the
cap).

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
   - the arm's URDF from a local `~/releases/env/*/urdf`. It is used only on a robot with no URDF of
     its own (no physical arm, e.g. billie-29). Every xArm6 URDF has the same links and joints, so
     timings are unaffected; clearances differ by about a millimeter.
   - The gripper mesh `ee-rome-v0-lod.stl` is sent as a real file, taken from the local env the
     editor uses. The `BILLIE-*` envs only have a broken symlink to it. Without the mesh, the planner
     falls back to a bounding box that misses the camera bracket. The sync fails if the file doesn't
     arrive, and `robot_run_plan.sh` refuses to plan without it.
2. **run** starts `robot_run_plan.sh` inside the container. It uses the live `pyroki-planner`
   process's environment (arm, flags, GPU), the planner venv, the synced code ahead of `/app`, and its
   own JAX cache. The robot's `/app`, planner process and cache are not touched.
3. **fetch** copies the results to `output/world_sim/<robot>/` and writes their reports.

The report's "Planner timing" section shows:
- **Planner build**, one row per phase (IK setup, single and batch IK warm-ups, model export, transit
  planner build and warm-up). Each row has its seconds, Python tracing, XLA compile or cache load, and
  JAX cache hits and misses. Its verdict ("loaded from cache" / "compiled") comes from JAX's own
  cache events, next to the resolver's own guess.
- Banners when the build was paid before this run (an earlier run or the editor's start), and which
  gripper geometry the planner used (mesh or box fallback).
- Per command: single IK, detour, transit and replay batch times, split into compile and compute,
  with and without obstacles. Also a robot time estimate: it feeds the replay batch times into the
  50Hz BufferingPlayer, so a stall shows as wall time above the playback time.
- Planner calls by type: count, poses, mean / median / p95 / max, the first-call penalty, and ms per
  pose for batches.
- A "Planner timeline" view: a Gantt of the build phases and every planner call.
- The planner log: the build's and the run's log lines and planner messages ("[compile] …", cache,
  fallback warnings), captured by `planner_capture.py`.

The first run on a robot, or after the robot geometry changes, includes the JAX compile: on billie-29
about 70s single IK, 165s batch IK and 80s transit. Later runs load it from the cache.

## Finding problems: random sweeps

```bash
python -m Simulation.world_sim.sweep generate --count 100 --seed 1        # here: random scenarios
python -m Simulation.world_sim.sweep robot output/world_sim/sweeps/<id>   # plan on billie-29, fetch, summarize
open output/world_sim/sweeps/<id>/index.html
```

`generate` writes 100 scenarios with Billie's start joints, 1–3 random objects and 2–3 `joints` /
`pose` moves. About 15% also replay a cached recording with an obstacle placed across the middle of
its path, clear of its first and last frames. Every target is a valid configuration: inside the
limits, at least 2cm from the objects, above the floor, and clear of Billie's body and the arm itself
by the planner's own self-collision model. So every command has a solution.

`robot` syncs like `robot.py`, starts the sweep detached on the robot (`sweep_check.py`, so a dropped
ssh doesn't stop it), polls, fetches, and runs `summarize`. `run <folder>` runs it on this machine
instead; failure results are the same, but the timings are CPU timings.

Each command is flagged with ([sweep_check.py](sweep_check.py)):

| Kind | Flag | Meaning |
|---|---|---|
| 1 — planner miss | `solver_miss` | The command failed, but [oracle.py](oracle.py) (RRT-Connect with the guard's rules, self-collision and floor) finds a collision-free path from its start to its goal |
| | `ik_miss` | `pose` landed more than 5mm / 3° from a target its goal configuration reaches |
| | `end_state_invalid` | The arm ended in self-collision, inside an obstacle or below the floor |
| 2 — time | `replay_stall` | A replay batch took longer than its 1.0s of playback |
| | `slow_single_move` | More than 2s of planning for a `joints` / `pose` (detours) |
| | `slow_replay_start` | More than 2s of planning before a replay moves (detour to frame 0, reroute) |
| | `recompile` | More than 0.5s of JAX compiles inside one command (from JAX's compile log) |
| Probably no solution | `no_path_found` | Failed, and the oracle found no path either |
| Not judged | `goal_invalid`, `untestable_start` | The goal is invalid on the robot's arm, or the command started from an invalid state |

Every flagged command becomes a repro scenario, `flagged/<scenario>-c<k>.json`. It contains:
- the same world, the arm's joints right before the command, and only that command;
- a `flag` block: the problems, the error, timings, the goal joints, and the oracle path for solver misses.

The repro is run again right away, and the flag records whether it `reproduced`. `summarize` copies
the repros into `scenarios/` as `<sweep id>-<scenario>-c<k>.json`, so the editor lists them. It also
writes a report per flagged scenario and `index.html`: problems grouped by kind, and time
percentiles per command type. The planner build time (compile, or cache load) is in the page header.
