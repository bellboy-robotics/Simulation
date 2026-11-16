# Fixing a flow for arm awareness

Instructions for a Claude agent asked to make a room flow (`*.billie.yaml`) run well with occupancy mapping on (the
`arm_plan_esdf` arm awareness). Work from evidence: two runs of the same flow on the same robot, one without mapping
(the reference) and one with it. Change the flow only where the mapped run lost time or failed, and explain each change
with the log line that called for it.

## What changes when mapping is on

- Every arm move is planned against a room built from the live 2.5 cm occupancy map, around the arm base.
- **Space the camera never saw counts as occupied**, except "probably free" cells (unknown cells beside seen free
  space, 20 cm around where the arm is and has been, above and around Billie's base). A move into never-seen space is
  refused ("blocked by space the camera never observed").
- The planner keeps the arm 2 cm from occupied cells. In a replay or AI task the finger tips may go 8 cm in, the rest
  of the gripper and the camera may not. Frames that come closer are **shortened** (cut back to the last clear point)
  or the replay is **rerouted**. Both are slow (2–14 s per batch of 50 frames) and wrong for contact motions: the
  gripper stops 2–4 cm before a handle it should hold.
- A `joints` goal (joints, fold, look moves) is checked against Billie's body, not the arm against itself.
- The mapper takes no frames while the base drives, or while the arm moves (it waits until the arm has been still
  0.75 s and the base 1 s). So the room is only seen at stops.
- `start_active_mapping` resets the map and takes 6 front views at half speed (~60 s).

Reference numbers (room 0159, billie-26, 2026-11-16): 11.5 min without mapping; 26 min mapped as the flow was; about
15 min estimated with the fixes below (13.5 min once the export no longer competes with the planner's room builds).

## 1. Get the two runs

The user runs the robot; you only read. Ask for (or find in the logs) one clean full run **without mapping** and one
**with mapping**, same robot, same flow, same day if possible.

- Reference run: mapping deleted first (`delete_mapping`), flow without `start_active_mapping`.
- Mapped run: the flow as it is, starting with `start_active_mapping`.

Logs are per node at `/app/out/current/log_<node>.txt` inside the container, in UTC (the robots' hosts show local
time). You need `brain` and `pyroki_planner`; `spatial_awareness`, `xarm_writer` and `nav_slide` help with specific
problems.

```bash
ssh bellboy@billie-26.bellboy 'docker ps --format "{{.Names}} {{.Image}} {{.Status}}"'      # billie-dev or billie
ssh bellboy@billie-26.bellboy 'docker exec billie-dev cat /app/out/current/log_brain.txt' > log_brain.txt
# container stopped: a dev container's /app is a checkout on the host, e.g.
ssh bellboy@billie-26.bellboy 'cat /home/bellboy/users/<user>/billie-onboard/billie/out/current/log_brain.txt'
grep "Starting flow\|Finished execution of flow\|delete_mapping" log_brain.txt     # each run's start and end
```

Read only. Never `cat` the robot's env or settings files, `/proc/*/environ`, or `docker inspect` without a `--format`
that leaves the environment out.

## 2. Measure

From the Simulation repo (`src/Simulation/flow_fix/`):

```bash
PYTHONPATH=src .venv/bin/python -m Simulation.flow_fix.room_timing \
    --brain map/log_brain.txt --planner map/log_pyroki_planner.txt --start 09:58:40 --end 10:24:38
PYTHONPATH=src .venv/bin/python -m Simulation.flow_fix.compare_runs \
    --a-brain ref/log_brain.txt --a-planner ref/log_pyroki_planner.txt --a-start 10:32:52 --a-end 10:43:07 \
    --b-brain map/log_brain.txt --b-planner map/log_pyroki_planner.txt --b-start 09:58:40 --b-end 10:24:38
```

`room_timing` prints every action with its duration, the planner time inside it, and its warnings, refusals and
aborts. `compare_runs` matches the two runs action by action, totals them per command, and lists what ran only in the
mapped run (manual recoveries, retries, the scan). Take the end time from "Finished execution of flow": the time after
it is idle, not the last action.

## 3. Diagnose

| What you see in the mapped run | Log line | Cause | Fix |
|---|---|---|---|
| A `joints` or `navigate_poi` refused, then the user moved things by hand (`check_joints`, `joints`, `scan`, `manual`) | "Arm move blocked ... -N mm from the room", "blocked by space the camera never observed (N spots of 5cm)" | the arm's path crosses never-seen space | F1 at the start of the room, else F4 |
| A replay that touches something plays slowly or stops short | "Playback: 50s for 655 frames (... actual 13 fps)", "N/50 moves shortened to stay out of a world obstacle", "Rerouted frames A-B" | the planner keeps the gripper off the object it must touch | F2 |
| A replay used only to reach a hard pose is rerouted or slow | "Rerouted frames 0-117", playback at 15 fps | the planner reroutes the recording's path | F3 |
| A `joints` after a base move waits 9–20 s | "Move planned in 16s: direct" (the plan itself is fast) | waiting for the room's first build at the new place | none in the flow (spatial_awareness); F4 does not help |
| Moves blocked right after entering a room or a bathroom backwards | blocked by never-seen space behind Billie | the camera faced away from where Billie went | F6 |
| Photos or moves blocked under furniture (bed, desk) | blocked by "occupied" cells above the floor | the floor there was never seen | F4 (a look at the floor) |
| A replay fails before it starts | "Base pose proximity exceeded", "First frame pose is too far away from the recording" | Billie stopped off the recording's spot | not mapping: F2's `transform: map`, or a tighter `min_distance` before it |
| Door handle registration fails | "GPA fitness (0.0) is less than threshold (0.6)" with only 3–5k handle points | not shown to be mapping | report it; do not change the flow for it |

Count each problem's cost (mapped minus reference for that action, plus the manual recovery after it) so the user can
see which fixes matter.

## 4. Fixes

Edit the flow file the user gives you. The source of most flows is the `flows-code` repository: a downloaded
`.billie.yaml` is overwritten by the next build, so say that the change must also go into the source. After every
edit, check the YAML still parses (`yaml.safe_load`).

### F1 · Map from further back

Before `start_active_mapping`, move Billie back from the spot where the room's work starts, so the first scan sees the
space Billie will stand and work in next. Otherwise, when Billie later moves back or turns, never-seen space around its
base blocks the next `joints`.

```yaml
    - light_on: {}
    - slide:
        direction: backward
        meters: 0.45          # was 0.15: about 0.3 m further back for the scan
        wait: true
        timeout_seconds: 30
    - start_active_mapping: {}
    - navigate_poi:            # then to the room's working spot, as before
        poi: [0.0175, 0.04, 1.56]
        min_distance: 0.2
```

Prefer a slide to a POI behind the door: the room map may not cover the corridor. Ask the user how much room there
is behind Billie.

### F2 · `direct_action` for replays that touch objects

A replay that must touch or hold something (open or close a door or cabinet, hook a handle, lift a lid) gets
`options: direct_action: true`. The planner then skips the room, objects and floor for the whole replay, including the
move to its first frame; it still checks joint limits and self-collision. The replay plays as recorded, at full speed.

```yaml
    - replay_policy:
        repo_id: bellboy-robotics/B-standard_br_close_door_1-20260128-132455-BILLIE-14
        transform: none
        options:
          attempts: 5
          direct_action: true    # contact replay: no room check (it must reach the handle)
          collision_sensitivity: 0
          record: true
          task: seminaris-potsdam-close-door-1
```

With `direct_action` nothing checks where Billie stands. If the replay uses `transform: none` and the base stops a
few cm off the recording's spot (see "Initial map coordinates" against the recording's base pose from
`recording_joints`), consider `transform: map` with `navigate: false`: it shifts the poses by the base's offset. Ask
the user before changing a replay's transform.

Do not add `direct_action` to replays that only move the arm through free space: the room check is what protects them.

### F3 · Replace a replay with `joints` when it only reaches a pose

When a replay's only job is to bring the arm to a hard place (under the bed, behind a chair) and the next step is a
photo or another move, replace it with a `joints` action to the recording's last frame:

```bash
PYTHONPATH=src .venv/bin/python -m Simulation.flow_fix.recording_joints bellboy-robotics/<repo_id>
```

It prints the joints at 0 %, 50 % and 100 %, whether the arm was still at the end, and how far the recording's path
strays from a straight joints move. A `joints` action moves all joints together; if the recording turned first and
lowered later ("add a via point"), add the 50 % joints as a step before the last frame. If the next item already moves
to a nearby pose, say the new step may be unnecessary.

Keep the replay when it touches anything (F2 instead).

### F4 · Stops to look before moving into unseen space

Where the mapped run was blocked by never-seen space (or the user had to move the camera to clear it), add a look
before that move: a `joints` action that aims the camera at the problem area, held still long enough for the mapper.

- Use the user's own recovery poses from the mapped run when there are any (`joints` and `check_joints` in the log
  around the block): they are what cleared it.
- For space under furniture, turn the folded arm (J1, and J5 down) to look at the floor in front of it.
- The mapper adds a frame only after the arm has been still 0.75 s (and the base 1 s); it needs about 3 frames, up to
  3 s. There is no flow action that waits for the mapper: hold the pose with one or two `photo: {}` actions (about
  1 s each), and check in `log_mapper` that frames were added at that time. Tell the user that a `wait` action would
  be cleaner.
- On a long base move into an area the arm will work in, stop once on the way (an intermediate `navigate_poi`), so the
  mapper sees the area before Billie arrives: it takes no frames while driving.

Every look pose is a new arm pose: check it with `pose_report` (below), and the user tries it on a robot.

### F5 · A fold that sees the room

Billie carries the arm folded between items. If the fold points the camera at the floor next to Billie, the mapper
learns little at each stop. Choose a travel pose that is compact and looks out at the room (level or slightly down,
ahead), from the flow's own photo poses where possible, and use it instead of `fold: {}` between items where the arm
will work next. Requirements, from `pose_report`:

- "reaches x" no further back than Billie's outline (about -0.23 m): it must not catch door frames;
- body clearance ≥ +20 mm, and the straight move from the previous pose "ok";
- "looks" pointing at the room, not at the floor right beside the base.

### F6 · Enter facing the direction of travel

When Billie drives into a space backwards (a bathroom, a narrow corridor), turn the arm so the camera looks the way
Billie goes, and keep that pose through the slides and twists. Example (room 0159's bathroom, the flow's own "Mirror
and lights" photo pose: camera level and backwards, 0.81 m above the arm base, nothing behind Billie's outline, 37 mm
from its body):

```yaml
    - joints:                  # instead of fold: {} before backing into the bathroom
        joints: [-189.12, -55.33, -153.42, 3.13, 118.14, -178.74]
        speed: 80
    - slide: {direction: backward, meters: 0.5, wait: true, timeout_seconds: 30}
    - twist: {direction: left, steps: 1}
    - slide: {direction: backward, meters: 0.6, wait: true, timeout_seconds: 30}
```

The mapper still adds frames only at the stops between the slides; the pose makes those frames show the space ahead.

### Also

- One `start_active_mapping` per room, after `light_on` (the scan needs light).
- A flow that changes floors (an elevator) should `delete_mapping` before the new floor until the mapper handles it.
- Leave look-before-acting (the planner's automatic looks) alone: flow looks are explicit and cheaper.

## 5. Check new poses

Every new or changed arm pose (F3, F4, F5, F6), with the robot's own with-tcp URDF (copied from its logs folder or
`/tmp/urdf/<SN>-with-tcp.urdf`; never call `urdf.load_urdf` on a robot: it uploads the URDF to S3):

```bash
PYTHONPATH=src:billie-onboard/billie/nodes/pyroki-planner:billie-onboard/billie/billie-utils \
billie-onboard/billie/nodes/pyroki-planner/.venv/bin/python -m Simulation.flow_fix.pose_report \
    --urdf <SN>-with-tcp.urdf --pose "look back=-189.12,-55.33,-153.42,3.13,118.14,-178.74" \
    --from "previous=<the joints the arm comes from>"
```

Reject a pose with negative body clearance, or a move from the previous pose that comes out "REFUSED". The planner's
model is a little bigger than the real arm, so a pass here is necessary, not sufficient: the user tries each new pose
on a robot before it runs in a hotel.

## 6. Hand over

- Show each change as a diff, with the log line and the time it should save (from `compare_runs`).
- Estimate the mapped run after the fixes: the reference time + the scan (~60 s) + the looks (~5 s each) + the room
  waits that remain. Say it is an estimate.
- Say what is not fixed by the flow (room waits after base moves, registration failures, base stopping off its spot).
- The user runs the new flow; compare again with `compare_runs` against the same reference.

## Rules

- Never send a command that moves an arm. Robots with real arms that are never moved by an agent: billie-12, billie-16.
  On other robots, only the user moves the arm.
- `start_active_mapping` exists only on `arm_plan_esdf` builds: a robot on a main release fails at that step.
- Do not change a replay's transform, thresholds or recording without the user's approval.
- Keep notes of what changed and why in your answer, not in the flow file.
