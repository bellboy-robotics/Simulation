# Fixing a flow for arm awareness

Instructions for a Claude agent asked to make a room flow (`*.billie.yaml`) run well with occupancy mapping on (the
`arm_plan_esdf` arm awareness).

Work in two passes:

1. **Before the first mapped run** (most of the work). From the flow file, a clean reference run **without** mapping,
   the replays' recordings and the robot's URDF, find the places where the mapped run would stop, and fix them. A
   mapped run of an unfixed flow is slow and hard to run: blocked moves need someone to recover them by hand.
2. **After the mapped run.** Measure it against the reference and fix what is left.

Explain every change with what called for it: a line of the flow, a pose report, a log line.

## What changes when mapping is on

- Every arm move is planned against a room built from the live 2.5 cm occupancy map, around the arm base.
- **Space the camera never saw counts as occupied.** The exceptions are "probably free" cells:
  - unknown cells beside seen free space;
  - 20 cm around where the arm is and has been;
  - above and around Billie's base.

  A move into never-seen space is refused ("blocked by space the camera never observed"). The map keeps everything seen
  since `start_active_mapping`, from every stop.
- The planner keeps the arm 2 cm from occupied cells. In a replay or AI task the finger tips may go 8 cm in; the rest
  of the gripper and the camera may not. Frames that come closer are **shortened** (cut back to the last clear point),
  or the replay is **rerouted**. Both are slow (2–14 s per batch of 50 frames) and wrong for contact motions: the
  gripper stops 2–4 cm before a handle it should hold.
- A `joints` goal (joints, fold, look moves) is checked against Billie's body, not the arm against itself.
- The mapper takes no frames while the base drives or while the arm moves. It waits until the arm has been still
  0.75 s and the base 1 s, then needs about 3 frames (up to 3 s). So the room is seen only at stops.
- `start_active_mapping` resets the map and takes 6 views at half speed (~60 s). The views cover only the front:
  straight ahead and 30° to each side, level and down. Nothing behind or beside Billie is seen.
- **`navigate_poi` folds the arm before driving** (`fold_arm: true` by default). The fold points the camera straight
  down at the floor just in front of the base (billie-26's fold, `pose_report`: looks [-0.03, -0.04, -1.0]). So
  after a drive, the mapper has seen only that patch of floor at the new place.
- After every base move, the planner's first arm move waits for the room to be built at the new place (9–18 s on
  billie-26). The flow cannot fix this; count it in estimates.

Reference numbers (room 0159, billie-26, 2026-11-16):

| Run | Time |
|---|---|
| Without mapping | 11.5 min |
| Mapped, flow unchanged | 26 min |
| Mapped, with the fixes below (estimate) | about 15 min |

## Inputs

Ask the user for:

- **The flow file.** The source of most flows is the `flows-code` repository. A downloaded `.billie.yaml` is
  overwritten by the next build, so every change must also go into the source.
- **A clean reference run without mapping:** same robot, same flow, from start to "Finished execution of flow", with
  no manual help.
- **The robot's with-tcp URDF:** from its logs folder, or `/tmp/urdf/<SN>-with-tcp.urdf` on the robot. Never call
  `urdf.load_urdf` on a robot: it uploads the URDF to S3.
- **How much room there is behind Billie at the door** (for F1), and whether Billie may drive with the arm up (for F5
  and F6).

Logs are per node at `/app/out/current/log_<node>.txt` inside the container, in UTC (the robots' hosts show local
time). You need `brain` and `pyroki_planner`; `nav_slide`, `xarm_writer` and `spatial_awareness` help with specific
problems.

```bash
ssh bellboy@billie-26.bellboy 'docker ps --format "{{.Names}} {{.Image}} {{.Status}}"'      # billie-dev or billie
ssh bellboy@billie-26.bellboy 'docker exec billie-dev cat /app/out/current/log_brain.txt' > log_brain.txt
# container stopped: a dev container's /app is a checkout on the host, e.g.
ssh bellboy@billie-26.bellboy 'cat /home/bellboy/users/<user>/billie-onboard/billie/out/current/log_brain.txt'
grep "Starting flow\|Finished execution of flow\|delete_mapping" log_brain.txt     # each run's start and end
```

Read only. Never `cat` the robot's env or settings files, or `/proc/*/environ`. Never run `docker inspect` without a
`--format` that leaves the environment out.

## The tools

All run from the Simulation repo (`src/Simulation/flow_fix/`). None of them moves anything.

| Tool | What it gives |
|---|---|
| `flow_check` | Before any mapped run: walks the flow as the camera and lists, in flow order, the arm moves into space no view covered, the low poses with no close floor look, and the spots for F1, F2, F3, F5 and F6. |
| `room_timing` | A run's timeline: every action's duration, the planner time inside it, and its warnings, refusals and aborts. |
| `compare_runs` | Two runs matched action by action: totals per command, and what ran in only one of them. |
| `recording_joints` | A replay's joints at 0/50/100 %, its base pose, its gripper, and whether a straight joints move would cut a corner. |
| `pose_report` | Where poses put the camera ("looks"), how far the arm reaches along and beside Billie, the clearance to Billie's body, and whether the straight move from a previous pose passes. |

```bash
PYTHONPATH=src:billie-onboard/billie/nodes/pyroki-planner:billie-onboard/billie/billie-utils \
billie-onboard/billie/nodes/pyroki-planner/.venv/bin/python -m Simulation.flow_fix.flow_check \
    flow.billie.yaml --urdf <SN>-with-tcp.urdf --fold "<the robot's fold joints>" [--start-pose x,y,heading]
PYTHONPATH=src .venv/bin/python -m Simulation.flow_fix.room_timing \
    --brain ref/log_brain.txt --planner ref/log_pyroki_planner.txt --start 10:32:52 --end 10:43:14
PYTHONPATH=src .venv/bin/python -m Simulation.flow_fix.compare_runs \
    --a-brain ref/log_brain.txt --a-planner ref/log_pyroki_planner.txt --a-start 10:32:52 --a-end 10:43:14 \
    --b-brain map/log_brain.txt --b-planner map/log_pyroki_planner.txt --b-start 09:58:40 --b-end 10:24:38
PYTHONPATH=src .venv/bin/python -m Simulation.flow_fix.recording_joints bellboy-robotics/<repo_id>
PYTHONPATH=src:billie-onboard/billie/nodes/pyroki-planner:billie-onboard/billie/billie-utils \
billie-onboard/billie/nodes/pyroki-planner/.venv/bin/python -m Simulation.flow_fix.pose_report \
    --urdf <SN>-with-tcp.urdf --pose "name=j1,j2,j3,j4,j5,j6" [--pose ...] [--from "previous=j1,...,j6"]
```

Take a run's end from its "Finished execution of flow" line: the time after it is idle, not the last action.

## Pass 1 · Before the first mapped run

### 1.1 Outline the flow and time the reference

List the flow's areas, items and actions in order. For every place Billie stands still, note:

- how Billie got there (`navigate_poi` with coordinates, `slide`, `twist`);
- whether the arm was folded on the way;
- the arm goals there: `joints`, `fold`, replays, AI tasks.

Run `room_timing` on the reference run. Note:

- each action's time;
- each replay's playback rate ("Playback: ...");
- "Initial map coordinates" (where the base really stood at each replay).

These times are the baseline for the estimate.

### 1.2 Walk the flow as the camera

Start with `flow_check`: it does this walk for you and lists the spots in flow order. Then read the flow around each
spot it lists, and around the cases below that it cannot see.

What `flow_check` assumes (it is optimistic, so a line it prints is worth acting on, and silence is no promise):

- **Walls and furniture hide nothing.** Space behind a bed, inside a cabinet or under furniture counts as seen when
  the camera pointed that way. Only the low-pose rule (a close look down at the floor) catches part of it.
- **Billie's pose comes from the flow:** `navigate_poi` coordinates, slides along the heading, and twists at about 26°
  a step.
- **The scan's place:** after a named POI or a map change, Billie's place is unknown. The scan's place then comes from
  `--start-pose`, or else from the room's first `navigate_poi`, moved back by the slides before the scan. Check the
  line it prints about that.
- **The robot's fold:** pass it with `--fold`. It is the `ARM_FOLD` setting; take it from the reference run's log
  (a replay right after a `navigate_poi` starts from the fold: its "Setting initial joints to"), or ask the user.
- **On room 0159 (billie-26, 2026-11-16)** it flagged:
  - the one move the mapped run lost to never-seen space ("Hanging chair and side table", 6 unseen arm points);
  - the move before it (100 points), which passed on the robot;
  - the two low poses at the cabinet and under the bed.

  The two other blocks of that run were cells seen as occupied ("-30 mm from the room"), which no flow check can
  predict.

To check a spot yourself, compare **where the camera has looked** (in the room so far) with **where the arm goes**.
Run `pose_report` on the arm poses around it, each with `--from` the pose before it.

The camera has looked:

- at the start: the scan's six views (front, ±30°, level and down);
- at every stop after `navigate_poi`: the fold's patch of floor in front of the base;
- at each `joints` or `photo` pose, held still: along its "looks", from its "tcp";
- from earlier stops: what those saw, if the area is in view and within about 3 m.

An arm goal is **at risk** when its "tcp", or the arm on the straight way there, goes where no earlier view looked.
The typical cases:

| Case | Example | Fix |
|---|---|---|
| The first moves right after the scan, where Billie stood or will stand | the space Billie backs into at the door | F1 |
| Poses beside or behind Billie: "tcp" y beyond ±0.3 m, or x < 0 | a photo of a side table, before any view looked to that side | F4 |
| Poses low, close to the floor: "tcp" z below about -0.25 m, anywhere | a photo of the trash can, a reach to the floor | F4 (a floor look), even in open space: the occupancy grid needs to have seen the floor there to place it correctly |
| Poses low under furniture: under a bed or desk | "Under the bed" | F4 (a floor look). Occlusion: seen from above, the floor under a bed is still unknown. |
| Everything after Billie drives or slides backwards into a space | backing into a bathroom | F6 |
| The first arm goal after a long drive, into the area ahead | the window side after crossing the room | F5, or F4's stop on the way |
| Replays that touch or hold something | door, cabinet, handle | F2 |
| Replays that only bring the arm to a pose | reaching under the bed | F3 |

The cost is lopsided: a blocked move stops the flow until someone recovers it by hand (minutes), while a look costs
about 4 s. When unsure whether a place was seen, add the look.

### 1.3 Fix, in this order

F1 (the start), F6 (backwards entries), F5 (travel pose), F4 (looks for what remains), F2 and F3 (replays). The fixes
are under [Fixes](#fixes). Check the YAML still parses after every edit (`yaml.safe_load`).

### 1.4 Check every new or changed pose

Run `pose_report` on every new or changed pose, with `--from` the pose the arm comes from (sections 1.2 and F3–F6).
Reject a pose when:

- its body clearance is negative;
- the move from the previous pose comes out "REFUSED";
- it is a travel pose (F5, F6) that reaches more than about 5 cm further forward than the fold (billie-26's fold:
  +0.34 m; Rome's room fold: +0.39 m), behind Billie's outline (-0.23 m), or to the side (0.24 m).

The planner's model is a little bigger than the real arm, so a pass is necessary, not sufficient. The user tries each
new pose on a robot before it runs in a hotel.

### 1.5 Hand over pass 1

- **Each change as a diff,** with what called for it (the case from 1.2 and the pose report line).
- **An estimate of the mapped run,** marked as an estimate:
  - the reference time;
  - plus the scan (~60 s);
  - plus about 4 s per added look;
  - plus about 10 s for the room build after each base move that ends in arm work;
  - plus the time the replays without `direct_action` lose (if the reference shows 2–14 s of batch IK per 50 frames
    with mapping, count it).
- **What a flow cannot fix:** the room build after base moves, registration failures, the base stopping off its
  spot.
- **The doubts:** the places that may still block, so the user watches them in the first mapped run.

## Pass 2 · After the mapped run

Run `room_timing` on the mapped run and `compare_runs` against the same reference. Then fix what the first pass
missed:

| What you see in the mapped run | Log line | Cause | Fix |
|---|---|---|---|
| A `joints` or `navigate_poi` refused, then the user moved things by hand (`check_joints`, `joints`, `scan`, `manual`) | "Arm move blocked ... -N mm from the room", "blocked by space the camera never observed (N spots of 5cm)" | the arm's path crosses never-seen space | F1 at the start of the room, else F4. The user's recovery poses are the look to add. |
| A replay that touches something plays slowly or stops short | "Playback: 50s for 655 frames (... actual 13 fps)", "N/50 moves shortened to stay out of a world obstacle", "Rerouted frames A-B" | the planner keeps the gripper off the object it must touch | F2 |
| A replay used only to reach a hard pose is rerouted or slow | "Rerouted frames 0-117", playback at 15 fps | the planner reroutes the recording's path | F3 |
| A `joints` after a base move waits 9–20 s | "Move planned in 16s: direct" (the plan itself is fast) | waiting for the room's first build at the new place | none in the flow |
| Moves blocked right after entering a space backwards | blocked by never-seen space behind Billie | the camera faced away from where Billie went | F6 |
| Photos or moves blocked under furniture | blocked by "occupied" cells above the floor | the floor there was never seen | F4 (a floor look) |
| A replay fails before it starts | "Base pose proximity exceeded", "First frame pose is too far away from the recording" | Billie stopped off the recording's spot | not mapping: F2's `transform: map`, or a tighter `min_distance` before it |
| Door handle registration fails | "GPA fitness (0.0) is less than threshold (0.6)" with only 3–5k handle points | not shown to be mapping | report it; do not change the flow for it |

Count each problem's cost (mapped minus reference for that action, plus the manual recovery after it), so the user
sees which fixes matter.

## Fixes

### F1 · Map from further back

**Find it before the run:** the room's first `start_active_mapping` (or where it is missing), and the moves just after
it.

Before `start_active_mapping`, move Billie back from the spot where the room's work starts. The scan then also sees
the space Billie will stand and work in next. Otherwise, when Billie later backs or turns, never-seen space around its
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

- One `start_active_mapping` per room, after `light_on` (the scan needs light).
- Prefer a slide to a POI behind the door: the room map may not cover the corridor.
- Ask the user how much room there is behind Billie.

### F2 · `direct_action` for replays that touch objects

**Find it before the run:** any of these is a contact replay:

- its repo or task names a door, handle, cabinet, drawer, lid or curtain, or says open or close;
- a `close_gripper` comes before it;
- `recording_joints` shows the gripper closing during the recording (its min well below its first value).

A replay that must touch or hold something gets `options: direct_action: true`. The planner then skips the room,
objects and floor for the whole replay, including the move to its first frame. It still checks joint limits and
self-collision. The replay plays as recorded, at full speed.

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

With `direct_action`, nothing checks where Billie stands. For a `transform: none` replay, compare the reference run's
"Initial map coordinates" with the recording's base pose (from `recording_joints`). If the base stops a few cm off,
suggest `transform: map` with `navigate: false`, which shifts the poses by the base's offset. Ask the user before
changing a replay's transform.

Do not add `direct_action` to replays that only move the arm through free space: the room check is what protects
them.

### F3 · Replace a replay with `joints` when it only reaches a pose

**Find it before the run:** a replay whose recording does not change the gripper and touches nothing, followed by a
photo or another move (from `recording_joints`). Ask the user when the name does not say what it does.

Replace it with a `joints` action to the recording's last frame. `recording_joints` prints:

- the joints at 0 %, 50 % and 100 %;
- whether the arm was still at the end;
- how far the recording's path strays from a straight joints move.

A `joints` action moves all joints together. If the recording turned first and lowered later ("add a via point"), add
the 50 % joints as a step before the last frame. Check both steps with `pose_report --from`. If the next item already
moves to a nearby pose, say that the new step may be unnecessary.

A replay that brings the arm back out of a tight place (e.g. `fold_from_under_bed`) is a path, not a pose. Keep it,
unless the gap check and `pose_report` show that a via point and the last frame do the same.

Keep the replay when it touches anything (F2 instead).

### F4 · Stops to look before moving into unseen space

**Find it before the run:** the at-risk goals from 1.2 that F1, F5 and F6 do not cover. After the mapped run, the
blocked moves.

Before the at-risk move, add a look: a `joints` action that aims the camera at the area, held still long enough for
the mapper.

```yaml
    - joints:                  # look at the floor under the bed before reaching there
        joints: [...]
        speed: 80
    - sleep:
        seconds: 2             # arm still 0.75 s, then ~3 frames
```

- **Close to the floor:** before any pose whose "tcp" comes near the floor, look at the floor where the gripper will
  go, from close and above. Without it, the occupancy grid may not place the floor there correctly: unseen floor
  counts as occupied, and a floor seen only from far, at a slant, can come out at the wrong height. The scan's low
  views see the floor ahead at a distance, and the fold sees only the patch in front of the base.
- **Under furniture:** turn the folded arm (J1) toward it and tilt the camera down at the floor in front of it (J5).
  From higher up, the floor under the furniture stays hidden.
- **Beside or behind Billie:** reuse a photo pose of the flow that already looks that way, earlier, from the same
  stop.
- **After the mapped run:** use the user's own recovery poses (the `joints` and `check_joints` in the log around the
  block). They are what cleared it.
- **Long base moves** into an area the arm will work in: stop once on the way (an intermediate `navigate_poi` with
  `fold_arm: false`, the arm in the F5 pose), so the mapper sees the area before Billie arrives. It takes no frames
  while driving.
- **Don't double up:** a `photo` already holds the arm still. A look right before a photo pose that looks the same way
  is unnecessary.
- **If the flow build rejects `sleep`:** hold with one or two `photo: {}` actions instead, and tell the user.

### F5 · Travel with a pose that sees the room

**Find it before the run:** every `navigate_poi` and `fold` followed by arm work at the new stop, especially after
long drives.

`navigate_poi` folds the arm, and the fold sees only the floor in front of the base. Do not change the robot's fold:
hotels want the camera looking down while Billie travels the corridors. A separate room fold is planned for v2.
Until then, inside a room, move the arm to a room travel pose with `joints` before each `navigate_poi`, and drive
without folding. The mapper then sees the area ahead at every stop.

```yaml
    - joints:                  # room travel pose: looks ahead and down
        joints: [0.96, -48.69, 8.03, -2.59, -18.71, 177.23]
        speed: 80
    - navigate_poi:
        poi: [0.1, 4.51, 1.53]
        min_distance: 0.2
        fold_arm: false        # keep the room travel pose
```

Two candidates, from `pose_report` on billie-28's URDF:

| Pose | Joints | tcp | Looks | Reaches x, side | Body, arm against itself |
|---|---|---|---|---|---|
| Room fold (Rome's) | [0.96, -48.69, 8.03, -2.59, -18.71, 177.23] | [0.39, 0.02, 0.20] | ahead, ~30° down | -0.16..+0.39 m, 0.07 m | +37 mm, **-7 mm** (link4~link1) |
| See ahead (the scan's centre level view) | [0.0, -36.88, -94.14, -179.10, -56.44, -1.32] | [0.34, 0.0, 0.65] | ahead, ~15° down | -0.14..+0.34 m, 0.07 m | +37 mm, +51 mm |
| *the fold, for comparison* | billie-26's [0.96, -13.94, -17.8, -2.59, 33.74, -177.23] | [0.29, 0.0, 0.01] | straight down | -0.03..+0.34 m, 0.07 m | +37 mm, +62 mm |

- **The room fold** is the one already in use (Rome), and it is low (tcp 0.2 m above the arm base). It reaches 5 cm
  further forward than the fold. On billie-28's model its links 1 and 4 overlap by 7 mm. A `joints` move to it is
  checked only against Billie's body, so it passes. A planned move (a pose goal or a replay) that starts from it may
  be refused: check on the robot, or move with `joints` first.
- **See ahead** has full clearance everywhere and sees further. Its top is 0.65 m above the arm base, so the user
  approves driving with the arm that high.
- On billie-28's model, the moves to both from the door pose, the fold and "Under the bed" pass. Under the bed to the
  room fold is tight: +6 mm. The move from there to the plain fold is refused (-22 mm), which is why that flow uses a
  `fold_from_under_bed` replay.

Requirements for any travel pose:

- about as compact as the fold (at most ~5 cm further forward, nothing outside Billie's outline behind or to the
  side), so it cannot catch furniture or door frames;
- body clearance ≥ +20 mm, and the straight move from the previous pose "ok";
- "looks" ahead at the room, level or slightly down.

Keep the plain fold (a `navigate_poi` without `fold_arm: false`) for the door passes, for leaving the room and in
corridors.

### F6 · Enter facing the direction of travel

**Find it before the run:** `slide` with `direction: backward`, or a `navigate_poi` whose heading points away from
where Billie drives, followed by arm work there.

When Billie drives into a space backwards (a bathroom, a narrow corridor), turn the arm so the camera looks the way
Billie goes, and keep that pose through the slides and twists. The example is room 0159's bathroom. The flow's own
"Mirror and lights" photo pose has the camera level and backwards, 0.81 m above the arm base, nothing behind Billie's
outline, 37 mm from its body:

```yaml
    - joints:                  # instead of fold: {} before backing into the bathroom
        joints: [-189.12, -55.33, -153.42, 3.13, 118.14, -178.74]
        speed: 80
    - slide: {direction: backward, meters: 0.5, wait: true, timeout_seconds: 30}
    - twist: {direction: left, steps: 1}
    - slide: {direction: backward, meters: 0.6, wait: true, timeout_seconds: 30}
```

- A `navigate_poi` in between needs `fold_arm: false`, or it folds the arm away again.
- The mapper still adds frames only at the stops between the slides. The pose makes those frames show the space
  ahead.
- Twist steps are not a fixed angle: in 0159, 7 steps turned Billie about half a turn.

### Also

- A flow that changes floors (an elevator) should `delete_mapping` before the new floor, until the mapper handles it.
- Leave look-before-acting (the planner's automatic looks) alone: flow looks are explicit and cheaper.
- `start_active_mapping` exists only on `arm_plan_esdf` builds: a robot on a main release fails at that step.

## Rules

- Never send a command that moves an arm. billie-12 and billie-16 have real arms that an agent never moves. On other
  robots, only the user moves the arm.
- Do not change a replay's transform, thresholds or recording without the user's approval.
- Keep notes of what changed and why in your answer, not in the flow file.
