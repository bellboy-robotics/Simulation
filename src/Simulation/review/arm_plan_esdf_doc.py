"""Builds the arm_plan_esdf PR review document (PR #1386) and an HTML preview of it.

    python -m Simulation.review.branch_inventory --repo billie-onboard \
        --out billie-onboard/out/review/arm_plan_esdf/inventory.json
    python -m Simulation.review.arm_plan_esdf_doc

Same method and convention as arm_planner_doc.py and free_area_doc.py: the charts are written below by hand, one
per trigger; the tables come from the inventory, so every function and class the branch adds or changes is listed.
The function tables are longer than one GitHub comment, so they are split over several comments, file by file.
"""

import json
import os
import re
import sys

from Simulation.review import arm_planner_doc
from Simulation.review.arm_planner_doc import (
    _CHART_MENTION,
    _GITHUB_MAX_BYTES,
    _NODE,
    chart_items,
    kind_marker,
    label_strings,
    mentioned_charts,
    preview_html,
    table,
    wrap_labels,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# arm_plan_esdf is developed in billie-onboard, so its review lives there, in out/ (ignored, never pushed).
_OUT_DIR = os.path.abspath(os.path.join(_ROOT, "billie-onboard", "out", "review", "arm_plan_esdf"))
# The inventory branch_inventory.py wrote, the PR description, the comment parts (numbered), the preview.
INVENTORY = os.path.join(_OUT_DIR, "inventory.json")
OUT_MD = os.path.join(_OUT_DIR, "arm_plan_esdf.md")
OUT_COMMENT_MD = os.path.join(_OUT_DIR, "arm_plan_esdf_comment_{}.md")
OUT_HTML = os.path.join(_OUT_DIR, "arm-plan-esdf-review.html")
# [bytes] A comment part is closed before it passes this, leaving room under GitHub's 65,536 for the heading.
_PART_BYTES = 60_000

# Node styles: fill = process (b brain, p pyroki-planner, s spatial_awareness, m nvblox_mapper, c the camera nodes,
# x other nodes: nav_events, status, the uploader; u shared billie_utils code), border = change (New / Chg / Old).
STYLES = """  classDef trig fill:#ffffff,stroke:#1d2330,stroke-width:1.5px,color:#1d2330;
  classDef bNew fill:#dbe8fb,stroke:#2f62b0,stroke-width:3px,color:#1d2330;
  classDef bChg fill:#dbe8fb,stroke:#2f62b0,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef bOld fill:#dbe8fb,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef pNew fill:#eadff8,stroke:#6d40b8,stroke-width:3px,color:#1d2330;
  classDef pChg fill:#eadff8,stroke:#6d40b8,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef pOld fill:#eadff8,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef sNew fill:#fbe7d2,stroke:#a95a14,stroke-width:3px,color:#1d2330;
  classDef sChg fill:#fbe7d2,stroke:#a95a14,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef sOld fill:#fbe7d2,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef mNew fill:#d4efed,stroke:#1f7a74,stroke-width:3px,color:#1d2330;
  classDef mChg fill:#d4efed,stroke:#1f7a74,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef mOld fill:#d4efed,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef cNew fill:#ececec,stroke:#5f6368,stroke-width:3px,color:#1d2330;
  classDef cChg fill:#ececec,stroke:#5f6368,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef cOld fill:#ececec,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef uNew fill:#e1efe4,stroke:#3f7a52,stroke-width:3px,color:#1d2330;
  classDef uChg fill:#e1efe4,stroke:#3f7a52,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef uOld fill:#e1efe4,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef xNew fill:#fbf3cf,stroke:#8a6d00,stroke-width:3px,color:#1d2330;
  classDef xChg fill:#fbf3cf,stroke:#8a6d00,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef xOld fill:#fbf3cf,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef bDec fill:#dbe8fb,stroke:#2f62b0,stroke-width:1.5px,color:#1d2330;
  classDef pDec fill:#eadff8,stroke:#6d40b8,stroke-width:1.5px,color:#1d2330;
  classDef sDec fill:#fbe7d2,stroke:#a95a14,stroke-width:1.5px,color:#1d2330;
  classDef mDec fill:#d4efed,stroke:#1f7a74,stroke-width:1.5px,color:#1d2330;
  classDef cDec fill:#ececec,stroke:#5f6368,stroke-width:1.5px,color:#1d2330;
  classDef ok fill:#d6efdc,stroke:#2e7d46,stroke-width:1.5px,color:#1d2330;
  classDef stop fill:#f8dada,stroke:#b3261e,stroke-width:1.5px,color:#1d2330;"""

# Every change that reaches beyond the branch's features, first in the document (the review convention's item 0):
# (what changed, who it affects, risk). Filled from the whole branch diff against the merge-base.
SYSTEM_WIDE: list[tuple[str, str, str]] = [
    (
        "<b>Every single arm move goes through the planner</b>: any <code>set_state_with_arm_joints(clear_queue=True)</code> "
        "(joints, home, fold, map_pose, the scan, a replay's first state) calls <code>plan_move</code> first; "
        "<code>pose</code> uses <code>plan_move</code> instead of <code>pyroki_node.solve</code> (chart 2)",
        "Every arm command on every robot",
        "<b>High.</b> <code>plan_move</code> waits with <b>no timeout</b> (<code>messages/pyroki_world.py:527</code>): a "
        "planner that is down or hung stops every arm move, home and fold included. Moves can now be refused or "
        "rerouted where main moved straight; a pose no IK reaches is refused (main moved to the best effort). "
        "<b>Regression, fixed:</b> a joints goal outside J3's range drove billie-07's arm into itself (2026-10-20); "
        "the joint limits and the self check now refuse it (37b8078f)",
    ),
    (
        "<b>Every planner request waits for spatial_awareness first</b> (<code>surroundings.refresh()</code>, "
        "<code>get_arm_surroundings</code>, 4 s timeout), which asks the mapper and answers on its next tick (chart 3)",
        "AI steps, batch solves, replays, every arm move",
        "<b>High.</b> Up to one 100 ms tick plus a mapper round trip per solve; up to 2 s when the mapper is busy "
        "(publish, save). With the pose not aligned, a solve within 1 m of the room is refused; one missed answer "
        "marks the pose not aligned. Delay not yet measured on an Orin",
    ),
    (
        "<b>IK results are post-processed and self-collision is always on</b>: <code>keep_solution_clear()</code> runs "
        "on every single and batch solve; <code>ENABLE_PYROKI_SELF_COLLISION</code> is no longer read (chart 6, chart 9)",
        "AI and replay_policy motion",
        "<b>Medium-high.</b> AI and replay frames can be shortened (only an info log and "
        "<code>is_obstacle_adjusted</code>). Robots that ran without self-collision get different solutions and more compute",
    ),
    (
        "<b>replay_policy always asks the planner</b> (<code>plan_replay</code>, no timeout) and raises "
        "<i>Replay blocked</i> when no route is found (chart 7); replays and AI tasks run inside <code>arm_touch.touching()</code>",
        "Every replay_policy run",
        "<b>Medium-high.</b> A planner hang blocks replays; a recording can be changed (routes spliced in, brain "
        "commands moved onto route frames) or refused",
    ),
    (
        "<b>Default TSDF mapping changed</b>: <code>NVBLOX_VOXEL_SIZE_M</code> 0.01 → 0.05, <code>INTEGRATE_MAX_DEPTH_M</code> "
        "4 → 7 m, the camera's depth cap for mapping 4000 → 7000 mm (reconfigured on every start and stop); frames are "
        "dropped unless the arm has been still (under 2 °/s for 0.75 s); resuming a room map without "
        "<code>map.nvblx</code> raises",
        "All 3D mapping and published room maps",
        "<b>Medium-high.</b> Published maps drop from 1 cm to 5 cm. Fewer frames while the arm moves. Resume fails on room "
        "maps published before this branch",
    ),
    (
        "<b>Mapper defaults that change every live map</b>: the gripper mask is now <code>billie-gripper-v1</code> grown 15 % "
        "(main: the legacy box); the arm-motion gate is on and <b>fails closed</b>: a frame without a v2 synchronization "
        "record, or with a telemetry gap over 0.25 s, is dropped; the bridge drops a frame unless both arm clocks bracket "
        "it (chart 15)",
        "All mapping",
        "<b>Medium.</b> Fewer frames integrated (more so on robots whose arm streams are slow or gappy); a bigger region "
        "around the gripper is never mapped",
    ),
    (
        "<b>In occupancy mode the mapper's live joint clearance is off</b> (no ESDF: <code>write_joint_clearance()</code> "
        "returns none) (chart 17)",
        "Whatever reads the mapper's arm-clearance lane",
        "<b>Medium.</b> That safety lane is absent while occupancy mapping runs; the arm planner's own room (chart 3) is the "
        "only arm check then",
    ),
    (
        "<b>Camera recording and reconfigure</b>: <code>start_recording</code> raises when already recording (main "
        "stopped the old one); the mapper is a camera client and owns SVO recording during capture; "
        "<code>reconfigure</code> validates <code>0 &lt; min &lt; max</code> before its no-change check; rgbd metadata gains "
        "<code>svo_frame_index</code> (may be <code>None</code>)",
        "Episode recording, mapping start and stop",
        "<b>Medium.</b> A leaked recorder blocks every later recording until the camera restarts; episode recording "
        "and a mapping capture exclude each other. A robot with <code>ZED_DEPTH_MIN_MM</code> ≤ 0 cannot start or stop "
        "mapping (fleet values not in the repo)",
    ),
    (
        "<b>Camera bridge calibration and placement</b>: <code>$BILLIE_DIR/vision/camera_calibration.json</code> now wins "
        "over <code>T_GRIPPER_FROM_CAMERA</code>; a non-rigid matrix crashes the bridge at startup; frames not bracketed "
        "by arm samples are dropped (were clamped); the arm base height comes from <code>arm_base_height_m()</code>, "
        "with <b>BILLIE-07 = 0.49 m</b> hard-coded over its setting (chart 13)",
        "Every map frame and robot_pose",
        "<b>Medium.</b> Robots whose file and env calibration differ place map frames differently; billie-07's map moves "
        "~5 cm in z; more frames skipped",
    ),
    (
        "<b>Shared arm-move guard</b>: <code>xarm_writer.enqueue_joints</code> → <code>check_move()</code>, "
        "<code>player.play_joints_batch</code> → <code>check_joint_sequence()</code>; importing "
        "<code>messages.xarm_writer</code> pulls in <code>pyroki_world</code> and <code>arm_collision_model</code> (chart 8)",
        "Every process that queues joints, the xarm_writer node included",
        "<b>Low-medium.</b> Refuses moves only while objects or the floor are set. A malformed "
        "<code>AUTOXING_BASE_TO_ARM_BASE_Z_M</code> now fails xarm_writer and the brain at import. <b>Regression, fixed:</b> "
        "unknown joints would have blocked every checked move; the joints reader is now registered at brain startup (1221dfec)",
    ),
    (
        "<b>spatial_awareness always-on work</b>: lidar-to-map alignment on every scan, the standing walls, the lidar grid, "
        "a 1 s re-check of the map file, and the arm ESDF export (a 118³ field built and gzipped every second); new S3 "
        "lanes <code>esdf</code> (1 s) and <code>occupancy</code> (2 s) (chart 10, chart 11)",
        "Orin CPU on every robot; the maps bucket",
        "<b>Medium.</b> New threads and periodic CPU work, per-second S3 PUTs during occupancy mapping. CPU cost not "
        "measured on an Orin",
    ),
    (
        "<b>New dataflow wiring</b>: the mapper gets <code>servo_angles_deg</code> (~125 Hz, queue 4096), "
        "<code>tracked_pose</code>, <code>map/info</code>; the bridge gets the servo and TCP streams; new clients camera → "
        "mapper, spatial_awareness → planner, mapper → spatial_awareness; nav_events outputs <code>map/info</code>",
        "mapper, bridge, camera, spatial_awareness",
        "<b>Low-medium.</b> Two more nodes process the 125 Hz arm streams even when not mapping",
    ),
    (
        "<b><code>publish_map</code> timeout 300 → 3600 s</b>; with a replay capture, publish uploads a large "
        "<code>capture.zip</code> synchronously",
        "The brain; anything asking the mapper meanwhile",
        "<b>Medium.</b> The brain can wait up to an hour; arm surroundings fall back to the map file after 2 s each meanwhile",
    ),
    (
        "<b>AI tasks pause and resume the 3D mapper</b> (<code>mapper_paused()</code>, 2 s timeout) (chart 6)",
        "Every AI task",
        "<b>Low-medium.</b> Up to 2 s more at each end with a slow mapper; no map frames during AI tasks",
    ),
    (
        "<b>Mapper dependencies and build</b>: <code>transformers&gt;=5.8</code>, <code>torchvision</code>, <code>mcap</code>, "
        "<code>zstandard</code>, <code>opencv-python</code>, <code>pillow</code>, editable <code>camera-autoxing-bridge</code>; a "
        "patched nvBlox wheel (LFS, occupancy API) that <code>post-install.sh</code> verifies; 5 nvBlox patches",
        "The release image, the mapper venv",
        "<b>Medium.</b> Bigger, slower builds; the build fails without <code>git lfs pull</code>; the mapper runs a patched nvBlox",
    ),
    (
        "<b>Planner dependencies and solver cache</b>: <code>flatbuffers</code>, <code>absl-py</code>; exported solvers saved "
        "under <code>$BILLIE_DIR/pyroki-jax-cache/exported</code> with no cleanup",
        "pyroki-planner",
        "<b>Low.</b> A robot needs <code>./install.sh nodes/pyroki-planner</code> (billie-07 failed without absl); disk use "
        "grows with each code change",
    ),
    (
        "<b>New and changed commands</b>: <code>set_obstacles</code> (not hidden), hidden <code>start_active_mapping</code> and "
        "<code>load_saved_mapping</code>; the mapping card gains 5 buttons; occupancy maps cannot be published",
        "Operators",
        "<b>Low.</b> What operators see",
    ),
    (
        "<b>nav_events</b> subscribes to <code>/map/info</code>, strips a leading <code>/</code> from every topic, flattens "
        "the <code>map/info</code> payload",
        "nav_events on every robot",
        "<b>Low.</b> An empty topic would raise in <code>on_tick</code>",
    ),
    (
        "<b>Cloud status</b> gains <code>representation</code> and an <code>evaluation_capture</code> block",
        "Cloud and web consumers",
        "<b>Low.</b> Additive",
    ),
    (
        "<b>Uploader</b>: <code>.destination</code> sidecars with a bucket allowlist; <code>$file</code> quoted",
        "The uploader containers",
        "<b>Low.</b> Nothing writes <code>.destination</code> files yet; the containers need reinstalling",
    ),
    (
        "<b>brain <code>node.py</code></b>: <code>send_cloud_message</code> and <code>upload_for_debugging</code> take an "
        "<code>activity_id</code>, which skips the silent-command check",
        "Brain cloud messages",
        "<b>Low.</b> Only the debug plot worker uses it",
    ),
    (
        "<b>New repo-level files</b>: <code>MAPPING_BRANCH.md</code>, <code>NVBLOX_OFFLINE_REPLAY_QUALITY_PLAN.md</code>, "
        "<code>SHARED_MAPPING_QUALITY.md</code> at the root; <code>experiments/texture-mapping-20261012/</code> (an 8,401-line "
        "patch, a 627 KB <code>base.bundle</code> not in LFS); a <code>billie/mapping-core</code> package with its own CI "
        "workflow",
        "The repo and CI",
        "<b>Low.</b> Repo size; mapping-core ties <code>billie_utils.atomic_write</code> and <code>matrix</code> to its CI",
    ),
]

# What each chart's flow does differently from main, shown under its "When" line.
SUMMARIES: dict[int, str] = {
    0: "On main the brain sent arm moves to the xArm with no world check: a joints move as it was, a pose solved "
    "by the pyroki IK alone. Now every arm command is planned against what is around the arm: the room, built "
    "by spatial_awareness at 2.5 cm from the mapper's live occupancy, the lidar and AutoXing's walls before every "
    "plan; objects set with <code>set_obstacles</code>; the floor; and the arm itself. A route goes around what is "
    "in the way, and a move with no clear route is refused. The field is also exported to the cloud once a "
    "second, the mapper pauses during AI tasks, and a new button starts mapping with a short arm scan.",
    1: "New files: <code>spatial_awareness/occupancy_field.py</code>, <code>arm_surroundings.py</code>, "
    "<code>planner_answers.py</code>, <code>arm_esdf_export.py</code>; in the planner <code>esdf_world.py</code>, "
    "<code>arm_spheres.py</code>, <code>surroundings.py</code>, <code>plan_single.py</code>, <code>plan_batch.py</code>, "
    "<code>replay_planner.py</code>, <code>self_check.py</code>; in the mapper <code>occupancy_requests.py</code>; in "
    "the brain <code>active_mapping.py</code> and <code>mapper_pause.py</code>; the messages in "
    "<code>billie_utils/messages/arm_surroundings.py</code> and <code>mapper.py</code>.",
    2: "On main a joints move went straight to the xArm writer and a pose was solved by <code>pyroki_node.solve</code>, "
    "with no world check. Now every single move (joints and pose) asks the new <code>plan_move</code> request, which "
    "refreshes the room first (chart 3). The planner refuses a goal outside the joint limits, checks the straight "
    "line against the room, the objects, the floor and the arm itself (pyroki's self-collision model), tries a "
    "transit route when it is blocked, and refuses when the route is blocked too. A link that starts too close may "
    "move, but not get deeper. Route frames are driven at a speed that keeps the tool under the controller's limit.",
    3: "New. Before every plan the planner asks spatial_awareness for its surroundings. spatial_awareness asks the "
    "mapper for the occupancy around the arm without blocking its own event loop, builds the crop on a worker thread "
    "and answers on the next tick. Without a mapper answer in 2 s it answers from the saved map file; without a "
    "pose, with the arm-frame objects only. When spatial_awareness does not answer, the planner keeps the last "
    "surroundings and treats the pose as not aligned.",
    4: "New. The arm's field is built from scratch for each new crop (0.2 s budget, measured 0.19 s on an idle Orin). "
    "Floating specks are dropped to unknown, holes filled, the lidar fills the low band the camera never saw, "
    "AutoXing's walls fill unknown space up to 2 m, Billie's base is cleared (the arm is not), and each cell gets its "
    "distance to the nearest occupied cell's cube, capped at 20 cm when the pose is aligned.",
    5: "New request on the mapper. It copies only the asked area's 2.5 cm blocks off the GPU, and answers "
    "<i>unchanged</i> when the asker already holds the same map epoch and frame count.",
    6: "On main the AI's frames were solved by the IK with no world check. Now each solve plans against the room "
    "(chart 3), the objects and the floor, the fingers may touch (8 cm) and the gripper body a little (2 cm), each "
    "frame's move is shortened to its farthest clear point (the arm against itself included), and the 3D mapper "
    "stops taking new frames for the whole task.",
    7: "On main a recording was replayed as recorded. Now every replay with recorded joints asks the new "
    "<code>plan_replay</code> request, which checks the recording against the room, the objects, the floor and the "
    "arm itself, routes around each blocked stretch, and refuses the replay when a route is blocked too.",
    8: "New. <code>set_obstacles</code> hands objects to spatial_awareness, which keeps map-frame objects in the map "
    "and places them at every plan, so the robot may drive afterwards. Every joint move the brain queues is checked "
    "by the new arm guard against the active objects and the floor; it reads the arm's joints from a reader "
    "registered once at the brain's startup.",
    9: "On main self-collision was a setting (<code>ENABLE_PYROKI_SELF_COLLISION</code>) and the solvers had no world "
    "costs. Now every solver always includes self-collision and adds the room's distance field and 8 capsule slots "
    "(fixed shapes, so new obstacles never recompile); a transit solver plans routes; and the same pyroki model "
    "checks the planned paths no solver runs on (chart 2). The room grid is 2.5 cm.",
    10: "New. Once a second, while the pose is known, spatial_awareness builds the same field for the 1.5 m around "
    "the arm and writes it as <code>arm_esdf.bin.gz</code>; the mapper's S3 mirror uploads it on a new "
    "<code>esdf</code> lane, which the cloud's Map 3D shows as \"ESDF (arm)\" (cloud PR #521).",
    11: "New. spatial_awareness keeps a 2.5 cm lidar grid around Billie (hits and misses at the lidar's height) and "
    "the rays that pass through AutoXing's mapped walls, both used when building the field (chart 4).",
    12: "New command and button. Starts occupancy mapping from zero, then turns the camera through 6 views at half the "
    "arm speed, each a planned move (chart 2), waiting at each for the mapper to add frames, and ends at the "
    "mapping pose.",
    13: "The arm base's height above the floor now comes from one place for the camera bridge (the map) and "
    "spatial_awareness (the arm in it): a per-robot fix, else the robot's setting, else the fleet default. The "
    "mapper writes the robot's state next to its live files, for the viewers.",
    14: "On main <code>start_mapping</code> moved the arm, capped the ZED at 4 m only when the env differed, and started one "
    "TSDF mapper at 1 cm with a 5 cm coarse preview. Now the mapper can start in <b>occupancy</b> mode (a native nvBlox "
    "occupancy layer, no TSDF / ESDF / mesh, recreated per session); TSDF defaults to 5 cm with no preview; the ZED "
    "goes to 7 m; and a start can open a <b>replay capture</b> (an MCAP writer, plus a camera SVO2 recording the mapper "
    "asks for). The brain checks the representation and capture the mapper reports, and rolls back on a mismatch.",
    15: "On main the bridge sent a frame from one or two arm reads (clamped interpolation), and the mapper integrated it "
    "after the pose-jump and keyframe gates. Now the bridge needs a true timestamp bracket on both arm clocks from "
    "continuous joint and TCP streams, and attaches a v2 synchronization record and the camera settings to every "
    "frame. The mapper adds a fail-closed arm-motion gate, an optional semantic mask, per-frame capture with a "
    "decision log, an occupancy integration path, and a larger default gripper exclusion.",
    16: "New. The mapper subscribes to AutoXing's raw pose, its map identity and the full-rate arm joints, keeps the "
    "latest of each, and records them into the replay MCAP while a capture runs. The capture gate needs every stream "
    "written at least once.",
    17: "On main the tick refreshed the TSDF mesh and mirrored the GLBs and the trajectory. In occupancy mode it exports a "
    "compressed snapshot of the occupancy blocks every 2 s or more instead, mirrored on a new S3 lane; the cloud status "
    "gains the representation and capture health; joint clearance is off while the map is occupancy.",
    18: "On main save flushed the GLB and the baked ESDF, and finalize released the CUDA backends. Now TSDF saves also "
    "refresh the native ESDF and keep <code>map.nvblx</code> for a later resume; occupancy saves export the snapshot and "
    "the native map and keep the GPU map; finalize stops the camera's SVO recording and closes the replay capture with "
    "an integrity gate, tied to the saved map's manifest.",
    19: "On main publish zipped the ESDF and mesh into <code>latest/operate.zip</code>, rotating the old one. Now the bundle "
    "also carries the native <code>.nvblx</code> files; a saved map with a replay capture has it validated and uploaded "
    "first; an immutable <code>captures/&lt;id&gt;/operate.zip</code> is written and <code>latest</code> copied from it; the "
    "brain's timeout is 1 hour.",
    20: "On main only published room maps could be loaded, and resuming one mapped into an empty backend. Now a saved "
    "occupancy map loads back into GPU memory after validation against its native SQLite (integration paused); room maps "
    "carry and install their native <code>.nvblx</code>, so a resume continues the real map; room maps without it refuse "
    "to resume; loading a room map always switches back to TSDF.",
    21: "New. Offline tools: a capture inspector, a validated TSDF replay (SVO2 depth recomputed, arm poses re-synchronized, "
    "semantic or mirror masks), an experimental occupancy replay and validation scripts. The mapper's startup wires the "
    "capture and the camera SVO callbacks; the image build verifies a vendored nvBlox wheel rebuilt with an occupancy-API patch.",
}

# (number, anchor, title, when, mermaid body, notes)
CHARTS: list[tuple[int, str, str, str, str, list[str]]] = [
    (
        0,
        "overview",
        "Overview",
        "Which arm command reaches which node, and where the room around the arm comes from.",
        """flowchart TD
  subgraph TOP[" "]
    J(["joints · pose"]):::trig
    AI(["AI task"]):::trig
    RP(["replay_policy"]):::trig
    SO(["set_obstacles"]):::trig
    AM(["Start Active Occupancy Mapping"]):::trig
  end
  J --> B2["brain: every single move is planned · chart 2"]:::bChg
  AM --> B12["brain: the initial scan · chart 12"]:::bNew
  B12 --> B2
  AI --> B6["brain: AI frames, the mapper paused · chart 6"]:::bChg
  RP --> B7["brain: the recording is planned · chart 7"]:::bChg
  B2 == "plan_move" ==> PL["pyroki-planner · charts 2 and 9"]:::pChg
  B6 == "plan · batch_plan" ==> PL
  B7 == "plan_replay" ==> PL
  PL == "get_arm_surroundings · chart 3" ==> SA["spatial_awareness: the arm's field · chart 4"]:::sNew
  SO == "set_obstacles · chart 8" ==> SA
  SA == "get_occupancy_blocks · chart 5" ==> MP["nvblox_mapper: the live 2.5 cm occupancy<br/>the mapper's own flows from chart 14"]:::mChg
  SA -- "once a second · chart 10" --> EX["arm_esdf.bin.gz → S3 → cloud"]:::sNew
  B2 --> G["xarm_writer + the arm guard · chart 8"]:::uChg
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        1,
        "files",
        "Files",
        "Which file holds what, in the arm-awareness part of the branch.",
        """flowchart TD
  subgraph BR["brain"]
    BS["<b>behaviours/state.py</b>: every single move planned<br/><b>play/replay_reroute.py</b>: every replay planned"]:::bChg
    BA["<b>behaviours/ai.py</b> + <b>mapper_pause.py</b>: touch, the mapper paused<br/><b>active_mapping.py</b> + <b>commands/mapping_3d.py</b>: the scan"]:::bNew
  end
  subgraph PLN["pyroki-planner"]
    PS["<b>plan_single.py</b> · <b>plan_batch.py</b> · <b>replay_planner.py</b><br/>the checks and the routes"]:::pNew
    PE["<b>esdf_world.py</b> · <b>arm_spheres.py</b> · <b>self_check.py</b><br/><b>surroundings.py</b>: the room, the arm, itself"]:::pNew
    PR["<b>resolver.py</b> · <b>resolver_single.py</b> · <b>resolver_batch.py</b><br/>the solvers"]:::pChg
  end
  subgraph SAW["spatial_awareness"]
    SF["<b>occupancy_field.py</b>: the field<br/><b>arm_surroundings.py</b> · <b>planner_answers.py</b>"]:::sNew
    SE["<b>arm_esdf_export.py</b>: the cloud file"]:::sNew
  end
  U["<b>billie_utils/messages/arm_surroundings.py</b> · <b>mapper.py</b> · <b>pyroki_world.py</b><br/><b>arm_motion_guard.py</b> · <b>esdf_grid.py</b>"]:::uChg
  M["<b>nvblox_mapper/occupancy_requests.py</b><br/>the area's blocks off the GPU"]:::mNew
  BS --> U
  BA --> U
  U ==> PS
  PS --> PE
  PS --> PR
  PE ==> SF
  SF ==> M
  SE ==> M
  style BR fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
  style PLN fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
  style SAW fill:#fdf4ea,stroke:#a95a14,color:#a95a14
""",
        [],
    ),
    (
        2,
        "single_move",
        "A joints or pose move",
        "A <code>joints</code> or <code>pose</code> command, or any single arm move the brain makes "
        "(<code>clear_queue</code>), e.g. a scan view of chart 12.",
        """flowchart TD
  subgraph TOP[" "]
    J(["joints command · a single move"]):::trig
    P(["pose command"]):::trig
  end
  J --> SJ["<b>set_state_with_arm_joints()</b><br/>behaviours/state.py"]:::bChg
  P --> SP["<b>set_state_with_arm_pose()</b><br/>behaviours/state.py"]:::bChg
  SJ --> CQ{{"clear_queue: a single move?"}}:::bDec
  CQ -- "no: a frame of a trajectory" --> EQ
  CQ -- "yes: always planned" --> PM["<b>_plan_single_move()</b><br/>behaviours/state.py · joints unknown → refused"]:::bNew
  SP --> PM
  PM == "plan_move · «data» PlanMoveRequest" ==> OPM["<b>on_plan_move()</b><br/>plan_single.py · <b>surroundings.refresh()</b> first · chart 3"]:::pNew
  OPM --> PLM["<b>plan_move()</b><br/>plan_single.py"]:::pNew
  PLM --> GP{{"a pose goal?"}}:::pDec
  GP -- "yes" --> SPG["<b>solve_pose_goal()</b><br/>inside the base / no IK reaches it → refused"]:::pNew
  GP -- "no" --> PJM
  SPG --> PJM["<b>plan_joint_move()</b><br/>plan_single.py"]:::pNew
  PJM --> LB{{"goal outside the joint limits?<br/><b>limit_breach()</b> · self_check.py"}}:::pDec
  LB -- "yes" --> R1["MOVE_BLOCKED: the arm does not move"]:::stop
  LB -- "no" --> CP["<b>check_path()</b> on the straight line<br/>room · objects · floor: <b>rule_clearances_np()</b><br/>the arm itself: <b>path_past()</b>"]:::pNew
  CP -- "clear" --> D1["MOVE_DIRECT"]:::ok
  CP -- "blocked" --> RT["<b>IKResolver.route()</b> transit route<br/>checked again by <b>check_path()</b>"]:::pNew
  RT -- "clear" --> D2["MOVE_DETOUR: the route's frames"]:::ok
  RT -- "blocked" --> R2["MOVE_BLOCKED: names the link and what it hits"]:::stop
  D1 --> UR{{"pose not aligned and the path near anything?<br/><b>unaligned_refusal()</b> · surroundings.py"}}:::pDec
  D2 --> UR
  UR -- "yes" --> R3["MOVE_BLOCKED"]:::stop
  UR -- "no" --> UP["<b>unknown_passage()</b>: logs a path through never-observed cells"]:::pNew
  UP == "reply · «data» PlanMoveResponse" ==> EQ["<b>enqueue_joints()</b> per frame at <b>_route_speeds()</b><br/>xarm_writer.py → <b>check_move()</b> · chart 8"]:::bChg
  style TOP fill:none,stroke:none
""",
        [
            "<code>check_path()</code> applies the start rule: a link (or a self pair) already closer than allowed at "
            "the start may stay as close, never closer (<code>from_start()</code>, <code>start_breach()</code>).",
            "Self pairs: gripper~link4 is left out and link2~link4 may overlap 2.5 cm; measured on 8,033 real recorded "
            "configurations, where the coarse wrist capsules overlap without contact (<code>self_check.py</code>).",
        ],
    ),
    (
        3,
        "surroundings",
        "The planner's room before each plan",
        "Any plan request (chart 2, chart 6, chart 7) calls <code>surroundings.refresh()</code> first.",
        """flowchart TD
  T(["plan · batch_plan · plan_move · plan_replay"]):::trig
  T --> RF["<b>surroundings.refresh()</b><br/>pyroki-planner/surroundings.py"]:::pNew
  RF == "get_arm_surroundings · «data» GetArmSurroundingsRequest" ==> OG["<b>on_get_arm_surroundings()</b><br/>spatial_awareness/main.py"]:::sNew
  OG --> ST["<b>PlannerAnswers.start()</b><br/>planner_answers.py"]:::sNew
  ST --> OQ{{"a robot pose?<br/><b>ArmSurroundings.occupancy_query()</b>"}}:::sDec
  OQ -- "no" --> NP["answered at once: arm-frame objects and floor, no crop"]:::stop
  OQ -- "yes" --> SEND["<b>request_occupancy_blocks()</b> · no waiting<br/>the held map version sent along"]:::sNew
  SEND == "get_occupancy_blocks · chart 5" ==> MP["nvblox_mapper"]:::mNew
  MP == "reply · «data» OccupancyBlocksResponse" ==> ONR["<b>PlannerAnswers._on_reply()</b><br/>build on a worker: <b>PlannerAnswers._build()</b>"]:::sNew
  ST -- "no reply in 2 s: <b>PlannerAnswers.tick()</b>" --> ONR
  ONR --> FB{{"<b>ArmSurroundings._fetch_blocks()</b><br/>occupancy for this area?"}}:::sDec
  FB -- "unchanged · Billie moved < CROP_REUSE_M" --> RE["the last crop"]:::ok
  FB -- "new blocks" --> BF["<b>build_field()</b> · chart 4"]:::sNew
  FB -- "no reply · not mapping occupancy" --> FILE["<b>ArmSurroundings._current_crop()</b><br/>from the saved map file"]:::sNew
  RE --> TK["<b>PlannerAnswers.tick()</b>: the answer goes out on the next tick"]:::sNew
  BF --> TK
  FILE --> TK
  TK == "reply · «data» ArmSurroundingsResponse" ==> AP["<b>surroundings.apply()</b><br/><b>esdf_world.set_room()</b> · <b>world_collision.set_obstacles()</b><br/>aligned or not"]:::pNew
  RF -- "no answer · error" --> KEEP["the last surroundings kept, pose = not aligned"]:::stop
""",
        [
            "The answer is never sent from inside the mapper's reply handler: <code>BillieNode.wait_for_event</code> "
            "drops other events while it waits, so spatial_awareness never blocks its event loop "
            "(<code>planner_answers.py</code>).",
        ],
    ),
    (
        4,
        "field",
        "The arm's field from the occupancy",
        "A new crop is needed (chart 3) or the cloud export is due (chart 10).",
        """flowchart TD
  T(["a new crop: chart 3 · chart 10"]):::trig
  T --> BF["<b>build_field()</b><br/>occupancy_field.py"]:::sNew
  BF --> AS["<b>area_states()</b>: the mapper's 2.5 cm blocks<br/>occupied · free · unknown"]:::sNew
  AS --> DF["<b>drop_floating()</b>: specks of ≤ 3 cells with nothing<br/>within 5 cm → unknown"]:::sNew
  DF --> FH["<b>fill_holes()</b>: C2 holes in surfaces"]:::sNew
  FH --> LI{{"a lidar snapshot? · chart 11"}}:::sDec
  LI -- "yes" --> AL["<b>add_lidar()</b>: never-observed cells only<br/>returns occupied floor → 29.5 cm · rays free at its height"]:::sNew
  LI -- "no" --> WA
  AL --> WA{{"AutoXing walls and the lidar correction?"}}:::sDec
  WA -- "yes" --> AW["<b>add_walls()</b>: up to 2 m in unknown cells<br/>unless the lidar saw through"]:::sNew
  WA -- "no" --> CB
  AW --> CB["<b>clear_base()</b>: Billie's base, not the arm"]:::sNew
  CB --> BD["<b>box_distances()</b>: the exact box rule<br/>capped at 20 cm · a safe uncapped bound when not aligned"]:::sNew
  BD --> OUT["«data» Field: the crop's distances, unknown cells, a note"]:::ok
""",
        [],
    ),
    (
        5,
        "occupancy_blocks",
        "The mapper answers get_occupancy_blocks",
        "spatial_awareness asks for the occupancy around the arm (chart 3, chart 10).",
        """flowchart TD
  T(["get_occupancy_blocks · «data» GetOccupancyBlocksRequest"]):::trig
  T --> BH["<b>blocks_handler()</b><br/>occupancy_requests.py · registered in <b>MapperRuntime.main()</b>"]:::mNew
  BH --> AN["<b>answer()</b><br/>under the backend's lock"]:::mNew
  AN --> OM{{"mapping occupancy, backend ready?"}}:::mDec
  OM -- "no" --> ER["error: not mapping occupancy"]:::stop
  OM -- "yes" --> SV{{"the asker holds this map epoch and frame count?"}}:::mDec
  SV -- "yes" --> UN["unchanged: no blocks sent"]:::ok
  SV -- "no" --> AB["<b>OccupancyBackend.area_blocks()</b><br/>only the area's blocks, off the GPU"]:::mNew
  AB --> EV["<b>encode_values()</b>: log-odds × 10 as int8"]:::mNew
  EV --> RE["«data» OccupancyBlocksResponse: blocks and values as bytes"]:::ok
""",
        [],
    ),
    (
        6,
        "ai",
        "An AI task",
        "An AI task (<code>ai_task</code>) runs its policy.",
        """flowchart TD
  T(["AI task"]):::trig
  T --> EP["<b>execute_policy()</b><br/>behaviours/ai.py"]:::bChg
  EP --> TC["<b>arm_touch.touching()</b>: fingers 8 cm, gripper body 2 cm"]:::uNew
  EP --> MPA["<b>mapper_paused()</b><br/>behaviours/mapper_pause.py"]:::bNew
  MPA == "pause_mapping · then resume_mapping in finally" ==> PH["<b>MappingSession.handle_pause_mapping()</b><br/>brain_paused: new frames are skipped, the map kept"]:::mNew
  MPA -- "the mapper is down" --> WN["a warning; the task goes on"]:::stop
  EP == "plan · batch_plan, per step" ==> OP["<b>on_plan()</b> · <b>on_batch_plan()</b><br/>pyroki-planner/main.py · <b>surroundings.refresh()</b> · chart 3"]:::pChg
  OP --> RS["IK solve with the room, the objects and self-collision"]:::pChg
  RS --> KS["<b>keep_solution_clear()</b> → <b>keep_moves_clear()</b><br/>plan_batch.py"]:::pNew
  KS --> CL{{"<b>_clear()</b>: the move keeps clear?<br/>room · objects · floor + the arm itself"}}:::pDec
  CL -- "yes" --> FR["the frame as solved"]:::ok
  CL -- "no" --> SH["shortened to its farthest clear point (halving)"]:::ok
  FR --> UA["<b>unaligned_refusal()</b> · <b>unknown_passage()</b>"]:::pNew
  SH --> UA
""",
        [],
    ),
    (
        7,
        "replay",
        "replay_policy",
        "A recording is replayed (<code>replay_policy</code>).",
        """flowchart TD
  T(["replay_policy"]):::trig
  T --> RR["<b>reroute_around_obstacles()</b><br/>play/replay_reroute.py"]:::bNew
  RR --> HJ{{"every frame has recorded joints?"}}:::bDec
  HJ -- "no" --> AS["played as recorded"]:::ok
  HJ -- "yes: always planned" --> PR["<b>plan_replay()</b><br/>messages/pyroki_world.py · touch passed along"]:::uNew
  PR == "plan_replay · «data» PlanReplayRequest" ==> OPR["<b>on_plan_replay()</b><br/>replay_planner.py · <b>surroundings.refresh()</b> · chart 3"]:::pNew
  OPR --> RER["<b>reroute_recording()</b><br/>replay_planner.py"]:::pNew
  RER --> CLR["<b>_clearances()</b> → <b>_distances()</b><br/>room · objects · floor + the arm itself, start rule"]:::pNew
  CLR --> STR{{"<b>_stretches()</b>: blocked stretches?"}}:::pDec
  STR -- "none" --> UN["REPLAY_UNCHANGED"]:::ok
  STR -- "some" --> RO["per stretch: <b>IKResolver.route()</b> + <b>check_path()</b>"]:::pNew
  RO -- "all clear" --> RTD["REPLAY_REROUTED: routes spliced in"]:::ok
  RO -- "a route blocked" --> BL["REPLAY_BLOCKED → ArmMoveBlockedError"]:::stop
""",
        [],
    ),
    (
        8,
        "obstacles",
        "set_obstacles and the arm guard",
        "The <code>set_obstacles</code> command; and every joint move the brain queues on the xArm writer.",
        """flowchart TD
  subgraph TOP[" "]
    T(["set_obstacles command"]):::trig
    Q(["any queued arm move"]):::trig
  end
  T --> SO["<b>world_obstacles.set_obstacles()</b><br/>commands/world_obstacles.py · a rect → <b>_with_rect()</b>"]:::bNew
  SO == "set_obstacles · «data» SetObstaclesRequest" ==> SSO["<b>on_set_obstacles()</b> → <b>ArmSurroundings.set_obstacles()</b><br/>spatial_awareness · capsules, ≤ 8 slots"]:::sNew
  SSO -- "too many capsules · bad spec" --> ER["refused, the old objects stay"]:::stop
  SSO == "reply · «data» SetObstaclesResponse" ==> RM["<b>remember_obstacles()</b><br/>for the arm guard and the plots"]:::uNew
  RM --> NX["placed at the pose of every later plan · chart 3"]:::ok
  Q --> EJ["<b>enqueue_joints()</b><br/>messages/xarm_writer.py"]:::uChg
  EJ --> CM["<b>check_move()</b><br/>arm_motion_guard.py"]:::uNew
  CM --> AO{{"objects or floor active? <b>active_obstacles()</b>"}}:::bDec
  AO -- "no" --> SEND["sent"]:::ok
  AO -- "yes" --> CJ["<b>current_joints()</b>: the reader registered in brain <b>main()</b><br/><b>ActiveObstacles.check()</b>"]:::uNew
  CJ -- "clear" --> SEND
  CJ -- "blocked · joints unknown" --> BL["ArmMoveBlockedError"]:::stop
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        9,
        "planner_start",
        "The planner starts",
        "The pyroki-planner node starts (or rebuilds its solvers).",
        """flowchart TD
  T(["pyroki-planner node starts"]):::trig
  T --> RB["<b>IKResolver._run_build()</b><br/>resolver.py"]:::pOld
  RB --> LAM["<b>load_arm_models()</b>: URDF, RobotCollision<br/>(gripper~link5 left out) · <b>esdf_world.load_from_environment()</b><br/>2.5 cm room grid (PYROKI_ROOM_VOXEL_M)"]:::pNew
  LAM --> BS["<b>_build_solver()</b>: single · batch · transit solvers<br/>self-collision always · the room's field · 8 capsule slots"]:::pChg
  BS --> AR["<b>IKResolver._assign_solver_result()</b>"]:::pChg
  AR --> SC["<b>self_check.configure()</b>: compiled self distances,<br/>the joint limits · used by chart 2, chart 6, chart 7"]:::pNew
""",
        [],
    ),
    (
        10,
        "esdf_export",
        "The arm's field to the cloud",
        "spatial_awareness's tick, at most once a second, while the robot's pose is known.",
        """flowchart TD
  T(["spatial_awareness tick"]):::trig
  T --> DUE{{"<b>ArmEsdfExporter.due()</b>: not building,<br/>interval passed, no reply pending"}}:::sDec
  DUE -- "no" --> W["nothing this tick"]:::stop
  DUE -- "yes" --> TK["<b>ArmEsdfExporter.tick()</b><br/>arm_esdf_export.py · the area around the arm base"]:::sNew
  TK == "get_occupancy_blocks · no waiting · chart 5" ==> MP["nvblox_mapper"]:::mNew
  MP == "reply" ==> OR["<b>ArmEsdfExporter._on_reply()</b>"]:::sNew
  OR --> CH{{"map unchanged and the arm moved < CROP_REUSE_M?"}}:::sDec
  CH -- "yes" --> SK["no new file"]:::stop
  CH -- "no" --> BLD["<b>ArmEsdfExporter._build()</b> on a worker<br/><b>build_field()</b> · chart 4 → <b>encode_field()</b> → gzip<br/><b>write_atomically()</b> arm_esdf.bin.gz"]:::sNew
  BLD --> S3["the mapper's S3 mirror: lane esdf<br/><b>live_map_mirror_config()</b> · nvblox_mapper/main.py"]:::mNew
  S3 --> CL["cloud Map 3D: ESDF (arm)"]:::ok
""",
        [],
    ),
    (
        11,
        "lidar",
        "Lidar and walls evidence",
        "spatial_awareness's tick with a new lidar frame.",
        """flowchart TD
  T(["tick with a new lidar frame"]):::trig
  T --> TL["<b>_take_lidar()</b><br/>spatial_awareness/main.py"]:::sNew
  TL --> FL["<b>follow_lidar()</b>: a new grid when Billie nears its edge"]:::sNew
  FL --> SD{{"<b>LidarGrid.scan_due()</b>?"}}:::sDec
  SD -- "yes" --> AD["<b>LidarGrid.add()</b>: hits and misses at the lidar's height"]:::sNew
  T --> SW["<b>_see_through_walls()</b>"]:::sNew
  SW --> SR["<b>StandingWalls.set_reference()</b>: AutoXing's map walls"]:::sNew
  SR --> SWD{{"<b>StandingWalls.scan_due()</b> and a pose?"}}:::sDec
  SWD -- "yes" --> AS["<b>StandingWalls.add_scan()</b>: rays that pass a mapped wall"]:::sNew
  AD --> SN["<b>LidarGrid.snapshot()</b> · <b>StandingWalls.standing()</b><br/>used by <b>build_field()</b> · chart 4"]:::ok
  AS --> SN
""",
        [],
    ),
    (
        12,
        "active_mapping",
        "Active mapping: the initial scan",
        "The operator presses <b>Start Active Occupancy Mapping</b> in the mapping controls.",
        """flowchart TD
  T(["Start Active Occupancy Mapping"]):::trig
  T --> SAM["<b>start_active_mapping()</b><br/>commands/mapping_3d.py · arm preflight"]:::bNew
  SAM --> SM["<b>start_mapping()</b>: occupancy, a new map<br/>the arm to the mapping pose"]:::bChg
  SM --> IS["<b>initial_scan()</b><br/>behaviours/active_mapping.py"]:::bNew
  IS --> V["each of the 6 SCAN_VIEWS_DEG: <b>set_smooth_joints()</b><br/>at SCAN_ARM_SPEED (half) · planned · chart 2"]:::bOld
  V -- "refused" --> SK["the view skipped, a warning"]:::stop
  V -- "moved" --> WF["<b>_wait_for_frames()</b>: get_status until 3 new frames, ≤ 6 s"]:::bNew
  WF --> BK["back to the mapping pose · a summary with the mapping buttons"]:::ok
""",
        ["An abort raises <code>InterruptedError</code> in any wait, which stops the scan like any command."],
    ),
    (
        13,
        "placement",
        "The arm in the map",
        "The camera bridge builds its pose config; spatial_awareness places the arm; the mapper takes a frame.",
        """flowchart TD
  subgraph TOP[" "]
    CB(["camera bridge: pose config"]):::trig
    SA(["spatial_awareness: the arm base"]):::trig
    MF(["mapper: a frame with its pose"]):::trig
  end
  CB --> GPC["<b>_get_pose_config()</b><br/>camera_autoxing_bridge/main.py"]:::cChg
  SA --> AMT["<b>arm_mount()</b><br/>arm_surroundings.py"]:::sNew
  GPC --> AH["<b>arm_base_height_m()</b> · messages/pyroki_world.py<br/>ARM_BASE_HEIGHT_FIX_M → the robot's setting → 0.4405 m"]:::uNew
  AMT --> AH
  MF --> RST["<b>robot_state()</b> · nvblox_mapper/parsing.py<br/>base, joints, pose, arm mount, camera"]:::mNew
  RST --> RJ["robot.json next to the live files, for the viewers"]:::ok
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        14,
        "start_mapping",
        "Mapper: start mapping",
        "The brain's <code>start_mapping</code> command or button (TSDF or occupancy, optionally a replay capture).",
        """flowchart TD
  T(["start_mapping · tsdf / occupancy · capture_replay"]):::trig
  T --> SM["<b>start_mapping()</b><br/>brain/commands/mapping_3d.py"]:::bChg
  SM == "reconfigure · 7000 mm, texture 100" ==> RC["<b>ZedCamera.reconfigure()</b><br/>camera/zed/zed_camera.py · recording → refused"]:::cChg
  RC -- "fails" --> RS["<b>_restore_camera_depth_settings()</b>, raise"]:::stop
  SM -- "move_arm_on_start" --> MP["the arm to the mapping pose"]:::bOld
  SM == "start_mapping · «data» StartMappingRequest" ==> HS["<b>MappingSessionRuntime.handle_start_mapping()</b><br/>nvblox_mapper/mapping_session.py"]:::mChg
  HS --> AA{{"already mapping?"}}:::mDec
  AA -- "yes, same representation" --> DUP["duplicate start ignored"]:::stop
  AA -- "no · reset" --> RST["<b>reset_mapping_state()</b> → <b>BackendRuntime.configure_mapping()</b><br/>occupancy 2.5 cm · tsdf <b>mapping_voxel_size()</b> 5 cm"]:::mChg
  AA -- "no · resume" --> RES["<b>_resume_loaded_native_map()</b>: no map.nvblx → refused"]:::mNew
  RST --> RP{{"occupancy?"}}:::mDec
  RP -- "yes" --> OS["<b>OccupancyBackend.start()</b>: the patched wheel's API checked"]:::mNew
  RP -- "tsdf" --> OV["static ESDF overlay as before"]:::mOld
  OS --> CR{{"capture_replay?"}}:::mDec
  OV --> CR
  CR -- "yes" --> EC["<b>EvaluationCaptureRuntime.start()</b>: frames.mcap, manifest"]:::mNew
  EC == "start_recording · «data» StartRecordingRequest" ==> SR["<b>ZedCamera.start_recording()</b> (asked by <b>start_camera_svo()</b>)<br/>camera.svo2 + camera.json · already recording → raises"]:::cChg
  CR -- "no" --> ON["mapping_active"]:::ok
  SR --> ON
  ON == "reply · status" ==> CK{{"brain: representation and capture as asked?"}}:::bDec
  CK -- "no" --> STOP["stop, restore the camera, raise"]:::stop
""",
        [],
    ),
    (
        15,
        "frame_to_map",
        "Mapper: a camera frame into the map",
        "The camera bridge's tick pulls an RGB-D frame for the mapper; the xArm streams arrive at the bridge.",
        """flowchart TD
  subgraph TOP[" "]
    TK(["bridge tick"]):::trig
    AS(["xArm joints · TCP streams"]):::trig
  end
  AS --> HI["<b>on_xarm_servo_angles()</b> · <b>on_xarm_tcp()</b><br/>camera_autoxing_bridge/main.py · histories"]:::cNew
  TK --> EM["<b>_emit_mapper_frame()</b><br/>camera_autoxing_bridge/main.py"]:::cChg
  EM == "rgbd" ==> CAP["<b>ZedCamera.capture_rgbd()</b>: + camera settings, svo_frame_index"]:::cChg
  EM --> BR{{"<b>bracket_xarm()</b>: both arm clocks bracket the image?"}}:::cDec
  HI --> BR
  BR -- "no · stale > 0.2 s · no 2nd arm read" --> SK["frame skipped"]:::stop
  BR -- "yes" --> LP["<b>_lerp_xarm_pose6()</b> · <b>_lerp_servo()</b> · sync record v2"]:::cChg
  LP == "mapper_rgbd" ==> IF["<b>MapperRuntime.integrate_frame()</b><br/>nvblox_mapper/runtime.py"]:::mChg
  IF -- "a capture runs" --> SF["<b>EvaluationCaptureRuntime.submit_frame()</b>"]:::mNew
  IF --> G{{"gates: paused · pose jump · <b>arm_velocity_gate()</b> · <b>should_integrate()</b>"}}:::mDec
  G -- "a gate fails" --> DEC["decision logged, not integrated"]:::stop
  G -- "semantic masking on" --> SEM["<b>LiveSemanticMasking.submit()</b> → <b>poll_semantic()</b> later"]:::mNew
  G -- "pass" --> IS["<b>_integrate_static_frame()</b> → <b>BackendRuntime.integrate_frame()</b>"]:::mNew
  SEM --> IS
  IS --> KIND{{"occupancy backend?"}}:::mDec
  KIND -- "yes" --> OI["<b>OccupancyBackend.integrate_depth()</b>: masked pixels never clear"]:::mNew
  KIND -- "tsdf" --> NI["<b>NvbloxBackend.integrate()</b>: gripper mask <b>apply_gripper_policy()</b>, optional depth edges"]:::mChg
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        16,
        "telemetry",
        "Mapper: telemetry for the replay",
        "AutoXing's pose, its map identity and the arm joints arrive at the mapper.",
        """flowchart TD
  T(["nav_events tracked_pose · map/info · xarm servo_angles_deg"]):::trig
  T --> NE["<b>on_tick()</b> · <b>dora_payload()</b><br/>src/nav_events · map/info flattened"]:::xChg
  NE --> H["<b>on_autoxing_pose()</b> · <b>on_autoxing_map()</b> · <b>on_arm_joints()</b><br/>nvblox_mapper/runtime.py · bad input → ValueError"]:::mNew
  H --> OA["<b>EvaluationCaptureRuntime.observe_auxiliary()</b><br/>the latest of each kept (pre-roll)"]:::mNew
  OA --> AC{{"a capture runs and a slot is free?"}}:::mDec
  AC -- "yes" --> W["written into frames.mcap"]:::ok
  AC -- "no slot" --> DR["counted as dropped: the gate fails"]:::stop
""",
        [],
    ),
    (
        17,
        "mapper_tick",
        "Mapper: the tick",
        "The mapper's 1 Hz tick; and AutoXing's robot_pose for the clearance lane.",
        """flowchart TD
  T(["mapper tick"]):::trig
  T --> OT["<b>MapperRuntime.on_tick()</b><br/>nvblox_mapper/runtime.py"]:::mChg
  OT --> R{{"occupancy?"}}:::mDec
  R -- "yes" --> EO["<b>BackendRuntime.export_occupancy()</b>: ≥ 2 s apart, map changed<br/><b>OccupancyBackend.block_batches()</b> under the lock → occupancy.bin.gz"]:::mNew
  R -- "tsdf" --> RM["refresh the mesh, as before"]:::mOld
  EO --> S3["<b>live_map_stat()</b> in <b>tick_s3_mirror()</b><br/>new lane occupancy (2 s)"]:::mChg
  RM --> S3
  S3 --> DG["<b>MapperStatusRuntime.status()</b>: representation, occupancy, capture, arm guard"]:::mChg
  DG --> ST["status node <b>compact_mapping_status()</b>"]:::xChg
  T --> JC["<b>ClearanceRuntime.write_joint_clearance()</b>: occupancy → none (no ESDF)"]:::mChg
""",
        [],
    ),
    (
        18,
        "save",
        "Mapper: stop, save, finish",
        "The brain's <code>stop_mapping</code>, <code>save_mapping</code> or <code>finish_and_save_mapping</code>.",
        """flowchart TD
  subgraph TOP[" "]
    S(["stop_mapping"]):::trig
    F(["save · finish and save"]):::trig
  end
  S --> SR["<b>_stop_mapping_and_restore_depth()</b><br/>brain · the camera restored only without a capture"]:::bChg
  SR == "stop_mapping" ==> HS["<b>MappingSessionRuntime.handle_stop_mapping()</b><br/>the capture and SVO keep running"]:::mChg
  F --> FS["<b>finish_and_save_mapping()</b> · brain"]:::bChg
  FS == "save_mapping · finalize" ==> HV["<b>MappingSessionRuntime.handle_save_mapping()</b>"]:::mChg
  HV -- "finalize" --> SS["<b>_stop_replay_source()</b> → <b>stop_camera_svo()</b>"]:::mNew
  SS == "stop_recording" ==> ZS["<b>ZedCamera.stop_recording()</b>: camera.json completed"]:::cChg
  HV --> FL["<b>_flush_mapping_for_save()</b>"]:::mChg
  FL --> K{{"occupancy?"}}:::mDec
  K -- "yes" --> OF["export_occupancy(force) · save_native_map · the GPU map kept"]:::mNew
  K -- "tsdf" --> TF["<b>NvbloxBackend.update_esdf()</b> · GLB · <b>NvbloxBackend.save_native_map()</b> · bake"]:::mNew
  OF --> SN["<b>_build_local_snapshot()</b>"]:::mChg
  TF --> SN
  SN --> EF["<b>EvaluationCaptureRuntime.finish()</b> → the gate: frames, decisions, baseline, SVO, streams"]:::mNew
  EF -- "gate failed" --> ERR["saved, but an error is raised and the map is unpublishable · chart 19"]:::stop
  EF -- "passed" --> OK["replay_capture written into the manifest"]:::ok
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        19,
        "publish",
        "Mapper: publish",
        "The brain's <code>publish_map</code> (TSDF maps only).",
        """flowchart TD
  T(["publish_map"]):::trig
  T == "publish_map · 3600 s timeout" ==> HP["<b>MappingSessionRuntime.handle_publish_map()</b><br/>blocks the mapper's loop while it uploads"]:::mChg
  HP --> RC{{"the saved map references a capture?"}}:::mDec
  RC -- "yes" --> VC["<b>publish_replay_capture()</b> → <b>validate_replay_capture()</b><br/>gate not passed → MapUploadError"]:::mNew
  VC --> UP["<b>_upload_capture_archive()</b>: captures/&lt;id&gt;/capture.zip"]:::mNew
  RC -- "no" --> ZB
  UP --> ZB["<b>zip_operate_bundle()</b>: ESDF, mesh + map.nvblx"]:::mChg
  ZB --> PR["<b>publish_room_map()</b>: captures/&lt;id&gt;/operate.zip → latest"]:::mChg
  PR --> DEL["the saved map deleted · the capture folder kept"]:::ok
""",
        [],
    ),
    (
        20,
        "load_maps",
        "Mapper: load and delete maps",
        "The brain's <code>load_saved_mapping</code> (new), <code>load_room_map</code>, <code>load_current_map</code> or <code>delete_mapping</code>.",
        """flowchart TD
  subgraph TOP[" "]
    LS(["load_saved_mapping"]):::trig
    LR(["load_room_map · load_current_map"]):::trig
    DL(["delete_mapping"]):::trig
  end
  LS --> BLS["<b>load_saved_mapping()</b><br/>brain/commands/mapping_3d.py"]:::bNew
  BLS == "load_saved_mapping · «data» LoadSavedMappingRequest" ==> HL["<b>MappingSessionRuntime.handle_load_saved_mapping()</b>"]:::mNew
  HL --> V{{"not mapping · empty live map · occupancy manifest ·<br/>map.nvblx matches its voxel?"}}:::mDec
  V -- "no" --> R1["refused"]:::stop
  V -- "yes" --> LD["configure_mapping(occupancy) · <b>NvbloxBackend.load_native_map()</b><br/>export · integration paused"]:::mNew
  LR --> FV["<b>_fetch_and_validate_room_map()</b> → <b>_install_validated_room_map()</b><br/>back to TSDF · map.nvblx installed"]:::mChg
  DL --> DM["<b>delete_mapping()</b> brain → <b>reset_mapping_state()</b><br/>capture finished, SVO stopped"]:::bChg
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        21,
        "offline",
        "Mapper: offline tools, startup, build",
        "The offline command lines; the mapper node starting and closing; the image build.",
        """flowchart TD
  CI(["nvblox_capture_inspect"]):::trig --> CS["<b>capture_summary()</b><br/>the gate and the hashes"]:::mNew
  CS ~~~ CRP(["nvblox_capture_replay"]):::trig
  CRP --> RCP["<b>replay_capture()</b> · replay_cli.py<br/>preflight gates · SVO2 frames · masks<br/>always TSDF"]:::mNew
  NS(["mapper node starts · closes"]):::trig --> MN["<b>MapperRuntime.main()</b>: <b>bind_capture_source()</b>,<br/>3 new inputs · <b>MapperRuntime.close()</b>:<br/>capture finished, SVO stopped"]:::mChg
  MN ~~~ BS(["camera bridge starts"]):::trig
  BS --> LM["<b>_load_matrix_from_json_or_env()</b><br/>the file wins over env ·<br/>a non-rigid matrix stops the bridge"]:::cChg
  LM ~~~ IB(["image build"]):::trig
  IB --> PI["post-install.sh: the wheel needs the occupancy API<br/>build-nvblox-torch-wheel.sh: the occupancy patch<br/>always applied"]:::mChg
""",
        [],
    ),
]

# [chars] Label lines are split at this length in the charts that are wide side by side (arm_planner_doc splits at 46),
# so GitHub's 880px column shows their text at 85% or more.
_NARROW_LINE_CHARS = {0: 30, 2: 34, 6: 32, 14: 30, 15: 30, 18: 30, 20: 34, 21: 24}

# The first of the mapper's charts: from it on, the charts go into the PR's first comment (the description is full).
MAPPER_FIRST = 14

# Callers branch_inventory's name matching gets wrong here, fixed by hand: (file, name) -> (callers to add,
# caller prefixes to drop).
_CALLER_FIXES: dict[tuple[str, str], tuple[list[str], list[str]]] = {}

# Review findings: (text) lines in the review notes, written from the reading.
_FINDINGS: list[str] = [
    "**The saved map file never gives the planner a room.** The planner asks for a 2.5 cm crop "
    "(<code>esdf_world.py</code>), the mapper bakes <code>esdf_grid.esdfbin</code> at 5 cm, and "
    "<code>ArmSurroundings._current_crop()</code> returns no crop when the cells differ (confirmed). So the room exists "
    "only while the live occupancy maps; the map-file fallback of chart 3 gives none unless "
    "<code>PYROKI_ROOM_VOXEL_M</code> is 0.05. Decide: resample the file to 2.5 cm, or accept no fallback.",
    "**<code>plan_move</code> and <code>plan_replay</code> wait with no timeout** (<code>messages/pyroki_world.py:527, :599</code>, "
    "confirmed), and every arm move and replay now goes through them. A timeout with a clear refusal would keep a hung "
    "planner from freezing the arm.",
    "**The brain's guard checks map-frame objects where they were when they were set.** It keeps the arm-frame capsules "
    "from <code>set_obstacles</code> (<code>remember_obstacles()</code>); after Billie drives, the guard refuses or allows "
    "moves against the old positions, while the planner places them at the current pose (chart 8).",
    "**The guard's step check can start from a stale target**: <code>_last_target_deg</code> carries over between "
    "commands, so with objects set the first streamed move of a new command is checked from an old target and can trip "
    "the 90° spinning refusal (<code>arm_motion_guard.py</code>).",
    "**The guard's target list grows without bound**: <code>_queued_targets</code> gets every streamed AI or replay "
    "frame, even with no objects, and is cleared only by a single move (a small memory leak).",
    "**The guard's error names a command that does not exist** (<code>clear_virtual_obstacles</code>, confirmed); the way "
    "to clear is <code>set_obstacles</code> with an empty list.",
    "**The occupancy room stays after mapping stops**: the mapper serves blocks whenever the occupancy backend is "
    "loaded, not only while mapping. That matches \"believe old data\" (2026-10-14), but after a floor or map change the "
    "old occupancy would still constrain the arm; no map-identity check was found.",
    "**<code>reconfigure</code> validates before its no-change check**: even a no-op reconfigure raises when "
    "<code>ZED_DEPTH_MIN_MM</code> ≤ 0 (no robot's value is in the repo).",
    "<code>svo_frame_index</code> may be <code>None</code> in the rgbd response metadata; not checked whether dora accepts it.",
    "**Confirm before main:** the bridge now prefers the calibration file over the env (6f941b1b looks deliberate), and "
    "<code>ARM_BASE_HEIGHT_FIX_M = {\"BILLIE-07\": 0.49}</code> is a per-robot value in shared code, until billie-07's "
    "height is measured.",
    "**Publishing blocks the mapper for up to an hour**: <code>handle_publish_map()</code> runs on the mapper's loop and now "
    "uploads the whole capture. Meanwhile no integration, ticks, status or <code>get_occupancy_blocks</code> answers, so "
    "every arm plan falls back to the map file after 2 s (chart 3, chart 19).",
    "**A failed capture gate makes the map unpublishable**: the save completes with <code>replay_gate_passed: false</code>, "
    "and publish always validates that capture and raises; nothing publishes the map without it (chart 18, chart 19).",
    "**Capture folders are never deleted** after publishing (only the saved map is): <code>EVALUATION_DIR</code> grows "
    "without bound on the robot.",
    "**Stop Mapping during a capture leaves the SVO recorder and the MCAP writer running** (confirmed, "
    "<code>handle_stop_mapping()</code>): the arm and pose streams keep being written, and the camera cannot be "
    "reconfigured until Finish and Save, a reset or a delete.",
    "**Resuming a room map does not check its voxel size** (<code>_resume_loaded_native_map()</code>), unlike "
    "<code>load_saved_mapping</code>: a map published at 1 cm resumed under the new 5 cm default would mismatch.",
    "<code>export_occupancy(force=True)</code> waits on a previous export that may have failed, so a save can fail with an "
    "older background error; the export also holds the integration lock for the whole GPU copy.",
    "**Load Saved Map**: the button appears right after Finish and Save, but the load refuses while the live map has "
    "frames (Delete Live Map first); a failed load leaves the backend closed; it writes the private "
    "<code>backend._frames</code>.",
    "Occupancy finalize keeps the GPU map after Finish and Save (TSDF releases it); <code>operate.zip</code> now carries "
    "<code>map.nvblx</code>, so every robot's room-map download grows; <code>observe_auxiliary()</code> serialises every "
    "125 Hz arm sample to JSON even with no capture.",
    "**Open question:** the capture gate needs <code>autoxing_map</code> written once; <code>/map/info</code> is latched by "
    "AutoXing, so if it came before the mapper started and is not re-sent, the gate fails.",
    "If <code>_stop_replay_source()</code> raises during finalize, <code>mapping_active</code> is already off and nothing was "
    "saved.",
    "The function tables cover Python only (the inventory reads <code>.py</code> files): the nvBlox patches, shell "
    "scripts, the uploader and the CI workflow are in the system-wide table instead.",
]


def _fix_callers(rows: list[dict]) -> None:
    """Applies _CALLER_FIXES to the inventory rows, in place. rows: Inventory rows."""
    for r in rows:
        add, drop = _CALLER_FIXES.get((r["file"], r["name"]), ([], []))
        r["used_by"] = sorted({u for u in r["used_by"] if not u.startswith(tuple(drop))} | set(add))


def styles_for(body: str) -> str:
    """The classDef lines of the node styles a chart uses. body: The chart's mermaid text. Returns: The lines."""
    used = set(re.findall(r":::(\w+)", body))
    return "\n".join(line for line in STYLES.split("\n") if line.split()[1] in used)


def anchor(number: int) -> str:
    """The link to a chart's heading, as GitHub names heading anchors.

    number: The chart's number. Returns: "#chart-n--title" for its "## Chart N · title" heading.
    """
    title = next(t for n, _, t, _, _, _ in CHARTS if n == number)
    text = re.sub(r"[^\w\s-]", "", f"chart {number} · {title}".lower())
    return "#" + re.sub(r"\s", "-", text)


def link_mentions(text: str, number: int) -> str:
    """A sentence with each mention of another chart ("chart 3", "charts 3 and 4") turned into a link.

    text: The sentence (markdown or HTML). number: Its own chart, left unlinked. Returns: The sentence.
    """

    def link(match: re.Match) -> str:
        """One mention linked: "chart 2" whole, "charts 3 and 4" per number. match: A _CHART_MENTION match.

        Returns: The text.
        """
        numbers = [int(n) for n in re.findall(r"\d+", match.group(2))]
        if len(numbers) == 1:
            return match.group(0) if numbers[0] == number else f"[{match.group(0)}]({anchor(numbers[0])})"
        return re.sub(
            r"\d+",
            lambda d: d.group() if int(d.group()) == number else f"[{d.group()}]({anchor(int(d.group()))})",
            match.group(0),
        )

    return _CHART_MENTION.sub(link, text)


def click_lines(body: str, number: int) -> str:
    """Mermaid click lines making each box that names another chart open it (preview only).

    body: The chart's mermaid text. number: The chart's number. Returns: The lines.
    """
    lines = []
    for node, label in _NODE.findall(body):
        targets = mentioned_charts(label, number)
        if targets:
            lines.append(f'  click {node} href "{anchor(targets[0])}" _self')
    return "\n".join(lines) + ("\n" if lines else "")


def system_wide_section() -> list[str]:
    """The document's opening section: every change that reaches beyond the branch's features. Returns: Lines."""
    lines = [
        "## ⚠️ System-wide changes — confirm each one was intended before merging",
        "",
        "These reach beyond the arm awareness and the mapper themselves. Each one has to be a deliberate choice.",
        "",
        "| # | What changed | Who it affects | Risk |",
        "|---|---|---|---|",
    ]
    for i, (what, who, risk) in enumerate(SYSTEM_WIDE, 1):
        lines.append(f"| {i} | {link_mentions(what, -1)} | {who} | {link_mentions(risk, -1)} |")
    return lines + [""]


def review_notes(rows: list[dict], charts_of: dict[int, list[int]]) -> list[str]:
    """The review notes: what the reading found, code nothing calls, and how much no chart draws.

    rows: Inventory rows. charts_of: Row index -> the charts that draw it. Returns: Markdown lines.
    """
    lines = ["## Review notes", ""] + [f"- {link_mentions(f, -1)}" for f in _FINDINGS]
    unused = [
        r
        for r in rows
        if not r["used_by"] and r["status"] != "removed" and r["name"] != "main"
        and not r["name"].endswith(("__init__", "__call__", "__post_init__"))
    ]  # fmt: skip
    if unused:
        lines += [
            f"- **{len(unused)} functions and classes nothing calls** (by name; some are entry points, handlers "
            "registered by name, or used only by tests): "
            + ", ".join(f"{kind_marker(r)} in `{r['file'].replace('billie/', '')}`" for r in unused)
            + "."
        ]
    undrawn = [i for i in range(len(rows)) if i not in charts_of]
    lines += [
        f"- {len(undrawn)} of {len(rows)} functions and classes are not drawn in a chart; each is a helper of one "
        "that is, or a removed one (see its *Called from*).",
        "",
    ]
    return lines


def _rows() -> list[dict]:
    """The inventory rows the review covers: everything but tests, callers fixed. Returns: The rows."""
    with open(INVENTORY) as f:
        rows = [r for r in json.load(f)["rows"] if "/tests/" not in r["file"] and "/test/" not in r["file"]]
    _fix_callers(rows)
    return rows


def _guide(count: int) -> list[str]:
    """The how-to-review and how-to-read sections. count: Functions and classes covered. Returns: Lines."""
    return [
        f"**{count} functions and classes** are added, changed or removed (tests not counted). Every one is listed "
        "in the comments below, [All changed functions and classes](#all-changed-functions-and-classes).",
        "",
        "## How to review",
        "",
        "1. Confirm each system-wide change in the table above.",
        "2. Read [Chart 0 · Overview](#chart-0--overview): which command reaches which node.",
        "3. Open one chart at a time. Each starts from what triggers it; every branch is a real condition in the code.",
        "4. The mapper's charts (14 to 21) are in the first comment, [The mapper's charts](#the-mappers-charts).",
        "5. [All changed functions and classes](#all-changed-functions-and-classes) lists every one, file by file.",
        "6. [Review notes](#review-notes) lists what the reading found worth a look.",
        "",
        "## How to read the charts",
        "",
        "| What you see | Meaning |",
        "|---|---|",
        "| Blue / purple / orange / teal / grey / green fill | Runs in the brain / pyroki-planner / spatial_awareness / "
        "nvblox_mapper / the camera nodes / shared `billie_utils` code |",
        "| Thick solid / dashed / thin grey border | New on the branch / changed on the branch / unchanged, shown for context |",
        "| `name()` | A function. `module.name()` when the same name exists in two modules |",
        "| `Class.method()` | A method, called on an instance of `Class` |",
        "| `«class» Name` / `«data» Name` | A class / a data class (a message). On a thick arrow: request → reply |",
        "| Rounded pill | What starts the flow |",
        "| Hexagon | A decision, filled like the node it runs in; its arrows name the cases |",
        "| Green box / red box | Goes ahead / refused, skipped or failed |",
        "| Thin / thick / dashed arrow | A call in one node / a request to another node / the other case |",
        "",
    ]


def _wrapped(body: str, number: int) -> str:
    """A chart's labels split to fit its boxes, narrower for the wide charts (_NARROW_LINE_CHARS).

    body: The chart's mermaid text. number: The chart. Returns: The chart with its long label lines split.
    """
    default = arm_planner_doc._MAX_LINE_CHARS
    arm_planner_doc._MAX_LINE_CHARS = _NARROW_LINE_CHARS.get(number, default)
    try:
        return wrap_labels(body)
    finally:
        arm_planner_doc._MAX_LINE_CHARS = default


def _chart_lines(preview: bool, mapper: bool) -> list[str]:
    """The arm-awareness or the mapper charts, collapsed except the overview.

    preview: Add the click lines. mapper: The mapper's charts (MAPPER_FIRST on), else the others. Returns: Lines.
    """
    lines: list[str] = []
    for number, _, title, when, body, notes in CHARTS:
        if (number >= MAPPER_FIRST) != mapper:
            continue
        lines += [
            f"## Chart {number} · {title}",
            "",
            f"**When:** {link_mentions(when, number)}",
            "",
            f"**What changed:** {link_mentions(SUMMARIES[number], number)}",
            "",
            "<details open>" if number == 0 else "<details>",
            f"<summary><b>Open chart {number}</b></summary>",
            "",
            "```mermaid",
            _wrapped(body, number) + (click_lines(body, number) if preview else "") + styles_for(body),
            "```",
            "",
            "</details>",
            "",
        ]
        targets = sorted({n for label in label_strings(body) for n in mentioned_charts(label, number)})
        if targets:
            lines += ["Goes on in: " + " · ".join(f"[Chart {n}]({anchor(n)})" for n in targets), ""]
        lines += [f"- {link_mentions(note, number)}" for note in notes] + ([""] if notes else [])
    return lines


def _comment_parts(rows: list[dict], charts_of: dict[int, list[int]]) -> list[str]:
    """Every changed function and class, file by file, split into comments under GitHub's limit.

    rows: Inventory rows. charts_of: Row index -> its charts. Returns: The comments' markdown, in order.
    """
    files: list[tuple[str, list[int]]] = []
    for i, r in enumerate(rows):
        if not files or files[-1][0] != r["file"]:
            files.append((r["file"], []))
        files[-1][1].append(i)
    blocks = []
    for path, indexes in files:
        new = sum(rows[i]["status"] == "new" for i in indexes)
        blocks.append("\n".join([
            "<details>",
            f"<summary><code>{path.replace('billie/', '')}</code> · {len(indexes)} ({new} new, "
            f"{len(indexes) - new} changed or removed)</summary>",
            "",
            table(rows, indexes, charts_of, with_file=False),
            "",
            "</details>",
            "",
        ]))  # fmt: skip
    parts: list[list[str]] = [[]]
    for block in blocks:
        if parts[-1] and len("\n".join(parts[-1] + [block]).encode()) > _PART_BYTES:
            parts.append([])
        parts[-1].append(block)
    out = []
    for n, part in enumerate(parts, 1):
        head = (
            ["## All changed functions and classes", ""] if n == 1 else [f"## All changed functions and classes ({n})", ""]
        )
        intro = [
            f'Part {n} of {len(parts)}, grouped by file. *Chart* says where each one is drawn; "—" means it is a '
            "helper of a function that is.",
            "",
        ]
        out.append("\n".join(head + intro + part))
    return out


def build(preview: bool = False) -> tuple[str, list[str]]:
    """The review document as markdown: the PR description and its comments, each within GitHub's limit.

    preview: Also make the boxes that name another chart clickable (preview only).
    Returns: (the PR description: system-wide changes, guide, charts and review notes; the comments: every changed
        function and class).
    """
    rows = _rows()
    charts_of: dict[int, list[int]] = {}
    for number, _, _, _, body, _ in CHARTS:
        for i in sorted(chart_items(body, rows)):
            charts_of.setdefault(i, []).append(number)
    lines = [
        "# arm_plan_esdf: what changed and how it flows",
        "",
        "Review guide for the `arm_plan_esdf` branch (PR #1386, compared with `main`). Two bodies of work: the arm "
        "awareness (every arm command planned against the room around the arm, built from the live occupancy) and "
        "the nvblox mapper work merged into it (the live occupancy itself, capture and replay, the mapper's quality "
        "fixes).",
        "",
    ]
    lines += system_wide_section() + _guide(len(rows)) + _chart_lines(preview, mapper=False)
    lines += review_notes(rows, charts_of)
    mapper = [
        "## The mapper's charts",
        "",
        "The nvblox mapper work merged into this branch (capture and replay, the occupancy representation, the mapper's "
        "quality changes): charts 14 to 21. The system-wide changes and review notes in the description cover them too.",
        "",
    ] + _chart_lines(preview, mapper=True)
    return "\n".join(lines), ["\n".join(mapper)] + _comment_parts(rows, charts_of)


def main() -> None:
    """Writes the PR description, its comment parts and the HTML preview into billie-onboard/out/review/."""
    description, comments = build()
    os.makedirs(_OUT_DIR, exist_ok=True)
    outputs = [(OUT_MD, description)] + [(OUT_COMMENT_MD.format(n), c) for n, c in enumerate(comments, 1)]
    for path, text in outputs:
        with open(path, "w") as f:
            f.write(text)
        size = len(text.encode())
        print(f"{path}: {size} bytes" + ("" if size <= _GITHUB_MAX_BYTES else f" - OVER GitHub's {_GITHUB_MAX_BYTES}"))
    preview_description, preview_comments = build(preview=True)
    page = preview_html("\n".join([preview_description, *preview_comments])).replace(
        "<title>Arm Planner Call Map</title>", "<title>Arm Plan ESDF Review</title>"
    )
    with open(OUT_HTML, "w") as f:
        f.write(page)
    print(OUT_HTML)


if __name__ == "__main__":
    sys.exit(main())
