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

from Simulation.review import arm_planner_doc, box_notes
from Simulation.review.arm_planner_doc import (
    _GITHUB_MAX_BYTES,
    _NODE,
    chart_items,
    kind_marker,
    label_strings,
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
  classDef stop fill:#f8dada,stroke:#b3261e,stroke-width:1.5px,color:#1d2330;
  classDef ref fill:#f7f7f7,stroke:#9aa0a6,stroke-width:1px,stroke-dasharray:2 2,color:#1d2330;"""
# The frame of an area of a chart ("2.1 · ..."), the part other charts call by its number.
AREA_STYLE = "fill:none,stroke:#6b7280,stroke-width:1.5px,stroke-dasharray:5 4,color:#1d2330"

# Every change that reaches beyond the branch's features, first in the document (the review convention's item 0):
# (what changed, who it affects, risk). Filled from the whole branch diff against the merge-base.
SYSTEM_WIDE: list[tuple[str, str, str]] = [
    (
        "<b>Every single arm move is planned</b>: <code>set_state_with_arm_joints(clear_queue=True)</code> "
        "(joints, home, fold, look_up, map_pose, scans, a replay's first state) and <code>pose</code> call "
        "<code>plan_move</code> (30 s, then refused); <code>direct</code> still checks the joint limits and "
        "self-collision (chart 2)",
        "Every arm command",
        "<b>High.</b> A planner that is down refuses every move after 30 s, fold included. Moves can be refused "
        "or rerouted; refused without arm joints; a pose no IK reaches is refused (<code>closest=True</code> "
        "gives the old best effort); a pose inside the base only warns",
    ),
    (
        "<b>Every plan waits for spatial_awareness</b> (<code>get_arm_surroundings</code>, 6 s) (chart 4)",
        "Every arm move, AI step, replay",
        "<b>High.</b> A missed answer refuses moves within 1 m of the room (\"no room in time\"). The kept room "
        "answers at once while the base stands; only the first plan after a base move waits (0.8-1.4 s on an idle "
        "Orin)",
    ),
    (
        "<b>Blocking waits changed in every node</b> (new, <code>billie_utils/node.py</code>): "
        "<code>safe_next(timeout)</code> raises <code>TimeoutError</code> instead of "
        "<code>StopIteration</code>; responses arriving during a nested wait are kept for their waiter; "
        "timeouts are deadlines",
        "Every node that waits on a request",
        "<b>Medium.</b> Fixes two bugs from main: a nested wait dropped the outer response (a 300 s scan hang, "
        "billie-14), and an idle timeout stopped the node; code relying "
        "on <code>StopIteration</code> after a timeout now gets <code>TimeoutError</code>",
    ),
    (
        "<b>An AI task's <code>duration</code> excludes planner waits</b> (new, chart 8)",
        "Every AI task with a duration",
        "<b>Medium.</b> A task can run longer than <code>duration</code> in real time",
    ),
    (
        "<b>IK results are post-checked and self-collision is always on</b>: <code>keep_solution_clear()</code> "
        "on every solve; <code>ENABLE_PYROKI_SELF_COLLISION</code> ignored; batches keep 2 cm from the room, "
        "single solves and routes 3.5 cm (chart 7, chart 11)",
        "AI and replay motion",
        "<b>Medium-high.</b> Frames can be shortened (logged only); more compute where self-collision was off",
    ),
    (
        "<b>replay_policy is planned</b>: untransformed replays with recorded joints go through "
        "<code>plan_replay</code> (60 s) and raise <i>Replay blocked</i> without a route (chart 9); replay and "
        "AI frames run in <code>arm_touch.touching()</code> (fingers 8 cm into the room), the moves before them "
        "only with <code>pre_action_touch</code>",
        "replay_policy, AI tasks",
        "<b>Medium-high.</b> Recordings can be rerouted or refused, <code>transform=joints</code> too; an "
        "alignment refusal waits up to 10 s",
    ),
    (
        "<b>IK starts 16 → 4</b> (<code>MAX_N_ST_PERTURB</code>, default and limit)",
        "Every IK caller",
        "<b>Medium.</b> <code>n_st_perturb</code> above 4 fails. Billie-26: same accuracy, batches 3-4× faster",
    ),
    (
        "<b>New option <code>pre_action_touch</code></b>: the moves before an action (AI start and reverse "
        "playback, a replay's first frame) touch only with it",
        "ai_task, replay_policy",
        "<b>Medium.</b> By default they keep 2 cm from the room; a replay starting in contact is refused unless "
        "set",
    ),
    (
        "<b>The brain tells the mapper before every base move</b>: <code>send_output()</code> sends "
        "<code>base_moving</code> (≤ every 0.5 s) before nav moves (10 s lease), twist, velocity, slide (2 s) "
        "and rotate (10 s)",
        "Every base command",
        "<b>Low.</b> Failures only logged; the mapper takes no frames during the lease",
    ),
    (
        "<b>Unrequested arm moves</b>: a move refused while occupancy mapping runs first looks at up to 2 views "
        "at half speed, replans, and may return to the start (chart 3)",
        "Arm commands while occupancy mapping runs",
        "<b>Medium.</b> The arm moves before refusing; a busy mapper delays the refusal up to 30 s (status "
        "timeout)",
    ),
    (
        "<b>More requests per arm command</b>: <code>plan_move</code>, and on a refusal "
        "<code>get_aligned_pose</code> (2 s, polled ≤ 10 s), mapper status, <code>plan_look</code>",
        "Every arm command",
        "<b>Low-medium.</b> A refusal takes up to ~10 s longer",
    ),
    (
        "<b>No live joint clearance in occupancy mode</b> (<code>write_joint_clearance()</code> returns none) "
        "(chart 19)",
        "Readers of the mapper's clearance lane",
        "<b>Medium.</b> The planner's room (chart 4) is then the only arm check",
    ),
    (
        "<b>Camera recording</b>: <code>start_recording</code> raises if already recording; every SVO writes "
        "<code>camera.json</code>; <code>stop_recording</code> raises if finishing fails; "
        "<code>reconfigure</code> validates <code>0 &lt; min &lt; max</code> and confidence 0-100 first; rgbd "
        "metadata +7 fields; the mapper owns SVO recording during capture",
        "Episode recording, mapping start/stop",
        "<b>Medium.</b> A leaked recorder blocks recording until a camera restart; episodes and captures "
        "exclude each other; <code>ZED_DEPTH_MIN_MM</code> ≤ 0 blocks mapping",
    ),
    (
        "<b>Bridge calibration</b>: <code>camera_calibration.json</code> wins over "
        "<code>T_GRIPPER_FROM_CAMERA</code>; a non-rigid matrix crashes the bridge; unbracketed frames dropped; "
        "arm base height from <code>arm_base_height_m()</code>: the robot's setting, else 0.4405 m (chart 15)",
        "Every map frame and robot_pose",
        "<b>Medium.</b> Frames move where file and env differ; billie-07 needs "
        "<code>AUTOXING_BASE_TO_ARM_BASE_Z_M=0.49</code> (its floor sat ~5 cm low)",
    ),
    (
        "<b>Fixed-EEF twist compensation streams</b>: its 50 ms steps go to the writer unplanned "
        "(<code>stream=True</code>), each solved against the room by the IK (chart 2)",
        "Fixed-EEF twist",
        "<b>Low.</b> The steps are solved against the room, not checked as a path",
    ),
    (
        "<b>A single move must arrive</b> (new): the writer holds <code>joints_queue_emptied</code> until the arm "
        "is within 2° of the target and outputs <code>move_stalled</code> 3 s after; the brain stops waiting at "
        "3 × the move's time + 8 s and holds the arm (chart 10)",
        "Every single arm move",
        "<b>Medium.</b> A stuck or blocked arm raises <i>Arm move stalled</i> instead of hanging; on billie-28 no "
        "false alarm in ~15 moves",
    ),
    (
        "<b>Joints goals check the arm against Billie's body</b>: <code>joints</code>, <code>fold</code> and look "
        "views refuse a move into the backpack, platform, base or covering cylinder; arm against arm stays "
        "unchecked (chart 2)",
        "joints, fold, look_up, scans, looks",
        "<b>Medium.</b> A joints move into Billie's body is refused",
    ),
    (
        "<b>IK and replay waits end</b>: <code>pyroki_node.solve()</code> / <code>batch_solve()</code> and the "
        "BufferingPlayer give up after 30 s; AI tasks and replays first wait for the planner's compile (up to "
        "10 min) (chart 7, chart 8, chart 9)",
        "Every IK caller; ai_task, replay_policy",
        "<b>Medium.</b> A hung planner fails the step instead of blocking; a first task after a deploy waits",
    ),
    (
        "<b>Queued moves are only recorded</b> for debug plots (<code>arm_motion_guard.py</code> no longer "
        "checks; the planner does, chart 10)",
        "Every process queueing joints",
        "<b>Low.</b>",
    ),
    (
        "<b>spatial_awareness background work</b>: lidar alignment per scan, standing walls, lidar grid, and "
        "the arm ESDF export (a mapper request every second, none while driving; a 118³ field rebuilt when the "
        "map, the base or the arm's places change); the planner's room refreshed every 2 s while the base stands; S3 lanes "
        "<code>esdf</code> (1 s), <code>occupancy</code> (2 s) (chart 12, chart 13)",
        "Orin CPU; the maps bucket",
        "<b>Medium.</b> Periodic CPU and S3 PUTs; CPU not measured on an Orin",
    ),
    (
        "<b>Dataflow</b>: the mapper gets <code>servo_angles_deg</code> (125 Hz, queue 4096), "
        "<code>tracked_pose</code>, <code>map/info</code>, <code>xarm_reader/mode</code>; the bridge gets the "
        "servo and TCP streams; new clients camera → mapper, spatial_awareness → planner, mapper → "
        "spatial_awareness",
        "mapper, bridge, camera, spatial_awareness",
        "<b>Low-medium.</b> Two more nodes take the 125 Hz arm streams",
    ),
    (
        "<b>Publishing uploads the capture synchronously</b> on the mapper's loop; the brain still waits 300 s, as "
        "on main (chart 21)",
        "The brain; mapper clients",
        "<b>Medium.</b> A longer upload fails the brain's publish while the mapper is still busy; meanwhile arm "
        "plans use the kept room, else none",
    ),
    (
        "<b>AI tasks pause the mapper</b> (2 s timeout each way) (chart 8)",
        "Every AI task",
        "<b>Low-medium.</b> No map frames during AI tasks",
    ),
    (
        "<b>Mapper build</b>: <code>transformers&gt;=5.8</code>, <code>torchvision</code>, <code>mcap</code>, "
        "<code>zstandard</code>, <code>opencv-python</code>, <code>pillow</code>, the bridge (editable); a "
        "patched nvBlox wheel (LFS) checked by <code>post-install.sh</code>; 4 opt-in patches; two capture "
        "console scripts",
        "Release image, mapper venv",
        "<b>Medium.</b> Slower builds; fails without <code>git lfs pull</code>",
    ),
    (
        "<b>Planner build</b>: <code>flatbuffers</code>, <code>absl-py</code>; exported solvers cached under "
        "<code>$BILLIE_DIR/pyroki-jax-cache/exported</code>, never cleaned, fingerprinted by the solver files "
        "(<code>UNTRACED_FILES</code> excluded); <code>planner_status</code> tells whether they are compiled, "
        "and mapping commands wait up to 10 min for it (chart 11, chart 14)",
        "pyroki-planner; mapping commands",
        "<b>Low.</b> Needs <code>./install.sh nodes/pyroki-planner</code>; disk grows; a solver change "
        "re-traces (4.5 + 3 min on billie-28)",
    ),
    (
        "<b>Commands</b>: <code>set_obstacles</code>; hidden <code>start_active_mapping</code>, "
        "<code>scan</code>, <code>scan_unknown</code>, <code>load_saved_mapping</code>; new "
        "<code>check_joints</code> (plans, never moves); arguments <code>direct</code> (joints, fold, look_up, "
        "set_smooth_joints), <code>closest</code> (pose), and <code>start_mapping</code>'s "
        "<code>capture_replay</code>, <code>move_arm_on_start</code>, <code>representation</code>, "
        "<code>capture_format</code>; option <code>direct_action</code> (ai_task, replay_policy: no room, chart "
        "8, chart 9); 6 mapping buttons; occupancy maps cannot be published",
        "Operators",
        "<b>Low.</b> <code>direct</code> and <code>direct_action</code> skip the room (off by default)",
    ),
    (
        "<b>nav_events</b>: subscribes to <code>/map/info</code>, strips a leading <code>/</code> from topics, "
        "flattens <code>map/info</code>",
        "nav_events",
        "<b>Low.</b> An empty topic would raise",
    ),
    (
        "<b>Cloud status</b> gains <code>representation</code>, <code>evaluation_capture</code>, the mapper's "
        "<code>arm_manual</code>, and <code>arm.blocked_move</code> (the last refused move, chart 3)",
        "Cloud consumers; the brain's <code>status</code> output",
        "<b>Low.</b> Additive",
    ),
    (
        "<b>Uploader</b>: <code>.destination</code> sidecars with a bucket allowlist; <code>$file</code> quoted",
        "Uploader containers",
        "<b>Low.</b> Unused yet; reinstall needed",
    ),
    (
        "<b>brain <code>node.py</code></b>: <code>send_cloud_message</code> and "
        "<code>upload_for_debugging</code> take an <code>activity_id</code> (skips the silent check)",
        "Brain cloud messages",
        "<b>Low.</b> Only the plot worker uses it",
    ),
    (
        "<b>Repo files</b>: <code>MAPPING_BRANCH.md</code>, <code>NVBLOX_OFFLINE_REPLAY_QUALITY_PLAN.md</code>, "
        "<code>SHARED_MAPPING_QUALITY.md</code>; <code>experiments/texture-mapping-20261012/</code> (8,401-line "
        "patch, 627 KB bundle not in LFS); <code>billie/mapping-core</code> with its own CI",
        "Repo, CI",
        "<b>Low.</b> Repo size; mapping-core's CI depends on two billie_utils modules",
    ),
    (
        "<b>Message fields</b>: <code>touch</code> on the solve and plan requests (from the global "
        "<code>arm_touch.touching()</code>); camera <code>texture_confidence_threshold</code>; "
        "<code>StopRecordingResponse.recording_stopped</code>",
        "billie-utils message users",
        "<b>Low.</b> Defaults kept; an unclosed touch scope would leak",
    ),
    (
        "<b>Debug plots in a subprocess</b> (<code>DEBUG_PYROKI_PLANNER</code> only): AI, replay and "
        "blocked-move plots, posted later",
        "Robots with that flag",
        "<b>Low.</b> ≤ 20 queued jobs, 15 min limit",
    ),
]

# The branch's own design choices, at the end of the description: (decision, its effect, why). "Why" is the author's
# reason (the user, 2026-11-15: "author decisions"); a row without one shows AUTHOR_TO_EXPLAIN.
AUTHOR_DECISIONS: list[tuple[str, str, str]] = [
    (
        "<b>The scan looks only in front</b>: 6 views, J1 ±30°, 28° down, 10° up (chart 14)",
        "~25 % of the reach (sides, back) stays unknown. Until the space is seen, the scan takes the space near the "
        "base as free, so <b>the first scan needs Billie in free space</b> (billie-26, 2026-11-11: two scans started "
        "beside something unmapped stopped on xArm error 31; the same views passed in free space)",
        "Too many never-observed cells were counted as occupied, so the arm could not move.",
    ),
    (
        "<b>Never-observed space counts as occupied, except \"probably free\" cells</b>: free on 4 of 6 sides; "
        "within 10 cm of N = 1 seen free cell (<code>PROBABLY_FREE_MIN_SEEN</code>) and no seen occupied one; within "
        "20 cm of where the arm is or was; above the base to 2 m; in the base's shadow (50 cm, to 80 cm high); near "
        "the base for a scan (chart 5)",
        "Moves into unmapped space are refused until scanned (chart 3, chart 14). The exceptions are guesses: on "
        "billie-28 0.06 % of probably-free cells were later seen occupied (the same at N = 1 and N = 40)",
        "Too many never-observed cells were counted as occupied, so the arm could not move.",
    ),
    (
        "<b>Saved 3D maps are not used</b> (3480a777, <code>USE_SAVED_3D_MAP = False</code>, "
        "<code>PYROKI_ESDF_MAP</code> ignored): without occupancy mapping the arm has no room; alignment uses "
        "AutoXing's 2D map only (chart 4)",
        "Moves outside occupancy mapping are unchecked against the room, as on main",
        "A room-type map from another room steered a door replay off its handle (billie-26).",
    ),
    (
        "<b>The room is kept while the base stands</b>: answered at once, refreshed every 2 s, rebuilt only when "
        "the map changes or the arm reaches new never-seen space; nothing is built while driving (chart 4)",
        "The room can be up to 30 s old without a mapper answer; the first plan after a base move waits for a build",
        "The room already holds everything within the arm's reach, and nothing new is mapped while the arm moves or "
        "an AI task runs, so a motion (an AI task, a replay) needs no rebuild. Rebuilding per batch took 31 of an AI "
        "task's 40 s (billie-26).",
    ),
    (
        "<b>Gripper mask <code>billie-gripper-body</code></b>: only the gripper's box is masked, 9 % of the image "
        "(v1 hid 16 %) (the mapper section, row 9)",
        "The space in front of the gripper is mapped",
        "v1's growth 0.15 hid the space in front of the gripper; Billie's self-mask covers the rest.",
    ),
    (
        "<b>No frames in xArm manual mode</b> (the mapper section, row 10)",
        "Nothing is mapped while the arm is moved by hand",
        "A person usually moves the arm then, and their body becomes occupied cells that stay after they leave.",
    ),
    (
        "<b>Occupancy rays clear what they pass</b>: far pixels to 5 m, pixels above 2.2 m to there, Billie's own "
        "pixels to 5 cm before the body; missing blocks along cleared rays are created (the mapper section)",
        "Less unknown; a thin object the depth misses on a far ray can be cleared",
        "Space in clear view stayed unknown (billie-28, 2026-11-08).",
    ),
    (
        "<b>The self check allows 1 cm into Billie's body boxes</b> (2 cm for link2 near the backpack), for every "
        "move (chart 11)",
        "A real hit up to 1 cm is not refused (v2 item 4)",
        "Measured on 71,768 recorded configurations; a capsule around the platform refused billie-07's fold.",
    ),
    (
        "<b>An uncheckable pose counts as aligned</b>: with no reference walls in view the pose is "
        "<code>unchecked</code> and the arm goes on, unless map-frame objects are set",
        "The room is then placed with a pose nothing checked",
        "No evidence the pose is off either (2026-11-08).",
    ),
]
# Notes for the chart boxes with nothing to read them from (no function, or functions without a docstring), shown when
# the box is clicked in the preview (box_notes.py; the user, 2026-11-15): (chart, box id) -> note.
BOX_NOTES: dict[tuple[int, str], str] = {
    (0, "BR"): "The brain: every arm command now asks the planner before the arm moves.",
    (0, "PL"): "The pyroki planner: solves IK and checks every move against the room, objects, floor and the arm itself.",
    (0, "SA"): "spatial_awareness: builds the room around the arm from the occupancy, and places the objects.",
    (0, "MP"): "The nvblox mapper: the live 2.5 cm occupancy map from the camera.",
    (0, "EX"): "The arm's field, written once a second and mirrored to S3 for the cloud's 3D map.",
    (0, "G"): "The xArm writer: drives the arm through the queued joints.",
    (0, "ST"): "The robot status carries the last refused move; the cloud's arm simulation draws it.",
    (1, "BS"): "The brain's single moves and replays: planned, retried for alignment, looked at first.",
    (1, "BA"): "The brain's AI task, mapper pause, scans, notices and the base-move notice to the mapper.",
    (1, "PS"): "The planner's request handlers: single moves, batches, replays, look views.",
    (1, "PE"): "The planner's world: the room grid, the arm's spheres, the self check, the surroundings fetch.",
    (1, "PR"): "The IK solvers (single, batch, transit routes).",
    (1, "SF"): "spatial_awareness: the field from the occupancy, the surroundings answers.",
    (1, "SE"): "spatial_awareness's export of the arm's field to the cloud.",
    (1, "U"): "Shared messages and helpers the brain, planner and spatial_awareness use.",
    (1, "M"): "The mapper's parts for the arm: block requests, the base-motion undo, the block table.",
    (2, "RF"): "The planner refused the move; the brain waits for alignment or looks first (chart 3).",
    (3, "AP"): "spatial_awareness answers whether the robot's pose is aligned to the map.",
    (4, "OG"): "spatial_awareness's handler for the planner's request; the answer goes out from a later tick.",
    (4, "MP"): "The mapper answers with the occupancy blocks around the arm (chart 6).",
    (7, "RS"): "The IK solve: 4 starts; pose, joint pull, self-collision, room and capsule costs; 3.5 cm (one pose) or "
    "2 cm (batch) from the room.",
    (7, "BR"): "The brain gets the solved joints back.",
    (8, "PH"): "The mapper keeps its session and map but skips new frames while the AI task runs.",
    (8, "RUN"): "One attempt of the AI task: the move to its start, the action, and the reverse playback if needed.",
    (8, "PRE"): "The move to starting_joints, planned; it touches the world only with pre_action_touch.",
    (8, "DA"): "With direct_action the planner sees no room, objects or floor for the whole attempt.",
    (8, "SOL"): "The planner solves the AI's poses to joints (chart 7).",
    (8, "REV"): "Plays the attempt backwards after a failure (or with verify_reverse): a planned move to its first "
    "frame, then the frames straight to the writer.",
    (9, "IS"): "Solves the recording's first pose and moves there with a planned move.",
    (11, "AR"): "Keeps the built solvers and marks the planner ready.",
    (12, "MP"): "The mapper answers with the blocks around the arm (chart 6).",
    (12, "S3"): "The mapper's S3 mirror uploads arm_esdf.bin.gz on lane esdf.",
    (14, "WP2"): "Waits for the planner's first compile, as above.",
    (14, "SV"): "The planner aims up to 6 camera views at the cells; each is a half-speed planned move, waiting for "
    "frames; then the arm returns.",
    (15, "GPC"): "The camera bridge's pose settings: the calibration and the arm base height.",
    (16, "RC"): "Sets the ZED's depth cap and texture confidence for mapping; refused while recording.",
    (16, "HS"): "The mapper starts a session: reset or resume, the representation, the capture.",
    (16, "OS"): "Starts the occupancy backend; checks the patched nvBlox wheel has the occupancy API.",
    (16, "OV"): "TSDF mode keeps its static ESDF overlay, as on main.",
    (16, "EC"): "Opens the replay capture: frames.mcap and its manifest.",
    (17, "EM"): "The bridge pulls an RGB-D frame and joins it with the bracketing arm samples.",
    (17, "IF"): "The mapper takes a frame: the gates, then integration.",
    (17, "SF"): "Writes the frame and its decision into the capture.",
    (17, "SEM"): "Masks people and the robot in the frame, asynchronously; integrated when ready.",
    (18, "H"): "Keep the latest AutoXing pose, map identity and arm joints; record them while capturing.",
    (19, "OT"): "The mapper's 1 Hz tick: exports, mirrors and the status.",
    (19, "RM"): "TSDF mode refreshes the mesh, as on main.",
    (19, "DG"): "The mapper's status: representation, occupancy, capture health, the arm guard.",
    (19, "ST"): "The status node folds the mapper's status into the robot status.",
    (19, "JC"): "The joint-clearance lane; in occupancy mode there is none (no ESDF).",
    (20, "SR"): "The brain stops mapping and restores the camera depth, unless a capture runs.",
    (20, "HS"): "The mapper stops integrating; a running capture and SVO keep going.",
    (20, "HV"): "The mapper saves the map.",
    (20, "ZS"): "The camera stops recording and completes camera.json.",
    (20, "FL"): "Flushes the map before saving.",
    (20, "OF"): "Occupancy save: a forced export, the native map saved, the GPU map kept.",
    (20, "SN"): "Builds the saved map's local snapshot.",
    (20, "EF"): "Closes the capture and checks it: frames, decisions, baseline, SVO, streams.",
    (21, "HP"): "Publishes the map, uploading the capture first; the mapper's loop is blocked meanwhile.",
    (22, "FV"): "Downloads and validates a room map, installs it with its map.nvblx, back in TSDF.",
    (23, "CS"): "Summarises a capture: the gate result and file hashes.",
    (23, "MN"): "The mapper's startup wires the capture source and 3 new inputs; closing finishes the capture.",
    (23, "LM"): "Loads the camera calibration: the file wins over the env; a non-rigid matrix stops the bridge.",
    (23, "PI"): "The image build: the nvBlox wheel must have the occupancy API; the patch is always applied.",
    # the mapper section's frame path (no chart number)
    (-1, "BP"): "The brain paused mapping for an AI task: the frame is skipped.",
    (-1, "AM"): "A person moves the arm by hand (xArm mode 2): the frame is skipped.",
    (-1, "G"): "The mapper's other gates: a pose jump, the arm moving, the keyframe spacing.",
    (-1, "I"): "Integrates the frame: keeps an undo copy of its view cone, adds it, clears the rays, applies the "
    "cell rule.",
    (-1, "S"): "The frame stays undoable until a still base pose confirms it.",
    (-1, "R"): "spatial_awareness asks for the blocks around the arm; the map version counts frames and undos.",
    (-1, "SA"): "spatial_awareness builds the arm's field from those blocks.",
}

# Shown in a decision's "Why" until the author gives one.
AUTHOR_TO_EXPLAIN = "<i>author to explain</i>"

# What each chart's flow does differently from main, shown under its "When" line.
SUMMARIES: dict[int, str] = {
    0: "On main arm moves had no world check. Now every arm command is planned against the room (2.5 cm, from the "
    "live occupancy, the lidar and AutoXing's walls), <code>set_obstacles</code> objects, the floor and the arm "
    "itself: routed around, or refused after waiting for alignment or looking first. <code>direct</code> and "
    "<code>direct_action</code> skip the room. Also: the field goes to the cloud, the mapper pauses for AI "
    "tasks and hears of base moves, buttons scan in front and at never-seen cells, <code>check_joints</code> "
    "plans without moving.",
    1: "New: <code>occupancy_field.py</code>, <code>arm_surroundings.py</code>, <code>planner_answers.py</code>, "
    "<code>arm_esdf_export.py</code>; planner <code>esdf_world.py</code>, <code>arm_spheres.py</code>, "
    "<code>surroundings.py</code>, <code>plan_single.py</code>, <code>plan_batch.py</code>, "
    "<code>replay_planner.py</code>, <code>self_check.py</code>, <code>look.py</code>; mapper "
    "<code>occupancy_requests.py</code>, <code>base_motion.py</code>, <code>block_table.py</code>; brain "
    "<code>active_mapping.py</code>, <code>mapper_pause.py</code>, <code>base_motion_notice.py</code>, "
    "<code>unknown_notice.py</code>, <code>blocked_move.py</code>; messages <code>arm_surroundings.py</code>, "
    "<code>mapper.py</code>. Design: <code>spatial_awareness/arm_awareness.md</code>.",
    2: "On main a joints move went straight to the writer and a pose to <code>pyroki_node.solve</code>. Now both "
    "ask <code>plan_move</code> (30 s), which fetches the room (chart 4), refuses goals outside the joint "
    "limits, checks the straight line (room, objects, floor, self, Billie's parts as boxes), then two-step and "
    "transit routes, else refuses (chart 3). A link starting too close may not get closer. An unreachable pose "
    "is refused, or with <code>closest</code> moved to the closest joints. <code>direct</code> checks only "
    "limits and self. Route speed keeps the tool under the controller's limit. <code>check_joints</code> plans "
    "without moving.",
    3: "New. A move refused for alignment waits up to 10 s and replans. One still blocked during occupancy mapping "
    "takes up to 2 views of the never-seen cells (<code>plan_look</code>), replans, else returns to its start "
    "and replans. A refusal by never-seen cells alone shows them with a <b>Scan the unknown</b> button; every "
    "refused target goes into the status (<code>arm.blocked_move</code>) until a move to it is planned.",
    4: "New. Each plan asks spatial_awareness for the room, sending the arm's spheres and where it ran (a direct "
    "action asks nothing). While the base stands the kept room answers at once, unless never-seen cells are "
    "within 10 cm of the arm, and is refreshed in the background every 2 s. After a base move the first plan "
    "waits for a build (mapper asked without blocking, built on a worker); nothing is built while driving. No "
    "occupancy: no room. No answer in 6 s: moves near anything refused (\"no room in time\").",
    5: "New. Built per crop: specks dropped, the floor measured, holes filled, the lidar and AutoXing's walls fill "
    "unseen space, the base cleared. Never-seen cells count as occupied except the probably-free ones in the "
    "chart; the arm's zone is kept between builds. Distances use the box rule, "
    "capped at 20 cm when aligned.",
    6: "New mapper request: the area's 2.5 cm blocks off the GPU, or <i>unchanged</i> for the held version.",
    7: "On main <code>plan</code>/<code>batch_plan</code> only solved IK. Now they fetch the room (chart 4), solve "
    "with self-collision, the room field and capsules from 4 starts (16 on main), keep 3.5 cm (one pose) or 2 "
    "cm (batch), and shorten each frame to its farthest clear point; near the room while unaligned is refused.",
    8: "On main AI actions had no world check. Now each attempt plans its move to the start (chart 2), runs inside "
    "a touch scope (fingers 8 cm, body 2 cm), has each chunk solved (chart 7), and pauses the mapper. "
    "<code>pre_action_touch</code> lets the moves around the action touch; <code>direct_action</code> drops the "
    "room. Planner waits don't count against <code>duration</code>; moves through never-seen space get the Scan "
    "the unknown notice.",
    9: "On main recordings played as recorded. Now the move to the first frame is planned (chart 2, touching only "
    "with <code>pre_action_touch</code>); untransformed replays with recorded joints go through "
    "<code>plan_replay</code> (60 s): blocked stretches rerouted, refused if a route is blocked (alignment "
    "refusals wait first, chart 3); pose frames solved 50 at a time (chart 7). <code>direct_action</code> drops "
    "the room.",
    10: "New. <code>set_obstacles</code> hands objects to spatial_awareness, which keeps map-frame ones in the map. "
    "The brain's arm guard is gone: the planner checks every move; the writer only records targets for plots.",
    11: "On main self-collision was optional and solvers had no world costs. Now every solver has self-collision, "
    "the room field and 8 capsule slots (no recompiles); a transit solver for routes compiles in the "
    "background; 4 starts (16 on main); the same model checks unsolved paths (chart 2), with base, platform and "
    "backpack as boxes (1 cm allowed). 2.5 cm room grid, no saved map; <code>planner_status</code> reports "
    "compile state.",
    12: "New. Once a second spatial_awareness builds the field for 1.5 m around the arm and writes "
    "<code>arm_esdf.bin.gz</code>, mirrored on S3 lane <code>esdf</code> for the cloud's \"ESDF (arm)\" (cloud "
    "PR #521).",
    13: "New. A 2.5 cm lidar grid (hits and misses at the lidar's height) and rays through AutoXing's walls, for "
    "the field (chart 5).",
    14: "New. <b>Start Active Occupancy Mapping</b> starts from zero, <b>Scan Around</b> scans into the current "
    "map: both wait for the planner's compile, take 6 front views at half speed (planned, chart 2), wait for "
    "frames at each, and return the arm. <b>Scan the unknown</b> aims up to 6 views at the cells that last "
    "blocked a move.",
    15: "The arm base height comes from one place for the bridge and spatial_awareness: the robot's "
    "setting, else the default. The mapper writes the robot's state for the viewers.",
    16: "On main <code>start_mapping</code> started a 1 cm TSDF with a coarse preview. Now it can start "
    "<b>occupancy</b> (native nvBlox layer, per session); TSDF defaults to 5 cm, no preview; the ZED goes to 7 "
    "m; a start can open a <b>replay capture</b> (MCAP plus SVO2). The brain checks what the mapper reports and "
    "rolls back on a mismatch. The mapper owner's reasons: 5 cm keeps a whole room's capture in memory (the "
    "capture, not the map, was the goal); 7 m is the ZED's range; the texture confidence stops a new texture seen "
    "from far from outranking an older one (it smeared).",
    17: "On main frames used one or two arm reads (clamped). Now the bridge needs both arm clocks to bracket the "
    "image, from continuous streams, and attaches a v2 sync record and camera settings. The mapper adds an "
    "arm-motion gate (fail-closed), a manual-mode gate, an optional semantic mask, per-frame capture with "
    "decisions, an occupancy path (the mapper section's rules) and a new gripper mask. The arm-motion gate and the "
    "bracketing: without SLAM a frame needs the arm's exact pose, which a moving arm doesn't give.",
    18: "New. The mapper keeps AutoXing's raw pose, map identity and 125 Hz joints, and records them into the "
    "capture's MCAP; the capture gate needs each written once.",
    19: "On main the tick refreshed the mesh and mirrored GLBs and the trajectory. In occupancy mode it exports an "
    "occupancy snapshot every ≥ 2 s to a new S3 lane; status gains representation and capture health; joint "
    "clearance is off.",
    20: "On main save flushed the GLB and ESDF, and finalize released CUDA. Now TSDF saves keep "
    "<code>map.nvblx</code> for resume; occupancy saves export the snapshot and native map and keep the GPU "
    "map; finalize stops the SVO and closes the capture with an integrity gate.",
    21: "On main publish zipped ESDF and mesh into <code>latest/operate.zip</code>. Now the bundle adds the "
    "<code>.nvblx</code> files; a capture is validated and uploaded first; an immutable "
    "<code>captures/&lt;id&gt;/operate.zip</code> is written and copied to <code>latest</code>; the brain waits "
    "300 s, as on main.",
    22: "On main only published room maps loaded, resuming into an empty backend. Now saved occupancy maps load "
    "back to the GPU after validation; room maps carry their <code>.nvblx</code> so a resume continues the real "
    "map (without it, refused); loading a room map switches to TSDF.",
    23: "New. A capture inspector, a validated TSDF replay (SVO2 depth recomputed, poses resynced, masks), an "
    "experimental occupancy replay, validation scripts; startup wires capture and SVO; the image verifies a "
    "patched nvBlox wheel.",
}

# (number, anchor, title, when, mermaid body, notes)
CHARTS: list[tuple[int, str, str, str, str, list[str]]] = [
    (
        0,
        "overview",
        "Overview",
        "Which command reaches which node, and where the arm's room comes from. <b>The only chart that repeats "
        "code drawn elsewhere</b>; a box names a chart where that chart starts.",
        """flowchart TD
  subgraph TOP[" "]
    J(["joints · pose · fold · look_up · check_joints · chart 2"]):::trig
    AI(["AI task · chart 8"]):::trig
    RP(["replay_policy · chart 9"]):::trig
    AM(["Start Active Occupancy Mapping · Scan Around · Scan the unknown · chart 14"]):::trig
  end
  SO(["set_obstacles · chart 10"]):::trig
  PS(["no command: the planner starts · chart 11"]):::trig
  TK(["no command: spatial_awareness's tick · charts 12 and 13"]):::trig
  MC(["the mapper's own commands · chart 15 · charts 16 to 23"]):::trig
  J --> BR["brain: every arm move is planned"]:::bChg
  AI --> BR
  RP --> BR
  AM --> BR
  J ~~~ SO
  BR == "plan_move · plan_look · plan · batch_plan · plan_replay" ==> PL["pyroki-planner"]:::pChg
  BR ~~~ PS
  PS --> PL
  PL == "get_arm_surroundings" ==> SA["spatial_awareness: the arm's field"]:::sNew
  SO == "set_obstacles" ==> SA
  PL ~~~ TK
  TK --> SA
  SA == "get_occupancy_blocks" ==> MP["nvblox_mapper: the live 2.5 cm occupancy"]:::mChg
  SA ~~~ MC
  MC --> MP
  SA -- "once a second" --> EX["arm_esdf.bin.gz → S3 → cloud"]:::sNew
  BR --> G["xarm_writer"]:::uChg
  BR -- "the last refused move · chart 3" --> ST["status: arm.blocked_move →<br/>the cloud's arm simulation"]:::xChg
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        1,
        "files",
        "Files",
        "Which file holds what.",
        """flowchart TD
  subgraph BR["brain"]
    BS["<b>behaviours/state.py</b>: every single move planned, the retry, the look<br/><b>play/replay_reroute.py</b>: every replay planned"]:::bChg
    BA["<b>behaviours/ai.py</b> + <b>mapper_pause.py</b>: touch, the mapper paused<br/><b>active_mapping.py</b> + <b>commands/mapping_3d.py</b>: the scan, the look<br/><b>unknown_notice.py</b>: the never-seen cells shown · <b>blocked_move.py</b><br/><b>base_motion_notice.py</b>: the mapper told before the base moves"]:::bNew
  end
  subgraph PLN["pyroki-planner"]
    PS["<b>plan_single.py</b> · <b>plan_batch.py</b> · <b>replay_planner.py</b><br/>the checks and the routes · <b>look.py</b>: the views"]:::pNew
    PE["<b>esdf_world.py</b> · <b>arm_spheres.py</b> · <b>self_check.py</b><br/><b>surroundings.py</b>: the room, the arm, itself"]:::pNew
    PR["<b>resolver.py</b> · <b>resolver_single.py</b> · <b>resolver_batch.py</b><br/>the solvers"]:::pChg
  end
  subgraph SAW["spatial_awareness"]
    SF["<b>occupancy_field.py</b>: the field<br/><b>arm_surroundings.py</b> · <b>planner_answers.py</b>"]:::sNew
    SE["<b>arm_esdf_export.py</b>: the cloud file"]:::sNew
  end
  U["<b>billie_utils/messages/arm_surroundings.py</b> · <b>mapper.py</b> · <b>pyroki_world.py</b><br/><b>arm_motion_guard.py</b> (the targets only) · <b>esdf_grid.py</b>"]:::uChg
  M["<b>nvblox_mapper/occupancy_requests.py</b>: the area's blocks off the GPU<br/><b>base_motion.py</b> · <b>block_table.py</b>: the frames"]:::mNew
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
        "A <code>joints</code>, <code>fold</code>, <code>look_up</code>, <code>pose</code> or "
        "<code>check_joints</code> command, or any single move (e.g. a scan view, chart 14). 2.1: a planned "
        "single move (also the AI's and a replay's move to their start); 2.2: a trajectory frame, queued "
        "unplanned; 2.3: a checked route.",
        """flowchart TD
  subgraph TOP[" "]
    J(["joints · fold · look_up · a single move"]):::trig
    P(["pose command · closest"]):::trig
    CJ(["check_joints: plan, never move"]):::trig
  end
  subgraph A21["2.1 · a single move"]
    SP["<b>set_state_with_arm_pose()</b><br/>behaviours/state.py"]:::bChg
    SJ["<b>set_state_with_arm_joints()</b><br/>behaviours/state.py"]:::bChg
    SJ --> CQ{{"clear_queue and not stream: a single move?"}}:::bDec
    CQ -- "yes · direct" --> PD["<b>_plan_direct()</b>: a warning, the room and objects not checked"]:::bNew
    CQ -- "yes: planned" --> PM["<b>_plan_single_move()</b><br/>behaviours/state.py · joints unknown → refused"]:::bNew
    SP --> PM
    PD == "plan_move · direct" ==> DM["<b>direct_move()</b>: only the joint limits<br/>and the self check · no refresh"]:::pNew
    PM == "plan_move · «data» PlanMoveRequest · 30 s" ==> OPM["<b>on_plan_move()</b><br/>plan_single.py · the room and objects fetched first · chart 4.1"]:::pNew
    CJF == "plan_move" ==> OPM
    OPM --> PLM["<b>plan_move()</b><br/>plan_single.py · a joints goal: Billie's<br/>body only (<b>self_check.suspended()</b>)"]:::pNew
    PLM --> GP{{"a pose goal?"}}:::pDec
    GP -- "yes" --> SPG["<b>solve_pose_goal()</b>: retries from other starts, joint 1 aimed<br/>inside the base → MOVE_BASE_COLLISION · none reaches it →<br/>MOVE_UNREACHED, or with closest the closest joints"]:::pNew
    GP -- "no" --> PJM
    SPG --> PJM["<b>plan_joint_move()</b><br/>plan_single.py · outside the joint limits → MOVE_BLOCKED"]:::pNew
    PJM --> CP["<b>check_path()</b> on the straight line<br/>room · objects · floor: <b>rule_clearances_np()</b><br/>the arm itself: <b>path_past()</b>"]:::pNew
    CP -- "clear" --> D1["MOVE_DIRECT"]:::ok
    CP -- "blocked" --> SR["<b>_best_simple_route()</b>: two moves, joint 1 or the arm first<br/><b>_simple_routes()</b> · <b>_densify()</b> · <b>check_path()</b>"]:::pNew
    SR -- "one is clear" --> D2["MOVE_DETOUR: the route's frames"]:::ok
    subgraph A23["2.3 · a checked route"]
      RT["<b>IKResolver.route()</b> transit route<br/>checked again by <b>check_path()</b>"]:::pNew
    end
    SR -- "none" --> RT
    RT -- "clear" --> D2
    RT -- "blocked" --> BL["MOVE_BLOCKED: names the link and what it hits"]:::stop
    D1 --> UR{{"the checks after the plan · chart 4.2<br/>pose not aligned and the path near anything?"}}:::pDec
    D2 --> UR
    UR -- "yes: unaligned" --> BL
    UR -- "no" --> UP["<b>_as_closest()</b>: a closest goal says how far off"]:::pNew
    BL == "reply" ==> RF["refused → chart 3: wait for the alignment, look first"]:::bNew
    DM == "reply" ==> EQ
    UP == "reply · «data» PlanMoveResponse" ==> EQ["each frame queued at <b>_route_speeds()</b>, then waited for · chart 10<br/>a closest goal: a warning with a Return button"]:::bNew
  end
  subgraph A22["2.2 · a queued frame"]
    QF["queued as it is · chart 10"]:::ref
  end
  J --> SJ
  P --> SP
  CJ --> CJF["<b>check_joints()</b> · behaviours/arm.py<br/>from the arm's joints now · no look, nothing moves<br/>answers ok, the status, the text, the route"]:::bNew
  CQ -- "no: a trajectory or stream frame" --> QF
  TW["a stream: <b>compensate_arm_for_base_motion()</b><br/>(<b>_compensate_until_twist_done()</b>): the fixed-EEF<br/>twist's 50 ms steps, solved against the room"]:::bChg
  TW --> SJ
  EQ ~~~ QF
  style TOP fill:none,stroke:none
""",
        [
            "<code>check_path()</code>'s start rule: a link (or self pair) already too close may stay as close, "
            "never closer (<code>from_start()</code>, <code>start_breach()</code>).",
            "Self pairs: gripper~link4 left out, link2~link4 may overlap 2.5 cm (8,033 recorded configurations; "
            "<code>self_check.py</code>). Base, platform and backpack are boxes a link may enter 1 cm (2 cm for "
            "link2 at the backpack; chart 11).",
            "A joints goal (joints, fold, a look view) skips the arm-against-arm pairs, whose spheres overfit "
            "folded poses (link4~link1); the pairs with Billie's body stay.",
            "No <code>plan_move</code> answer in 30 s is a refusal.",
        ],
    ),
    (
        3,
        "refused_move",
        "A refused move: wait for the alignment, look first",
        "A refused single move (chart 2.1). 3.1 also runs for a replay refused only for the alignment (chart "
        "9).",
        """flowchart TD
  T(["a single move refused · chart 2.1"]):::trig
  T --> UA{{"refused only because the pose is not aligned?<br/>the reply's unaligned"}}:::bDec
  subgraph A31["3.1 · wait to align"]
    RA["<b>retry_when_aligned()</b>: up to 10 s (ALIGN_WAIT_S),<br/><b>_pose_aligned()</b> every 0.5 s · behaviours/state.py<br/>no room in time (room_late): planned again at once"]:::bNew
    AP["spatial_awareness: aligned or not"]:::sOld
    RP["planned again: a move from where the arm is now,<br/><b>_plan_from_here()</b> · a replay as it was"]:::bNew
    R1["refused: the pose did not align in 10 s,<br/>then noted as every refusal (the box below)"]:::stop
    RA == "get_aligned_pose" ==> AP
    RA -- "aligned" --> RP
    RA -- "not aligned in 10 s" --> R1
  end
  subgraph A32["3.2 · look first"]
    LK["<b>_look_and_plan_again()</b><br/>behaviours/state.py"]:::bNew
    TO{{"not looking already, and the mapper<br/>takes occupancy frames (as chart 14 asks)?"}}:::bDec
    R2["the refusal stands"]:::stop
    PL["<b>plan_look()</b><br/>messages/pyroki_world.py · 30 s"]:::uNew
    OPL["<b>on_plan_look()</b><br/>look.py · the room fetched first · chart 4.1"]:::pNew
    PLK["<b>look.plan_look()</b>: <b>look_targets()</b> never-seen occupied cells<br/>near the move · <b>_views()</b>: <b>solve_view()</b>, <b>view_sees()</b>,<br/>each view checked from the live arm (chart 2.1's check)<br/><b>pick_views()</b>: at most 2 (LOOK_MAX_VIEWS)"]:::pNew
    NV{{"any views?"}}:::bDec
    LA["<b>look_around()</b>: each view a planned move at half speed,<br/>waiting for new mapper frames at each, as the scan (chart 14)<br/>active_mapping.py"]:::bNew
    RP2["<b>_plan_from_here()</b> again"]:::bNew
    BS["<b>_back_to_start()</b>: back to where the move started,<br/>planned again from there · the way back refused → stays"]:::bNew
    R3["still blocked from the start"]:::stop
    LK --> TO
    TO -- "no" --> R2
    TO -- "yes" --> PL
    PL == "plan_look · «data» PlanLookRequest" ==> OPL
    OPL --> PLK
    PLK == "reply · «data» PlanLookResponse" ==> NV
    NV -- "none: says why" --> R2
    NV -- "yes" --> LA
    LA --> RP2
    RP2 -- "still refused" --> BS
    BS -- "still refused" --> R3
  end
  OK["the move goes ahead · <b>blocked_move.planned()</b>:<br/>a refused move to this target is forgotten"]:::ok
  NT["refused · behaviours/state.py<br/>only never-seen cells block it: <b>unknown_notice.notify()</b>,<br/>a 3D page and a Scan the unknown button · chart 14<br/><b>blocked_move.note()</b>: the target for the robot status<br/>(arm.blocked_move: <b>blocked_move.as_json()</b> in the brain's<br/>heartbeat <b>ArmRecoveryMonitor.publish_status()</b> → status's<br/><b>build_status()</b>) · ArmMoveBlockedError"]:::bNew
  R2 --> NT
  R3 --> NT
  UA -- "yes" --> RA
  UA -- "no: blocked" --> LK
  RP -- "clear" --> OK
  RP -- "blocked, not for the alignment" --> LK
  RP2 -- "clear" --> OK
  BS -- "clear" --> OK
""",
        [
            "A look's own moves never look again (<code>_looking</code>); a failed mapper status means no look.",
        ],
    ),
    (
        4,
        "surroundings",
        "The planner's surroundings: before and after each plan",
        "4.1: before every plan (chart 2.1, chart 3.2, chart 7, chart 9). 4.2: after every planned path (chart "
        "2.1, chart 7, chart 9).",
        """flowchart TD
  T(["plan · batch_plan · plan_move · plan_replay · plan_look"]):::trig
  subgraph A41["4.1 · before the plan"]
    RA["<b>surroundings.refresh_at()</b> · pyroki-planner/surroundings.py<br/>the arm's spheres at the start: <b>surroundings.arm_at()</b><br/>where the arm ran of its last plan (<b>note_planned()</b>):<br/><b>ran_spheres()</b> · <b>_one_per_cell()</b>, kept until answered"]:::pNew
    DR{{"a direct action?<br/>yes → nothing fetched: no room, objects<br/>or floor (<b>surroundings.direct()</b>)"}}:::pDec
    RF["<b>surroundings.refresh()</b> → <b>get_arm_surroundings()</b><br/>no answer in 6 s: the last room kept, a move near anything<br/>refused as “no room in time” · error: not aligned"]:::pNew
    OG["<b>on_get_arm_surroundings()</b><br/>spatial_awareness/main.py"]:::sNew
    ST["<b>PlannerAnswers.start()</b><br/>planner_answers.py"]:::sNew
    OQ{{"a robot pose? <b>ArmSurroundings.occupancy_query()</b><br/>no → answered at once: arm-frame objects and floor"}}:::sDec
    HS{{"<b>ArmSurroundings.held_surroundings()</b>: the kept crop fits?<br/>the base where it was built (<b>_base_stayed()</b>) · ≤ 30 s or confirmed<br/>(<b>ArmSurroundings._reusable()</b>) · no never-seen cell within 10 cm of the arm (<b>_unknown_near()</b>)<br/>the arm in the lidar's plane as then (<b>_same_low()</b>) · not a scan"}}:::sDec
    HA["answered at once · <b>ArmSurroundings._remember_visited()</b>:<br/>where the arm is and has been, for the next build"]:::ok
    BG["refreshed in the background every 2 s (<b>PlannerAnswers._tick_refresh()</b>),<br/>only while the crop is around the base: <b>ArmSurroundings.kept_around()</b>"]:::sNew
    SEND["<b>request_occupancy_blocks()</b> · no waiting<br/>the held map version sent along"]:::sNew
    MP["nvblox_mapper"]:::mNew
    ONR["<b>PlannerAnswers._on_reply()</b><br/>build on a worker: <b>PlannerAnswers._build()</b>"]:::sNew
    FB{{"<b>ArmSurroundings._occupancy_crop()</b> · <b>ArmSurroundings._fetch_blocks()</b><br/>the same map and base, and nowhere new the arm has been<br/>near never-seen cells (<b>ArmSurroundings._fresh_places()</b>)?"}}:::sDec
    BF["the field built with the arm's spheres<br/>and where it has been · chart 5"]:::ref
    NO["no occupancy (not mapping, or no answer and no kept blocks):<br/>no room · <b>ArmSurroundings._current_crop()</b> has no map file<br/>(saved 3D maps are not used: USE_SAVED_3D_MAP)"]:::stop
    TK["<b>PlannerAnswers.tick()</b>: the answer goes out on the next tick"]:::sNew
    AP["<b>surroundings.apply()</b><br/><b>esdf_world.set_room()</b> · <b>world_collision.set_obstacles()</b><br/>aligned or not (<b>arm_alignment()</b>) · the never-seen cells"]:::pNew
    RA --> DR
    DR -- "no" --> RF
    RF == "get_arm_surroundings · «data» GetArmSurroundingsRequest<br/>+ arm_spheres + visited_m" ==> OG
    OG --> ST
    ST --> OQ
    OQ -- "yes" --> HS
    HS -- "yes" --> HA
    HA == "reply at once" ==> AP
    HA --> BG
    BG -. "the same request, nobody waiting" .-> SEND
    HS -- "no: the planner waits" --> SEND
    SEND == "get_occupancy_blocks · chart 6" ==> MP
    MP == "reply · «data» OccupancyBlocksResponse" ==> ONR
    ST -- "no reply in 2 s: <b>PlannerAnswers.tick()</b>" --> ONR
    ONR --> FB
    FB -- "yes: the kept crop, confirmed" --> TK
    FB -- "no: new blocks, or near never-seen cells" --> BF
    FB -- "no occupancy" --> NO
    BF --> TK
    NO --> TK
    TK == "reply · «data» ArmSurroundingsResponse" ==> AP
  end
  T2(["a path planned: plan_move · plan · batch_plan · plan_replay"]):::trig
  subgraph A42["4.2 · after the plan"]
    UR{{"<b>unaligned_refusal()</b> · surroundings.py<br/>the pose not aligned, and the path within 1 m<br/>(UNALIGNED_CLEARANCE_M) of the room or a map object?"}}:::pDec
    RU["refused as unaligned: the brain waits<br/>for the alignment · chart 3.1"]:::stop
    UP["<b>unknown_passage()</b>: logs where the path runs<br/>through never-observed space"]:::pNew
    UR -- "yes" --> RU
    UR -- "no" --> UP
  end
  T --> RA
  T2 --> UR
  AP ~~~ T2
""",
        [
            "Answers go out from the tick, never inside the mapper's reply handler, so spatial_awareness's "
            "event loop never blocks (<code>planner_answers.py</code>).",
        ],
    ),
    (
        5,
        "field",
        "The arm's field from the occupancy",
        "A new crop (chart 4) or the cloud export (chart 12).",
        """flowchart TD
  T(["a new crop: chart 4 · chart 12"]):::trig
  T --> BF["<b>build_field()</b><br/>occupancy_field.py"]:::sNew
  BF --> AS["<b>area_states()</b>: the mapper's 2.5 cm blocks<br/>occupied · free · unknown"]:::sNew
  AS --> DF["<b>drop_floating()</b>: specks of ≤ 3 cells with nothing<br/>within 5 cm → unknown"]:::sNew
  DF --> MF["<b>measure_floor()</b>: the floor's height from the camera's cells<br/>around Billie; the lidar band, walls and base placed from it"]:::sNew
  MF --> FH["<b>fill_holes()</b>: C2 holes in surfaces"]:::sNew
  FH --> LI{{"a lidar snapshot? · chart 13"}}:::sDec
  LI -- "yes" --> AL["<b>add_lidar()</b>: never-observed cells only<br/>returns occupied floor → 29.5 cm · rays free at its height<br/><b>drop_lidar_on_arm()</b>: the lidar's columns where<br/>the arm crosses its plane left out"]:::sNew
  LI -- "no" --> WA
  AL --> WA{{"AutoXing walls and the lidar correction?"}}:::sDec
  WA -- "yes" --> AW["<b>add_walls()</b>: up to 2 m in unknown cells<br/>unless the lidar saw through"]:::sNew
  WA -- "no" --> CB
  AW --> CB["<b>clear_base()</b>: Billie's base, not the arm"]:::sNew
  CB --> PF["never-observed → probably free (still unknown to the camera):<br/><b>drop_unknown_specks()</b>: free on 4 of 6 sides, nothing occupied touching<br/><b>mark_probably_free()</b>: a seen free cell within 10 cm, no occupied (<b>_ball()</b>)"]:::sNew
  PF --> NA["<b>free_near_arm()</b>: 20 cm around the arm now and where it has been<br/>«class» ArmZone, <b>ArmZone.near()</b>: the marks kept between builds of one area,<br/>only new places marked (<b>mark_near_spheres()</b>)"]:::sNew
  NA --> FA["<b>free_above_base()</b>: up to 2 m · <b>free_base_shadow()</b>: 50 cm<br/>around the base up to 80 cm · a scan: <b>free_near_base()</b> 20 cm"]:::sNew
  FA --> BD["<b>box_distances()</b>: the exact box rule, never-observed as occupied<br/>capped at 20 cm · a safe uncapped bound when not aligned"]:::sNew
  BD --> OUT["«data» Field: the crop's distances, unknown cells, a note"]:::ok
""",
        [],
    ),
    (
        6,
        "occupancy_blocks",
        "The mapper answers get_occupancy_blocks",
        "spatial_awareness asks for occupancy (chart 4, chart 12).",
        """flowchart TD
  T(["get_occupancy_blocks · «data» GetOccupancyBlocksRequest"]):::trig
  T --> BH["<b>blocks_handler()</b><br/>occupancy_requests.py · registered when the mapper starts (chart 23)"]:::mNew
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
        7,
        "frame_solve",
        "The planner solves frames",
        "Any IK request: AI frames (chart 8), a replay's playback and first pose (chart 9), pick, vision tools.",
        """flowchart TD
  subgraph TOP[" "]
    T1(["plan · «data» SolveRequest"]):::trig
    T2(["batch_plan · «data» BatchSolveRequest"]):::trig
  end
  T1 --> OP["<b>on_plan()</b><br/>pyroki-planner/main.py · one pose"]:::pChg
  T2 --> OB["<b>on_batch_plan()</b><br/>pyroki-planner/main.py · up to 50 poses"]:::pChg
  OP --> RF["the room and the objects fetched · chart 4.1<br/>inside the request's touch scope"]:::ref
  OB --> RF
  RF --> RS["IK solve with 4 starts (MAX_N_ST_PERTURB): the pose, the pull towards<br/>the recorded joints (replay) or the rest pose / joint_ref (AI),<br/>self-collision, the room and the capsules · the room at<br/>3.5 cm for one pose, 2 cm for a batch (BATCH_ROOM_MARGIN)"]:::pChg
  RS --> KS["<b>keep_solution_clear()</b> → <b>keep_moves_clear()</b><br/>plan_batch.py"]:::pNew
  KS --> CL{{"<b>_clear()</b>: the move keeps clear?<br/>room · objects · floor + the arm itself"}}:::pDec
  CL -- "yes" --> FR["the frame as solved"]:::ok
  CL -- "no" --> SH["shortened to its farthest clear point (halving)"]:::ok
  FR --> UA["the checks after the plan · chart 4.2"]:::ref
  SH --> UA
  UA == "reply · «data» SolveResponse · BatchSolveResponse" ==> BR["the brain: an AI task (chart 8),<br/>a replay (chart 9), pick, the vision tools"]:::bOld
  style TOP fill:none,stroke:none
""",
        [
            "<code>pyroki_node.solve()</code> and <code>batch_solve()</code> give up after 30 s "
            "(<code>SOLVE_TIMEOUT_S</code>) with an error; so does the replay's BufferingPlayer.",
        ],
    ),
    (
        8,
        "ai",
        "An AI task",
        "An <code>ai_task</code>.",
        """flowchart TD
  T(["AI task"]):::trig
  T --> EP["<b>execute_policy()</b><br/>behaviours/ai.py · «data» ExecutionOptions → «data» _TaskParams<br/>pre_action_touch · direct_action · first the planner's compile, waited for (chart 14)"]:::bChg
  EP --> MPA["<b>mapper_paused()</b><br/>behaviours/mapper_pause.py"]:::bNew
  MPA == "pause_mapping · then resume_mapping in finally" ==> PH["<b>MappingSession.handle_pause_mapping()</b><br/>brain_paused: new frames are skipped, the map kept"]:::mNew
  MPA -- "the mapper is down" --> WN["a warning; the task goes on"]:::stop
  EP --> RUN["<b>_Attempt.run()</b><br/>behaviours/ai.py · per attempt"]:::bChg
  RUN --> PRE["before the action: the move to starting_joints · chart 2.1<br/>inside <b>arm_touch.touching(pre_action_touch)</b> · off: 2 cm, never deeper"]:::bChg
  PRE --> ACT["the action, in <b>_Attempt.run()</b>: inside arm_touch.touching(),<br/>fingers 8 cm, gripper body 2 cm · each Gr00t chunk one pose at a time<br/>(<b>_Attempt._request_pyroki_translation()</b>) or, with the GPU, whole (<b>_Attempt._presolve_batch()</b>)"]:::bChg
  ACT --> DA["the whole attempt inside <b>arm_direct.directly(direct_action)</b>:<br/>on, the planner sees no room, objects or floor"]:::bChg
  DA == "plan · batch_plan · 30 s · chart 7" ==> SOL["the planner solves the frames · chart 7"]:::pChg
  SOL == "reply" ==> WT["<b>_Attempt._planner_wait_ended()</b>: the wait is taken out of the duration<br/>(<b>_Attempt._check_task_timeout()</b>) · moves cut short by never-seen cells:<br/><b>unknown_notice.notify_batch()</b>, once per command"]:::bNew
  WT --> Q["each frame queued without a plan · chart 2.2"]:::ref
  Q --> REV["after the attempt, if it failed or with verify_reverse: the reverse playback<br/>inside the touch scope of pre_action_touch: a single move to its first frame<br/>(chart 2.1), then the frames straight to xarm_writer's player"]:::bChg
""",
        [],
    ),
    (
        9,
        "replay",
        "replay_policy",
        "A <code>replay_policy</code>.",
        """flowchart TD
  T(["replay_policy · <b>replay_policy()</b>, after the planner's compile (chart 14)"]):::trig
  T --> IS["<b>_set_initial_state()</b>: the first pose (plan · chart 7) and the move to it · chart 2.1<br/>inside the touch scope of pre_action_touch · off: 2 cm, never deeper<br/>with options.direct_action: no room, objects or floor"]:::bOld
  IS --> RR["<b>reroute_around_obstacles()</b><br/>play/replay_reroute.py · inside the touch and direct scopes<br/>shortened by never-seen cells: the notice (chart 8)"]:::bNew
  RR --> HJ{{"transform poses or joints, and every frame has recorded joints?"}}:::bDec
  HJ -- "no" --> AS["played as recorded"]:::ok
  HJ -- "yes: always planned" --> PR["<b>plan_replay()</b><br/>messages/pyroki_world.py · touch passed along · 60 s"]:::uNew
  PR == "plan_replay · «data» PlanReplayRequest" ==> OPR["<b>on_plan_replay()</b><br/>replay_planner.py · the room fetched first · chart 4.1<br/>the checks after the plan · chart 4.2"]:::pNew
  OPR --> RER["<b>reroute_recording()</b><br/>replay_planner.py"]:::pNew
  RER --> CLR["<b>_clearances()</b> → <b>_distances()</b><br/>room · objects · floor + the arm itself, start rule"]:::pNew
  CLR --> STR{{"<b>_stretches()</b>: blocked stretches?"}}:::pDec
  STR -- "none" --> UN["REPLAY_UNCHANGED"]:::ok
  STR -- "some" --> RO["per stretch: a transit route, checked · chart 2.3"]:::ref
  RO -- "all clear" --> RTD["REPLAY_REROUTED: routes spliced in"]:::ok
  RO -- "a route blocked" --> BL["REPLAY_BLOCKED → ArmMoveBlockedError<br/>refused only for the alignment: wait for it first · chart 3.1"]:::stop
  UN --> TR{{"transform?"}}:::bDec
  RTD --> TR
  TR -- "poses" --> PP["<b>BufferingPlayer.run()</b> (from arm.replay_poses_batch()): 50 frames<br/>at a time · batch_plan · chart 7 · no answer in 30 s → error"]:::bChg
  TR -- "joints" --> PJ["<b>arm.replay_joints_batch()</b>: the recorded joints,<br/>with the routes spliced in"]:::bChg
""",
        [],
    ),
    (
        10,
        "obstacles",
        "set_obstacles and the queued moves",
        "<code>set_obstacles</code>; every joint move queued on the xArm writer; a single move's end.",
        """flowchart TD
  subgraph TOP[" "]
    T(["set_obstacles command"]):::trig
    Q(["any queued arm move"]):::trig
  end
  T --> SO["<b>world_obstacles.set_obstacles()</b><br/>commands/world_obstacles.py · a rect → <b>_with_rect()</b>"]:::bNew
  SO == "set_obstacles · «data» SetObstaclesRequest" ==> SSO["<b>on_set_obstacles()</b> → <b>ArmSurroundings.set_obstacles()</b><br/>spatial_awareness · capsules, ≤ 8 slots"]:::sNew
  SSO -- "too many capsules · bad spec" --> ER["refused, the old objects stay"]:::stop
  SSO == "reply · «data» SetObstaclesResponse" ==> RM["<b>remember_obstacles()</b><br/>for the debug plots"]:::uNew
  RM --> NX["placed at the pose of every later plan · chart 4"]:::ok
  Q --> EJ["<b>enqueue_joints()</b><br/>messages/xarm_writer.py"]:::uChg
  EJ --> RC["<b>record_move()</b>: the target noted for the debug plots<br/>arm_motion_guard.py · no check: the planner checked it (chart 2.1)"]:::uNew
  RC --> SEND["sent · a single move with <b>check_arrival</b>"]:::ok
  SEND == "joints · «data» EnqueueJointsRequest" ==> AW["<b>ArrivalWatch.arm()</b> ← <b>ArmWriter.on_enqueue_joints()</b><br/><b>ArrivalWatch.check()</b> ← <b>ArmWriter.on_tick()</b> · xarm_controller/arrival.py<br/>joints_queue_emptied held until the arm is<br/>within 2° of the target (ARRIVAL_TOLERANCE_DEG)"]:::xNew
  AW -- "not there 3 s after (ARRIVAL_TIMEOUT_S)" --> MS["move_stalled: <b>stall_message()</b><br/>the joint, how far short"]:::stop
  SEND --> WM["<b>_wait_until_moved()</b> · behaviours/state.py<br/>until joints_queue_emptied, by <b>move_deadline_s()</b>:<br/>3 × the writer's time + 5 s + 3 s"]:::bNew
  AW == "joints_queue_emptied" ==> WM
  MS == "move_stalled" ==> WM
  WM -- "move_stalled, or an xArm error" --> RE["RetriableError: Arm move stalled · xArm error"]:::stop
  WM -- "past the deadline" --> HA["<b>_hold_arm()</b>: the arm held where it is · RetriableError"]:::stop
  WM -- "emptied" --> DN["the move is done"]:::ok
  style TOP fill:none,stroke:none
""",
        [
            "On main a single move had no deadline: a stuck joint left the brain waiting.",
        ],
    ),
    (
        11,
        "planner_start",
        "The planner starts",
        "The planner starts.",
        """flowchart TD
  T(["pyroki-planner node starts"]):::trig
  T --> RB["<b>IKResolver._run_build()</b><br/>resolver.py"]:::pOld
  RB --> LAM["<b>load_arm_models()</b>: URDF, RobotCollision<br/>(gripper~link5 left out) · <b>esdf_world.load_from_environment()</b><br/>2.5 cm room grid (PYROKI_ROOM_VOXEL_M) · PYROKI_ESDF_MAP not loaded"]:::pNew
  LAM --> BS["<b>_build_solver()</b>: single · batch · transit solvers<br/>self-collision always · the room's field (2 cm for the batch) · 8 capsule slots"]:::pChg
  BS --> AR["<b>IKResolver._assign_solver_result()</b>"]:::pChg
  AR --> SC["<b>self_check.configure()</b>: compiled self distances, the joint limits<br/>the base parts as boxes: <b>BaseCollisionChecker.boxes</b> · <b>_box_pairs()</b> ·<br/><b>capsule_box_distance()</b>, 1 cm allowed into a box · used by chart 2, chart 7, chart 9"]:::pNew
  AR --> WU["warm-up: the single and batch solvers at 4 starts, then the route<br/>(transit) solver in the background, <b>warm_up_in_background()</b>;<br/>a route asked for meanwhile waits"]:::pNew
  PS(["planner_status · «data» PlannerStatusRequest → «data» PlannerStatusResponse · from the brain (chart 14)"]):::trig
  PS --> OPS["<b>_on_planner_status()</b> · world_collision.py, at once<br/>(registered in <b>world_collision.register_handlers()</b>):<br/>the IK compiled, <b>IKResolver.is_ready()</b> · the routes, <b>warmed()</b>"]:::pNew
  WU -.-> OPS
""",
        [],
    ),
    (
        12,
        "esdf_export",
        "The arm's field to the cloud",
        "spatial_awareness's tick, ≤ once a second, with a known pose.",
        """flowchart TD
  T(["spatial_awareness tick"]):::trig
  T --> DUE{{"<b>ArmEsdfExporter.due()</b>: not building,<br/>interval passed, no reply pending"}}:::sDec
  DUE -- "no" --> W["nothing this tick"]:::stop
  DUE -- "yes" --> TK["<b>ArmEsdfExporter.tick()</b><br/>arm_esdf_export.py · the area around the arm base"]:::sNew
  TK -- "<b>ArmEsdfExporter._base_moving()</b>: the base moved > 5 cm or 2°" --> BM["nothing while driving"]:::stop
  TK == "get_occupancy_blocks · no waiting · chart 6" ==> MP["nvblox_mapper"]:::mNew
  MP == "reply" ==> OR["<b>ArmEsdfExporter._on_reply()</b>"]:::sNew
  OR --> CH{{"map unchanged, the arm base moved < CROP_REUSE_M,<br/>and nowhere new the arm has been (<b>ArmEsdfExporter._places()</b>)?"}}:::sDec
  CH -- "yes" --> SK["no new file"]:::stop
  CH -- "no" --> BLD["<b>ArmEsdfExporter._build()</b> on a worker<br/>the field built (chart 5) with the arm's places, as the planner's room<br/>(<b>ArmSurroundings.arm_places()</b>, the exporter's own «class» ArmZone) → <b>encode_field()</b> → gzip<br/><b>write_atomically()</b> arm_esdf.bin.gz"]:::sNew
  BLD --> S3["the mapper's S3 mirror: lane esdf<br/><b>live_map_mirror_config()</b> · nvblox_mapper/main.py"]:::mNew
  S3 --> CL["cloud Map 3D: ESDF (arm)"]:::ok
""",
        [],
    ),
    (
        13,
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
  AD --> SN["<b>LidarGrid.snapshot()</b> · <b>StandingWalls.standing()</b><br/>used when the field is built (chart 5)"]:::ok
  AS --> SN
""",
        [],
    ),
    (
        14,
        "active_mapping",
        "Active mapping: the scan",
        "<b>Start Active Occupancy Mapping</b>, <b>Scan Around</b>, or <b>Scan the unknown</b> (chart 3's "
        "notice).",
        """flowchart TD
  subgraph TOP[" "]
    T(["Start Active Occupancy Mapping"]):::trig
    SC(["Scan Around"]):::trig
    SU(["Scan the unknown · chart 3's button"]):::trig
  end
  T --> SAM["<b>start_active_mapping()</b><br/>commands/mapping_3d.py · <b>_arm_joints()</b>: where the arm is"]:::bNew
  SAM --> SM["mapping started: occupancy, a new map,<br/>the arm to the mapping pose (after the wait below) · chart 16"]:::ref
  SC --> SN["<b>scan()</b><br/>commands/mapping_3d.py · <b>_arm_joints()</b>"]:::bNew
  SN --> TF{{"<b>taking_occupancy_frames()</b>?"}}:::bDec
  TF -- "no" --> RF["a warning: start occupancy mapping first"]:::stop
  TF -- "yes: into the current map" --> WP["<b>arm.wait_for_planner()</b> · behaviours/arm.py<br/><b>planner_status()</b> every 2 s up to 10 min · chart 11<br/>a planner without the request counts as ready"]:::bNew
  SM --> WP
  WP --> SCN["<b>_scan()</b>: the arm back to where it was"]:::bNew
  SCN --> IS["<b>initial_scan()</b><br/>behaviours/active_mapping.py"]:::bNew
  IS --> V["6 views in front of Billie, <b>_view()</b>: J1 ±30° and 0,<br/>J5 28° down or 10° up from SCAN_POSE_DEG<br/><b>set_smooth_joints()</b> at SCAN_ARM_SPEED (half) · planned · chart 2"]:::bNew
  V -- "refused" --> SK["the view skipped, a warning"]:::stop
  V -- "moved" --> WF["<b>_wait_for_frames()</b>: <b>mapper.get_status()</b> (30 s timeout)<br/>until 3 new frames, ≤ 3 s (VIEW_TIMEOUT_S)"]:::bNew
  WF --> BK["back to where the arm was (refused → stays at the last view)<br/><b>_send_scan_result()</b>: views reached, views mapped, the buttons"]:::ok
  SU --> SUN["<b>scan_unknown()</b> · commands/mapping_3d.py<br/>the cells that last blocked a move: <b>unknown_notice.last_blocked()</b><br/>none, or no occupancy frames → a warning"]:::bNew
  SUN --> WP2["the same wait for the planner"]:::bNew
  WP2 --> SV["the planner aims up to 6 views at them (plan_look, as in chart 3.2)<br/>taken one by one at half speed, waiting for frames · the arm back"]:::bNew
  SV --> BK
  style TOP fill:none,stroke:none
""",
        [
            "An abort stops the scan like any command (<code>InterruptedError</code>).",
        ],
    ),
    (
        15,
        "placement",
        "The arm in the map",
        "The bridge builds its pose config; spatial_awareness places the arm; the mapper takes a frame.",
        """flowchart TD
  subgraph TOP[" "]
    CB(["camera bridge: pose config"]):::trig
    SA(["spatial_awareness: the arm base"]):::trig
    MF(["mapper: a frame with its pose"]):::trig
  end
  CB --> GPC["<b>_get_pose_config()</b><br/>camera_autoxing_bridge/main.py"]:::cChg
  SA --> AMT["<b>arm_mount()</b><br/>arm_surroundings.py"]:::sNew
  GPC --> AH["<b>arm_base_height_m()</b> · messages/pyroki_world.py<br/>the robot's setting → 0.4405 m"]:::uNew
  AMT --> AH
  MF --> RST["<b>robot_state()</b> · nvblox_mapper/parsing.py<br/>base, joints, pose, arm mount, camera"]:::mNew
  RST --> RJ["robot.json next to the live files, for the viewers"]:::ok
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        16,
        "start_mapping",
        "Mapper: start mapping",
        "<code>start_mapping</code> (TSDF or occupancy, optionally capturing).",
        """flowchart TD
  T(["start_mapping · tsdf / occupancy · capture_replay"]):::trig
  T --> SM["<b>start_mapping()</b><br/>brain/commands/mapping_3d.py"]:::bChg
  SM == "reconfigure · 7000 mm, texture 100" ==> RC["<b>ZedCamera.reconfigure()</b><br/>camera/zed/zed_camera.py · recording → refused"]:::cChg
  RC -- "fails" --> RS["<b>_restore_camera_depth_settings()</b>, raise"]:::stop
  SM -- "move_arm_on_start" --> MP["the arm to the mapping pose,<br/>after the wait for the planner (chart 14)"]:::bChg
  SM == "start_mapping · «data» StartMappingRequest" ==> HS["<b>MappingSessionRuntime.handle_start_mapping()</b><br/>nvblox_mapper/mapping_session.py"]:::mChg
  HS --> AA{{"already mapping?"}}:::mDec
  AA -- "yes, same representation" --> DUP["duplicate start ignored"]:::stop
  AA -- "no · reset" --> RST["<b>MappingSessionRuntime.reset_mapping_state()</b>: arm_manual kept<br/>→ <b>BackendRuntime.configure_mapping()</b><br/>occupancy 2.5 cm · tsdf <b>mapping_voxel_size()</b> 5 cm"]:::mChg
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
        17,
        "frame_to_map",
        "Mapper: a camera frame into the map",
        "The bridge's tick pulls an RGB-D frame; the arm streams reach the bridge.",
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
  IF --> G{{"gates: paused · the arm moved by hand (xarm_reader/mode →<br/><b>MapperRuntime.on_arm_mode()</b>) · pose jump ·<br/><b>arm_velocity_gate()</b> · <b>should_integrate()</b>"}}:::mDec
  G -- "a gate fails" --> DEC["decision logged, not integrated"]:::stop
  G -- "semantic masking on" --> SEM["<b>LiveSemanticMasking.submit()</b> → <b>poll_semantic()</b> later"]:::mNew
  G -- "pass" --> IS["<b>_integrate_static_frame()</b> → <b>BackendRuntime.integrate_frame()</b><br/>Billie's own pixels and their depths: <b>DynamicObstacleRuntime.mask_robot_self_depth()</b><br/>the gripper's box (both backends): <b>apply_gripper_policy()</b> → <b>build_billie_gripper_body()</b>"]:::mNew
  SEM --> IS
  IS --> KIND{{"occupancy backend?"}}:::mDec
  KIND -- "yes" --> OI["<b>OccupancyBackend.integrate_depth()</b>: masked pixels never clear,<br/>Billie's own pixels clear to 5 cm before it · far and high pixels<br/>clear their rays (the mapper section): <b>OccupancyBackend._clear_ranges()</b> ·<br/><b>OccupancyBackend._high_ranges()</b> · <b>OccupancyBackend._far_free()</b> ·<br/><b>OccupancyBackend._allocate_cleared()</b> · <b>OccupancyBackend._apply_seen_rule()</b>"]:::mNew
  KIND -- "tsdf" --> NI["<b>NvbloxBackend.integrate()</b>: the gripper mask, optional depth edges"]:::mChg
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        18,
        "telemetry",
        "Mapper: telemetry for the replay",
        "AutoXing's pose, map identity and the arm joints reach the mapper.",
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
        19,
        "mapper_tick",
        "Mapper: the tick",
        "The mapper's 1 Hz tick; AutoXing's robot_pose for the clearance lane.",
        """flowchart TD
  T(["mapper tick"]):::trig
  T --> OT["<b>MapperRuntime.on_tick()</b><br/>nvblox_mapper/runtime.py"]:::mChg
  OT --> R{{"occupancy?"}}:::mDec
  R -- "yes" --> EO["<b>BackendRuntime.export_occupancy()</b>: ≥ 2 s apart, map changed<br/><b>OccupancyBackend.block_arrays()</b> (<b>OccupancyBackend.block_batches()</b>) under the lock<br/>→ <b>encode_arrays()</b> → <b>write_snapshot()</b> occupancy.bin.gz"]:::mNew
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
        20,
        "save",
        "Mapper: stop, save, finish",
        "<code>stop_mapping</code>, <code>save_mapping</code>, <code>finish_and_save_mapping</code>.",
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
  EF -- "gate failed" --> ERR["saved, but an error is raised and the map is unpublishable · chart 21"]:::stop
  EF -- "passed" --> OK["replay_capture written into the manifest"]:::ok
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        21,
        "publish",
        "Mapper: publish",
        "<code>publish_map</code> (TSDF only).",
        """flowchart TD
  T(["publish_map"]):::trig
  T == "publish_map · 300 s timeout" ==> HP["<b>MappingSessionRuntime.handle_publish_map()</b><br/>blocks the mapper's loop while it uploads"]:::mChg
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
        22,
        "load_maps",
        "Mapper: load and delete maps",
        "<code>load_saved_mapping</code> (new), <code>load_room_map</code>, <code>load_current_map</code>, "
        "<code>delete_mapping</code>.",
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
  DL --> DM["<b>delete_mapping()</b> brain → the mapping state reset (chart 16)<br/>capture finished, SVO stopped"]:::bChg
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        23,
        "offline",
        "Mapper: offline tools, startup, build",
        "Offline tools; mapper start and stop; the image build.",
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
_NARROW_LINE_CHARS = {0: 24, 2: 20, 3: 24, 4: 24, 8: 26, 10: 26, 14: 30, 16: 26, 17: 26, 20: 30, 22: 34, 23: 24}

# The mapper changes made for the arm awareness (not the merged mapper work): (change, where, why, tests).
MAPPER_OCCUPANCY_CHANGES = [
    (
        "**Occupancy blocks request.** `get_occupancy_blocks` returns the blocks in a box off the GPU, or "
        "\"unchanged\".",
        "`occupancy_requests.py` (new), `occupancy_backend.area_blocks`, `runtime.py`",
        "The arm's field is built from the live occupancy per plan.",
        "`test_occupancy_requests.py`",
    ),
    (
        "**robot.json.** Each frame writes the base pose, joints, arm mount and `T_map_from_camera`.",
        "`parsing.robot_state`, `runtime.py`, `store.py`",
        "Viewers draw Billie in the map.",
        "`test_occupancy_requests.py`",
    ),
    (
        "**Arm field to the cloud.** `arm_esdf.bin.gz` mirrored on lane `esdf` each second; never saved with a "
        "map; dot files skipped by saves.",
        "`config.py`, `main.py`, `mapping_session.py`, `artifacts.py`",
        "The cloud's ESDF (arm) view.",
        "`test_s3_mirror.py`, `test_mapper_lifecycle.py`",
    ),
    (
        "**Pause for AI tasks.** `pause_mapping` / `resume_mapping` keep the session and map; paused frames "
        "skipped; status `brain_paused`.",
        "`mapping_session.py`, `models.py`, `runtime.py`, `status_runtime.py`, `evaluation_capture.py`",
        "The AI gets the Orin.",
        "`test_mapper_lifecycle.py`",
    ),
    (
        "**One rule per cell.** Unseen 0; first view 1 (free) or 1.85 (occupied); each later view ±0.85, kept "
        "in [1, 4]; occupied above 1.8, so 100 occupied views clear after 3 free ones. Readers and saved maps "
        "keep the library's scale, so old maps load as before.",
        "`occupancy_backend.seen_rule`, `library_values`, `_scale_layer`, the block readers",
        "Old data took ~100 free views to clear and blocked the arm. Crude (v2 item 8).",
        "`test_seen_rule.py`",
    ),
    (
        "**Frames are provisional until the base proves still.** Each frame keeps an undo copy; a still pose ≥ "
        "1 s later confirms it; motion (> 2 cm, > 1°) or 5 s without a pose undoes it, newest first. Frames are "
        "dropped while the base moves, 1 s after, without base poses, and during a `base_moving` lease (2-10 "
        "s). An undo changes the map version.",
        "`base_motion.py` (new), `occupancy_backend` (`FrameUndo`, `undo`, `map_version`), "
        "`runtime.base_motion_gate`, `runtime.on_base_moving`, `status_runtime.py`, `occupancy_requests.py`",
        "Frames taken while driving smeared ghosts (AutoXing's pose comes ~1/s).",
        "`test_base_motion.py`, `test_seen_rule.py`, `test_mapper_pure_functions.py`",
    ),
    (
        "**Faster, farther frames.** Integration to 5 m; far pixels cleared their first 1 m (the whole ray "
        "since row 8); block views kept between frames (`block_table.py`), saving ~0.2 s per frame.",
        "`integration_config.OCCUPANCY_MAX_DEPTH_M`, `_far_pixels`, `_far_free`, `block_table.py` (new)",
        "~1 s per frame on billie-07; specks across open rooms never cleared.",
        "`test_seen_rule.py`, `test_block_table.py`",
    ),
    (
        "**Rays clear what they pass (new).** Far pixels clear to 5 m; pixels above 2.2 m clear to there; "
        "Billie's own pixels clear to 5 cm before the body; missing blocks along cleared rays are created.",
        "`occupancy_backend._clear_ranges`, `_high_ranges`, `_allocate_cleared`, `_far_free`, "
        "`_apply_seen_rule`; `dynamic_runtime.mask_robot_self_depth`, `backend_runtime.integrate_frame`, "
        "`models.DepthFrame`",
        "Clear-view space stayed unknown on billie-28: above the top block, past 1 m, behind the arm's pixels.",
        "`test_seen_rule.py`",
    ),
    (
        "**Smaller gripper mask (new).** `billie-gripper-body`: the gripper's box, no margin or growth (9 % of "
        "the image; v1 hid 16 %). Captures replay their own profile.",
        "`gripper_mask.build_billie_gripper_body`, `apply_gripper_policy`, `integration_config`, `replay_cli`",
        "The space in front of the gripper was never mapped.",
        "`test_gripper_mask.py`",
    ),
    (
        "**No frames in manual mode (new).** Input `xarm_reader/mode`; mode 2 skips frames (\"arm_manual\"), "
        "counted in status; survives a map reset.",
        "`config.ARM_MANUAL_MODE`, `runtime.on_arm_mode`, `runtime.integrate_frame`, `node_boundary`, "
        "`models.MapperState`, `mapping_session.reset_mapping_state`, `status_runtime`, `dataflow.py`",
        "Billie-28 mapped the operator, then refused their poses.",
        "`test_live_occupancy.py`, `test_node_boundary.py`, `test_live_semantic.py`, `test_mapper_lifecycle.py`",
    ),
    (
        "**Faster export (new).** Blocks copied in batches into two arrays and encoded at once.",
        "`occupancy_backend.block_arrays`, `block_batches`, `occupancy_stream.encode_arrays`, `write_snapshot`, "
        "`backend_runtime.export_occupancy`",
        "The per-block export halved billie-28's frame rate.",
        "`test_live_occupancy.py`",
    ),
    (
        "**Test and format fixes** for the changes above.",
        "`tests/`, `depth_edges.py`",
        "The checks pass.",
        "640 tests on a robot's environment",
    ),
]  # fmt: skip
# The mapper's frame path with the changes above (the same node styles as the charts).
MAPPER_FRAME_CHART = """flowchart TD
  subgraph TOP[" "]
    N(["brain: any base command<br/>(navigate, twist, slide, rotate, global positioning)"]):::trig
    F(["a camera frame and its pose<br/>(with the arm's joints and AutoXing's poses)"]):::trig
  end
  N --> SO["<b>OrchestratorNode.send_output()</b> → <b>base_motion_lease()</b><br/><b>BaseMotionNotice.before()</b>: a 2-10 s lease, renewed"]:::bChg
  SO == "base_moving · «data» BaseMovingRequest" ==> OB["<b>MapperRuntime.on_base_moving()</b><br/>frames dropped until the lease ends"]:::mNew
  OB --> BM
  F --> BM["<b>MapperRuntime.base_motion_gate()</b>: read the base poses,<br/>undo the suspect frames they condemn"]:::mNew
  BM -->|"announced, base moving, settling,<br/>or no base poses"| D1["dropped: base_motion"]:::stop
  BM --> BP{{"brain_paused?"}}:::mChg
  BP -->|"yes"| D2["dropped: paused"]:::stop
  BP --> AM{{"the arm in manual mode? (xarm_reader/mode)"}}:::mNew
  AM -->|"yes"| D3["dropped: arm_manual"]:::stop
  AM --> G["pose jump · arm motion · keyframe gates"]:::mOld
  G --> I["integrate_depth: copy the view cone's blocks,<br/>add the frame, clear the rays (far, high, Billie's own),<br/>apply the cell rule"]:::mChg
  I --> S["kept as a suspect with its undo<br/>(confirmed by a fresh still base pose)"]:::mNew
  I --> R["get_occupancy_blocks: the blocks in a box<br/>(version = frames + undos)"]:::mChg
  R --> SA["spatial_awareness: the arm's field"]:::sOld
  style TOP fill:none,stroke:none"""


def mapper_occupancy_section() -> list[str]:
    """The section listing every mapper change made for the arm awareness, with its frame path. Returns: Lines."""
    lines = [
        "## Mapper occupancy grid changes",
        "",
        "The mapper changes made for the arm awareness (the merged mapper work is in the charts below); each needs the "
        "mapper owner's review.",
        "",
        "| # | Change | Where | Why | Tests |",
        "|---|---|---|---|---|",
    ]
    for i, (change, where, why, tests) in enumerate(MAPPER_OCCUPANCY_CHANGES, 1):
        lines.append(f"| {i} | {change} | {where} | {why} | {tests} |")
    return lines + ["", "<details>", "<summary><b>Open the frame path</b></summary>", "", "```mermaid",
                    MAPPER_FRAME_CHART + "\n" + styles_for(MAPPER_FRAME_CHART), "```", "", "</details>", ""]  # fmt: skip


# The first chart of the PR's second comment: the nvblox mapper work merged into this branch (16-23), after the
# mapper's changes for the arm awareness.
MAPPER_FIRST = 16
# The last chart in the PR description: the system-wide list and charts 0-7 fill it to GitHub's limit (2026-11-15:
# 73.6 KB with charts 0-11), so charts 8-15 go in the first comment, after the review notes.
DESCRIPTION_LAST = 7

# Callers branch_inventory's name matching gets wrong here, fixed by hand: (file, name) -> (callers to add,
# caller prefixes to drop).
_CALLER_FIXES: dict[tuple[str, str], tuple[list[str], list[str]]] = {}

# What to do next, in v2: (what, the details). Every v2 item of the review lives here, nowhere else.
V2_ITEMS: list[tuple[str, str]] = [
    (
        "Look before acting: choose better views",
        "On billie-14 it said the cells blocking a move were all seen and took no views, while they were "
        "never-seen; on billie-28 view 1 was unreachable and skipped (chart 3.2). (1) A better algorithm to choose "
        "views, that finds better views. (2) Plan the route to each picked view while choosing (view 1 from the arm, "
        "view 2 from view 1), take the next-best view when one is blocked, and return the routes so the brain does "
        "not plan twice. (3) The full self check, arm against arm too, for joints the planner chose (look views, "
        "the scan), by a flag on the move request; the joints and fold commands keep the body-only check.",
    ),
    (
        "The kept room lags where the arm just ran by one build",
        "A request that brings new visited places is answered from the kept room before they free their zone, so a "
        "move right after can be refused and pass 3 s later (billie-28: Check said cannot move, the joints command "
        "to the same target went straight). Errs toward refusing. Wait for the build when the new places are near "
        "never-seen cells (chart 4.1).",
    ),
    (
        "Set the alignment's angle limit from recorded scans",
        "5 scans must agree within 3 cm and 2°, loose enough for rooms with few real walls in view (0.9-1.7° "
        "standing still in billie-28's office; billie-16: ±0.15°). Set it from recorded scans "
        "(<code>get_spatial_data</code>).",
    ),
    (
        "The 1 cm allowance into Billie's body boxes",
        "It applies to every move; a replay-only scope fits only untransformed replays (a transformed one is solved "
        "again). Collect how close moves come to the body boxes, then decide (chart 11).",
    ),
    (
        "Bring back saved 3D maps",
        "With <code>USE_SAVED_3D_MAP = False</code> the map-file crop (<code>_current_crop()</code>, "
        "<code>resample_crop()</code>) and 3D-map alignment never run (kept in the code). Return them with a check "
        "that the map is the right room.",
    ),
    (
        "The occupancy after an AutoXing map change",
        "Kept on purpose: AutoXing maps of one place overlap, so the occupancy stays true. After an elevator ride to "
        "another floor it does not: decide whether to delete it then. Until then, flows that change floor add a "
        "<code>delete_mapping</code> step.",
    ),
    (
        "Measure each robot's arm base height",
        "Robots use their <code>AUTOXING_BASE_TO_ARM_BASE_Z_M</code> setting, else 0.4405 m. The planner's floor "
        "guess is too noisy to set it (billie-28: -2.5 and -10 cm in one day).",
    ),
    (
        "A room fold and a corridor fold",
        "The fold (<code>ARM_FOLD</code>, used by <code>navigate_poi</code> before every drive) points the camera "
        "straight down, so after a drive the map has only the floor in front of the base. Hotels want the camera "
        "down in corridors, so the fold cannot change for everyone. Add a second, room fold that looks ahead and "
        "down (Rome uses one), used inside rooms. Until then, flows move the arm with <code>joints</code> and drive "
        "with <code>navigate_poi: fold_arm: false</code>.",
    ),
    (
        "Evidence theory for the occupancy cells",
        "The one rule per cell (the mapper section, row 5) is crude: replace it with evidence for occupied and free.",
    ),
]

# Review findings: (text) lines in the review notes, written from the reading.
_FINDINGS: list[str] = [
    "**Fixed, from main:** a nested wait dropped the outer response (a 300 s scan hang, billie-14), and an idle "
    "timeout stopped the node (<code>billie_utils/node.py</code>); a single move had no deadline (chart 10).",
    "**<code>direct</code> and <code>direct_action</code> skip the room and objects** (chart 2, chart 8, chart "
    "9): joint limits and self only, with a warning; off by default.",
    "**Every occupancy frame copies its view cone's blocks** for a possible undo, even with the base standing.",
    "**A deploy that changes a solver file re-traces the solvers** (~2 min on billie-07, 4.5 + 3 min on "
    "billie-28); unchanged code starts in ~40 s.",
    "**Two fixed parts stay capsules in the box check** (<code>NOT_BOXED</code>: <code>link_base</code>, "
    "<code>covering_cylinder_link</code>): their box corners refused clear moves on billie-07 (chart 11).",
    "<b>The items below, to the end of the mapper's, were sent to the mapper owner (2026-11-16)</b>; they stay as "
    "they are unless he asks for a change.",
    "**<code>reconfigure</code> validates before its no-change check**: a no-op raises when "
    "<code>ZED_DEPTH_MIN_MM</code> ≤ 0.",
    "<code>svo_frame_index</code> may be <code>None</code> in rgbd metadata; dora's handling unchecked.",
    "**Decided:** the calibration file wins over the env (6f941b1b), as in the brain: the map and the brain use "
    "the same calibration. Billie-07's 0.49 m arm base height moved from shared code to its "
    "<code>AUTOXING_BASE_TO_ARM_BASE_Z_M</code> setting (set it before main).",
    "**Publishing blocks the mapper for the whole upload** (<code>handle_publish_map()</code> uploads the capture on "
    "its loop): no integration, status or blocks meanwhile, so arm plans use the kept room, else none (chart 4, "
    "chart 21).",
    "**A failed capture gate makes the map unpublishable**: the save succeeds with <code>replay_gate_passed: "
    "false</code>, and publish always raises (chart 20, chart 21).",
    "**Capture folders are never deleted**: <code>EVALUATION_DIR</code> grows without bound.",
    "**Stop Mapping during a capture leaves the SVO recorder and MCAP writer running** "
    "(<code>handle_stop_mapping()</code>); the camera can't be reconfigured until Finish and Save, reset or "
    "delete.",
    "**Resuming a room map skips the voxel-size check** (<code>_resume_loaded_native_map()</code>): a 1 cm map "
    "under the 5 cm default mismatches.",
    "<code>export_occupancy(force=True)</code> can fail a save with an older export's error, and holds the "
    "integration lock for the whole GPU copy.",
    "**Load Saved Map** shows right after Finish and Save but refuses while the live map has frames; a failed "
    "load leaves the backend closed; it writes the private <code>backend._frames</code>.",
    "Occupancy finalize keeps the GPU map; <code>operate.zip</code> now carries <code>map.nvblx</code> (bigger "
    "downloads); <code>observe_auxiliary()</code> serialises every 125 Hz arm sample to JSON even without a "
    "capture.",
    "**Open question:** the capture gate needs <code>autoxing_map</code> once; <code>/map/info</code> is "
    "latched, so if it came before the mapper started the gate fails.",
    "If <code>_stop_replay_source()</code> raises in finalize, mapping is already off and nothing was saved.",
    "The function tables cover Python only; patches, shell scripts, the uploader and CI are in the system-wide "
    "table.",
]


def _fix_callers(rows: list[dict]) -> None:
    """Applies _CALLER_FIXES to the inventory rows, in place. rows: Inventory rows."""
    for r in rows:
        add, drop = _CALLER_FIXES.get((r["file"], r["name"]), ([], []))
        r["used_by"] = sorted({u for u in r["used_by"] if not u.startswith(tuple(drop))} | set(add))


def styles_for(body: str) -> str:
    """The classDef lines of the node styles a chart uses, and the frame style of its areas.

    body: The chart's mermaid text. Returns: The lines.
    """
    used = set(re.findall(r":::(\w+)", body))
    lines = [line for line in STYLES.split("\n") if line.split()[1] in used]
    areas = re.findall(r'subgraph (\w+)\["\d+\.\d+ ·', body)
    return "\n".join(lines + [f"  style {area} {AREA_STYLE}" for area in areas])


def anchor(number: int) -> str:
    """The link to a chart's heading, as GitHub names heading anchors.

    number: The chart's number. Returns: "#chart-n--title" for its "## Chart N · title" heading.
    """
    title = next(t for n, _, t, _, _, _ in CHARTS if n == number)
    text = re.sub(r"[^\w\s-]", "", f"chart {number} · {title}".lower())
    return "#" + re.sub(r"\s", "-", text)


# A mention of another chart or of an area of one: "chart 4", "chart 2.1", "charts 4.1 and 4.2".
_MENTION = re.compile(r"\b([Cc]harts?) (\d+(?:\.\d+)?(?: and \d+(?:\.\d+)?)?)")
# A chart's area: its subgraph, titled "2.1 · ...".
_AREA = re.compile(r'subgraph \w+\["(\d+\.\d+) ·')


def _chart_of(ref: str) -> int:
    """The chart a reference is in. ref: "4" (a chart) or "2.1" (an area of chart 2). Returns: The chart's number."""
    return int(ref.split(".")[0])


def mentions(text: str, number: int) -> list[str]:
    """The charts and areas a text names, in order of first mention, leaving out its own chart.

    text: A label or sentence. number: The chart it belongs to. Returns: References such as "4" or "2.1".
    """
    found: list[str] = []
    for match in _MENTION.finditer(re.sub(r"<[^>]+>", " ", text)):
        for ref in re.findall(r"\d+(?:\.\d+)?", match.group(2)):
            if _chart_of(ref) != number and ref not in found:
                found.append(ref)
    return found


def ref_anchor(ref: str, preview: bool) -> str:
    """The link to a chart or an area. GitHub cannot point inside a chart, so an area links to its chart's heading;
    the preview links "#area-2.1", which opens the chart and marks the area (AREA_SCRIPT).

    ref: "4" or "2.1". preview: For the preview page. Returns: The href.
    """
    return f"#area-{ref}" if preview and "." in ref else anchor(_chart_of(ref))


def link_mentions(text: str, number: int, preview: bool = False) -> str:
    """A sentence with each mention of another chart or area ("chart 4", "chart 2.1", "charts 4.1 and 4.2") linked.

    text: The sentence (markdown or HTML). number: Its own chart, left unlinked. preview: For the preview page.
    Returns: The sentence.
    """

    def link(match: re.Match) -> str:
        """One mention linked: "chart 2.1" whole, "charts 4.1 and 4.2" per reference. match: A _MENTION match.

        Returns: The text.
        """
        refs = re.findall(r"\d+(?:\.\d+)?", match.group(2))
        if len(refs) == 1:
            own = _chart_of(refs[0]) == number
            return match.group(0) if own else f"[{match.group(0)}]({ref_anchor(refs[0], preview)})"
        return re.sub(
            r"\d+(?:\.\d+)?",
            lambda d: d.group() if _chart_of(d.group()) == number else f"[{d.group()}]({ref_anchor(d.group(), preview)})",
            match.group(0),
        )

    return _MENTION.sub(link, text)


def click_lines(body: str, number: int) -> str:
    """Mermaid click lines making each box that names another chart or area open it (preview only).

    body: The chart's mermaid text. number: The chart's number. Returns: The lines.
    """
    lines = []
    for node, label in _NODE.findall(body):
        targets = mentions(label, number)
        if targets:
            lines.append(f'  click {node} href "{ref_anchor(targets[0], True)}" _self')
    return "\n".join(lines) + ("\n" if lines else "")


def check_one_home() -> None:
    """Fails the build when a function is drawn in two charts (from chart 2 on): each function has one home chart,
    and every other chart points to it (the user, 2026-11-01). Chart 0 (the overview) and 1 (the files) are exempt.

    Raises: ValueError naming each function drawn twice and its charts.
    """
    where: dict[str, set[int]] = {}
    for number, _, _, _, body, _ in CHARTS:
        if number < 2:
            continue
        for text in label_strings(body):
            for name in re.findall(r"([A-Za-z_][\w.]*)\(\)", re.sub(r"<[^>]+>", " ", text)):
                where.setdefault(name, set()).add(number)
    twice = {name: sorted(charts) for name, charts in where.items() if len(charts) > 1}
    if twice:
        raise ValueError(f"functions drawn in more than one chart (give each one home, point to it): {twice}")


def chart_box_notes(repo: str) -> dict[str, dict[str, str]]:
    """Every chart box's note for the preview: from the code and the chart (box_notes.py), or BOX_NOTES.

    repo: The billie-onboard checkout (its docstrings).
    Returns: Chart key (its number, "mapper" for the mapper's frame path) -> box id -> note HTML.
    """
    docs = box_notes.docstrings(repo)
    charts = [(str(n), n, when, body) for n, _, _, when, body, _ in CHARTS]
    charts.append(("mapper", -1, "A camera frame and its pose reach the occupancy mapper.", MAPPER_FRAME_CHART))
    notes: dict[str, dict[str, str]] = {}
    for key, number, when, body in charts:
        boxes = box_notes.parse_chart(body)
        for node, box in boxes.items():
            targets = mentions(box.label, number) if number >= 0 else []
            link = (ref_anchor(targets[0], True), f"Open chart {targets[0]}") if targets else None
            text = box_notes.box_note(box, boxes, when, docs, BOX_NOTES.get((number, node)), link)
            if text:
                notes.setdefault(key, {})[node] = text
    return notes


def area_script() -> str:
    """The preview's script for area links: "#area-2.1" opens chart 2, marks the 2.1 frame and scrolls to it.

    Returns: The script and its style, to append to the page.
    """
    charts = {ref: anchor(_chart_of(ref)) for _, _, _, _, body, _ in CHARTS for ref in _AREA.findall(body)}
    return f"""<style>g.cluster.area-marked > rect {{ stroke: #d97706 !important; stroke-width: 4px !important; }}</style>
<script>
  // An area link ("#area-2.1"): open its chart, mark the area's frame and scroll to it (the preview only).
  (() => {{
    const charts = {json.dumps(charts)};
    const show = (ref) => {{
      const heading = document.getElementById(charts[ref].slice(1));
      if (!heading) return false;
      let details = null;
      for (let n = heading.nextElementSibling; n && n.tagName !== "H2"; n = n.nextElementSibling) {{
        if (n.tagName === "DETAILS") {{ details = n; break; }}
      }}
      if (details) details.open = true;
      document.querySelectorAll("g.cluster.area-marked").forEach((g) => g.classList.remove("area-marked"));
      const frame = details && [...details.querySelectorAll("g.cluster")]
        .find((g) => (g.textContent || "").trim().startsWith(ref + " "));
      if (frame) {{ frame.classList.add("area-marked"); frame.scrollIntoView({{ behavior: "smooth", block: "center" }}); }}
      else heading.scrollIntoView({{ behavior: "smooth", block: "start" }});
      return true;
    }};
    document.addEventListener("click", (e) => {{
      const a = e.target.closest ? e.target.closest("a") : null;
      const href = a ? (a.getAttribute("href") || a.getAttribute("xlink:href") || "") : "";
      const m = href.match(/^#area-(\\d+\\.\\d+)$/);
      if (m && charts[m[1]] && show(m[1])) {{ e.preventDefault(); e.stopPropagation(); }}
    }}, true);
  }})();
</script>
"""


def author_decisions_section() -> list[str]:
    """The closing section: the branch's own design choices and the author's reasons. Returns: Lines."""
    lines = [
        "## Author decisions",
        "",
        "Choices made inside the branch's features, with the author's reasons.",
        "",
        "| # | Decision | Effect | Why |",
        "|---|---|---|---|",
    ]
    for i, (decision, effect, why) in enumerate(AUTHOR_DECISIONS, 1):
        lines.append(f"| {i} | {link_mentions(decision, -1)} | {link_mentions(effect, -1)} | {why or AUTHOR_TO_EXPLAIN} |")
    return lines + [""]


def v2_section() -> list[str]:
    """The last section: what to do next, in v2 (V2_ITEMS). Returns: Lines."""
    lines = ["## What to do next (v2)", ""]
    for i, (what, details) in enumerate(V2_ITEMS, 1):
        lines.append(f"{i}. **{what}.** {link_mentions(details, -1)}")
    return lines + [""]


def system_wide_section() -> list[str]:
    """The document's opening section: every change that reaches beyond the branch's features. Returns: Lines."""
    lines = [
        "## ⚠️ System-wide changes — confirm each before merging",
        "",
        "Changes that reach beyond this branch's features: other nodes, shared code, existing commands' behaviour, "
        "dependencies. The branch's own design choices are in [Author decisions](#author-decisions).",
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
            f"- **{len(unused)} functions and classes nothing calls by name** (some are entry points, handlers or "
            "test-only): "
            + ", ".join(f"{kind_marker(r)} in `{r['file'].replace('billie/', '')}`" for r in unused)
            + "."
        ]
    undrawn = [i for i in range(len(rows)) if i not in charts_of]
    lines += [
        f"- {len(undrawn)} of {len(rows)} functions and classes are in no chart: helpers of drawn ones, or removed "
        "(see *Called from*).",
        "",
    ]
    return lines


def _rows() -> list[dict]:
    """The inventory rows the review covers: everything but tests, callers fixed. Returns: The rows."""
    with open(INVENTORY) as f:
        rows = [r for r in json.load(f)["rows"] if "/tests/" not in r["file"] and "/test/" not in r["file"]]
    _fix_callers(rows)
    return rows


def _guide(count: int, page: bool = False) -> list[str]:
    """The how-to-review and how-to-read sections.

    count: Functions and classes covered.
    page: For the preview page, one document in reading order; False: the PR description and its comments.
    Returns: Lines.
    """
    where = (
        [
            "4. [Review notes](#review-notes), after chart 15: what the reading found.",
            "5. The mapper's changes and charts 16-23 ([Mapper charts](#mapper-charts)).",
        ]
        if page
        else [
            "4. Charts 8-15 are in the first comment ([More arm-awareness charts](#more-arm-awareness-charts)); the "
            "mapper's changes and charts 16-23 in the second ([Mapper charts](#mapper-charts)).",
            "5. [Review notes](#review-notes) (first comment): what the reading found.",
        ]
    )
    return [
        f"**{count} functions and classes** added, changed or removed (tests excluded), all listed in "
        "[All changed functions and classes](#all-changed-functions-and-classes).",
        "",
        "## How to review",
        "",
        "1. Confirm each system-wide change above.",
        "2. Read [Chart 0 · Overview](#chart-0--overview).",
        "3. Open one chart at a time: each starts at its trigger; every branch is a condition in the code.",
        *where,
        "6. [Author decisions](#author-decisions), at the end: the branch's own choices and why.",
        "7. [What to do next (v2)](#what-to-do-next-v2): every item left for v2.",
        "",
        "## How to read the charts",
        "",
        "| What you see | Meaning |",
        "|---|---|",
        "| Blue / purple / orange / teal / grey / green fill | brain / pyroki-planner / spatial_awareness / "
        "nvblox_mapper / camera nodes / `billie_utils` |",
        "| Thick / dashed / thin grey border | New / changed / unchanged context |",
        "| `name()` · `module.name()` | A function (the module when two share the name) |",
        "| `Class.method()` | A method |",
        "| `«class» Name` / `«data» Name` | A class / a message; on a thick arrow: request → reply |",
        "| Pill | The trigger |",
        "| Hexagon | A decision; its arrows name the cases |",
        "| Green / red box | Goes ahead / refused or failed |",
        "| Thin / thick / dashed arrow | A call / a request to another node / the other case |",
        "| Grey dashed frame **2.1 · …** | A chart area other charts run (\"→ chart 2.1\") |",
        "| Dotted box **→ chart N** | Drawn in chart N, its only home (chart 0 excepted) |",
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


def _chart_lines(preview: bool, first: int, last: int) -> list[str]:
    """Some of the charts, collapsed except the overview.

    preview: Add the click lines. first, last: The chart numbers to include (both included). Returns: Lines.
    """
    lines: list[str] = []
    for number, _, title, when, body, notes in CHARTS:
        if not first <= number <= last:
            continue
        lines += [
            f"## Chart {number} · {title}",
            "",
            f"**When:** {link_mentions(when, number, preview)}",
            "",
            f"**What changed:** {link_mentions(SUMMARIES[number], number, preview)}",
            "",
            "<details open>" if number == 0 else "<details>",
            f"<summary><b>Open chart {number}</b></summary>",
            "",
            "```mermaid",
            _wrapped(body, number) + styles_for(body),  # the preview's box notes carry the links (box_notes.py)
            "```",
            "",
            "</details>",
            "",
        ]
        targets = sorted({r for label in label_strings(body) for r in mentions(label, number)},
                         key=lambda r: [int(x) for x in r.split(".")])  # fmt: skip
        if targets:
            lines += ["Goes on in: " + " · ".join(f"[Chart {r}]({ref_anchor(r, preview)})" for r in targets), ""]
        lines += [f"- {link_mentions(note, number, preview)}" for note in notes] + ([""] if notes else [])
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
            f'Part {n} of {len(parts)}, by file. *Chart*: where it is drawn; "—": a helper of one that is.',
            "",
        ]
        out.append("\n".join(head + intro + part))
    return out


def _sections(preview: bool, page: bool = False) -> tuple[dict[str, list[str]], list[str]]:
    """The document's sections, for the PR's parts (build) or the preview page (page_markdown).

    preview: Also make the boxes that name another chart clickable. page: The guide for the one-page order.
    Returns: (section name -> its lines: head, charts_a (0 to DESCRIPTION_LAST), charts_b (to MAPPER_FIRST - 1),
        notes, mapper (its changes and charts), decisions (author decisions and v2); the function table parts).
    """
    check_one_home()
    rows = _rows()
    charts_of: dict[int, list[int]] = {}
    for number, _, _, _, body, _ in CHARTS:
        for i in sorted(chart_items(body, rows)):
            charts_of.setdefault(i, []).append(number)
    title = [
        "# arm_plan_esdf: what changed and how it flows",
        "",
        "Review guide for `arm_plan_esdf` (PR #1386, against `main`): the arm awareness (every arm command planned "
        "against the room, built from the live occupancy) and the merged nvblox mapper work (live occupancy, capture "
        "and replay, quality fixes).",
        "",
    ]
    mapper = mapper_occupancy_section() + [
        "## Mapper charts",
        "",
        "Charts 16-23: the merged nvblox mapper work (capture and replay, occupancy, quality fixes).",
        "",
    ] + _chart_lines(preview, MAPPER_FIRST, len(CHARTS) - 1)
    sections = {
        "head": title + system_wide_section() + _guide(len(rows), page),
        "charts_a": _chart_lines(preview, 0, DESCRIPTION_LAST),
        "charts_b": _chart_lines(preview, DESCRIPTION_LAST + 1, MAPPER_FIRST - 1),
        "notes": review_notes(rows, charts_of),
        "mapper": mapper,
        "decisions": author_decisions_section() + v2_section(),
    }
    return sections, _comment_parts(rows, charts_of)


def build(preview: bool = False) -> tuple[str, list[str]]:
    """The review document as markdown: the PR description and its comments, each within GitHub's limit.

    preview: Also make the boxes that name another chart clickable.
    Returns: (the PR description: system-wide changes, guide, charts 0 to DESCRIPTION_LAST, author decisions, v2;
        the comments: the review notes and the other arm-awareness charts, the mapper's changes and charts, then
        every changed function and class).
    """
    parts, tables = _sections(preview)
    more = [
        "## More arm-awareness charts",
        "",
        "Charts 8-11: AI task, replay_policy, set_obstacles, planner start. 12-13: spatial_awareness's tick. 14: scans. "
        "15: the arm in the map.",
        "",
    ]
    description = parts["head"] + parts["charts_a"] + parts["decisions"]
    # The review notes open the first comment: the description is at GitHub's limit.
    arm = parts["notes"] + more + parts["charts_b"]
    return "\n".join(description), ["\n".join(arm), "\n".join(parts["mapper"])] + tables


def page_markdown() -> str:
    """The preview page's document, in reading order (no GitHub size limit there): the system-wide changes and
    guide, charts 0-15, the review notes, the mapper's changes and charts, author decisions, v2, the function tables.

    Returns: The markdown.
    """
    parts, tables = _sections(True, page=True)
    order = ("head", "charts_a", "charts_b", "notes", "mapper", "decisions")
    return "\n".join([line for name in order for line in parts[name]] + tables)


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
    page = preview_html(page_markdown()).replace(
        "<title>Arm Planner Call Map</title>", "<title>Arm Plan ESDF Review</title>"
    ) + area_script() + box_notes.notes_script(chart_box_notes(os.path.join(_ROOT, "billie-onboard")))
    with open(OUT_HTML, "w") as f:
        f.write(page)
    print(OUT_HTML)


if __name__ == "__main__":
    sys.exit(main())
