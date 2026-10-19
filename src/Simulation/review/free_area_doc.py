"""Builds free_area_v2.md, the free_area_v2 PR review document, and an HTML preview of it.

    python -m Simulation.review.branch_inventory --repo ../billie-onboard \
        --out ../billie-onboard/out/review/free_area_v2/free_area_v2_inventory.json
    python -m Simulation.review.free_area_doc

Same method and convention as arm_planner_doc.py, whose chart-independent helpers this reuses: the charts
are written below by hand, one per trigger; the tables come from the inventory, so every function and
class the branch adds or changes is listed. Update a chart when the flow it draws changes.
"""

import json
import os
import re
import sys

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
    users_text,
    wrap_labels,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# free_area_v2 is developed in billie-onboard, so its review lives there, in out/ (ignored, never pushed).
_OUT_DIR = os.path.abspath(os.path.join(_ROOT, "..", "billie-onboard", "out", "review", "free_area_v2"))
# The inventory branch_inventory.py wrote, the two document parts (PR description, first comment), the preview.
INVENTORY = os.path.join(_OUT_DIR, "free_area_v2_inventory.json")
OUT_MD = os.path.join(_OUT_DIR, "free_area_v2.md")
OUT_COMMENT_MD = os.path.join(_OUT_DIR, "free_area_v2_functions.md")
OUT_HTML = os.path.join(_OUT_DIR, "free-area-v2-review.html")

# Node styles: fill = process (b brain, s spatial_awareness node, n nav_twist / nav_slide nodes,
# u billie_utils), border = change (New / Chg / Old).
STYLES = """  classDef trig fill:#ffffff,stroke:#1d2330,stroke-width:1.5px,color:#1d2330;
  classDef bNew fill:#dbe8fb,stroke:#2f62b0,stroke-width:3px,color:#1d2330;
  classDef bChg fill:#dbe8fb,stroke:#2f62b0,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef bOld fill:#dbe8fb,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef sNew fill:#eadff8,stroke:#6d40b8,stroke-width:3px,color:#1d2330;
  classDef sChg fill:#eadff8,stroke:#6d40b8,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef sOld fill:#eadff8,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef nNew fill:#fbe7d2,stroke:#a95a14,stroke-width:3px,color:#1d2330;
  classDef nChg fill:#fbe7d2,stroke:#a95a14,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef nOld fill:#fbe7d2,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef uNew fill:#e1efe4,stroke:#3f7a52,stroke-width:3px,color:#1d2330;
  classDef uChg fill:#e1efe4,stroke:#3f7a52,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef uOld fill:#e1efe4,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef bDec fill:#dbe8fb,stroke:#2f62b0,stroke-width:1.5px,color:#1d2330;
  classDef sDec fill:#eadff8,stroke:#6d40b8,stroke-width:1.5px,color:#1d2330;
  classDef nDec fill:#fbe7d2,stroke:#a95a14,stroke-width:1.5px,color:#1d2330;
  classDef ok fill:#d6efdc,stroke:#2e7d46,stroke-width:1.5px,color:#1d2330;
  classDef stop fill:#f8dada,stroke:#b3261e,stroke-width:1.5px,color:#1d2330;"""

# Every change that reaches beyond the free-area feature, first in the document (the review convention's
# item 0): (what changed, who it affects, risk). Filled from the whole branch diff against the merge-base.
SYSTEM_WIDE = [
    (
        "<code>force</code> = <code>default</code> means the Doppler setting, on slide / rotate / twist / fixed_eef_twist / blind_navigate_route only: the cloud form offers it as the third choice (chart 2)",
        "Those five commands",
        "<b>Regression, fixed:</b> the first version dropped any argument valued <code>\"default\"</code> on every command, so <code>set_privacy {\"mode\": \"default\"}</code> failed with <code>TypeError: missing mode</code>. Fixed in <code>d5c39ce9</code>; tested in <code>test_command_default_argument.py</code>",
    ),
    (
        "Brain reads <code>NAV_CHECK_FREE_SPACE_DEFAULT</code> with <code>environ[...]</code> at import",
        "Every robot",
        "<b>Brain does not start</b> where Doppler lacks it: add it to Doppler before deploying",
    ),
    (
        "slide / rotate / twist / fixed_eef_twist / blind_navigate_route slow down near an obstacle and stop at a stop distance. At a wall the move <b>ends OK</b> only when less than 0.2 m of it was left (<code>WALL_DONE_MAX_LEFT_M</code>: it reached its target); otherwise, and at anything else, it pauses (retry, then Resume / Cancel) instead of refusing. A twist has no known distance, so a wall always pauses it",
        "Every Flow using these moves",
        "A Flow can get a move up to 0.2 m short with no error (an INFO message says how far it got). A Flow that caught <code>AreaNotFreeError</code> now sees a pause, then <code>PausedMoveCancelledError</code> on Cancel. With <code>wait=True</code> a paused move waits for the operator with no time limit (open question). A wall stop short of the target pauses for now; next version: fail at once",
    ),
    (
        "Walls count only on the move's own path: each request also carries <code>path_points</code>, the same area built without padding, and a wall return outside it is ignored (<code>_points_in_path()</code>); any other return in the padding still counts",
        "Every checked move; the <code>is_area_free</code> command sends 0, so it is unchanged",
        "A corridor wall beside the robot no longer slows, stops or ends a move. A wall poking into the margin is not seen as a wall there",
    ),
    (
        "<code>wait=False</code> moves are watched after the call returns, slowed near an obstacle and stopped at the stop distance (<code>watch_unwaited_move()</code>)",
        "Flows that start a move and do something else meanwhile",
        "A watched move can now stop with a message (INFO when done at a wall, ERROR otherwise) while the Flow carries on. The watch ends only on its own move's done, matched by request id",
    ),
    (
        "<code>fixed_eef_twist</code> is checked like <code>twist</code>, in every direction, and gained <code>retry_timeout_s</code>",
        "Arm-compensated base moves",
        "forward / backward had no check at all before",
    ),
    (
        "nav_twist / nav_slide nodes: scoped cancels (<code>nav_twist/cancel_rotate</code>, <code>nav_slide/cancel</code> → <code>nav_slide/twist_cancel</code>), timestamp-guarded twist cancel, <code>completed_steps</code> in a cancelled twist's done, stop reason <code>Cancelled</code> vs <code>Aborted by user</code>, and an optional <code>request_id</code> in the twist / slide payload, echoed in the done; <code>dataflow.py</code> wiring",
        "Every caller of twist / slide / rotate, teleop mode changes, fixed-eef compensation",
        "Shared base-motion code; chart 13",
    ),
    (
        "The <code>is_area_free</code> command takes <code>area</code> (a rectangle readout or a polygon); <code>rect</code>, its old name, is still accepted",
        "Flows calling <code>is_area_free(rect=...)</code>",
        "Low: they keep working. Passing both, or neither, fails with a <code>ValueError</code>",
    ),
    (
        "<code>billie-utils</code> <code>IsAreaFreeRequest</code>: rectangle fields → <code>points</code> and <code>path_points</code>; new <code>IsAreaFreeInstantRequest</code>",
        "brain and the spatial_awareness node",
        "Deploy both together",
    ),
    (
        "WALL now means lidar returns on a mapped wall; a mapped wall with no returns reads FREE",
        "Every caller of <code>is_area_free</code> / <code>is_area_free_instant</code>",
        "Fewer false WALLs from map noise and shifted walls; a real wall offset from the map reads OCCUPIED",
    ),
    (
        "<code>OrchestratorNode.upload_image()</code> gained <code>privacy_safe</code>: uploads even in strict privacy",
        "Only the area-check map + lidar renders pass it",
        "Any future caller passing <code>True</code> must have no camera content",
    ),
    (
        "New required Doppler values (under base) <code>NAV_STOP_DISTANCE_WALL_M</code> (planned 0.20) and <code>NAV_STOP_DISTANCE_OBSTACLE_M</code> (planned 0.50); no default in code",
        "Every robot",
        "<b>Brain does not start</b> until both are in Doppler. Wide on purpose until the live tests (<code>spatial_awareness/free_area.md</code> §5)",
    ),
    (
        "nav_twist gets a <code>speed_cap</code> input (brain output <code>nav_twist/speed_cap</code>, <code>dataflow.py</code>), applied on every velocity step of a twist, a rotate and slide's twist bursts",
        "Every twist / slide / rotate, fixed-eef compensation",
        "A stale cap would slow a later move; every move starts by lifting it, and abort lifts it. The <code>velocity</code> command is not capped",
    ),
    (
        "Rust <code>blind_route_navigator</code> actor: new <code>set_speed_cap</code> request (drive legs m/s, rotate legs rad/s), lifted when a run ends; <code>drive_decision()</code> / <code>rotate_decision()</code> take the cap",
        "rust_host; needs a rebuild of the binary",
        "Brain sends it without waiting for the reply, so a nested wait can't swallow the run's own response",
    ),
    (
        "<code>IsAreaFreeInstantResponse</code> and <code>IsAreaFreeResponse</code> get optional fields: nearest wall / obstacle distance, frame age, frame interval",
        "brain and the spatial_awareness node",
        "Optional: an older spatial_awareness leaves them empty and brain stops as before; the pause's retry then resumes only on free",
    ),
    (
        "An <code>unknown</code> live-check answer (the lidar not delivering) no longer stops a move; it is logged and the move goes on",
        "Every checked move",
        "Decided 2026-10-18 until a second sensor exists: a lidar outage is not caught",
    ),
    (
        "New brain dependency <code>shapely</code> (<code>pyproject.toml</code>, <code>uv.lock</code>)",
        "The brain image",
        "Needs an image rebuild; v0.0.879 does not have it",
    ),
    (
        "AutoXing proxy tooling: <code>autoxing-map.html</code> area picker, <code>Cache-Control: no-store</code> on <code>/map</code> (<code>release/tunnels/</code>)",
        "Robots where <code>install-autoxing-proxy.sh</code> is re-run",
        "Low",
    ),
]

# What each chart's flow does differently from main, shown under its "When" line.
SUMMARIES = {
    0: "On main the five moves drove the base with no obstacle check, and <code>is_area_free</code> was a manual command on a rectangle. Now each move reads the area it is about to cover before starting, a live check re-reads what is still ahead while the base drives, the move slows near something and stops at a stop distance: done at a wall it had nearly reached, otherwise paused (retry, then ask the operator).",
    1: "Three new brain files split out of <code>_nav.py</code>: <code>area_check.py</code> (force, the checks, the pause), <code>swept_area.py</code> (the polygons), <code>event_wait.py</code> (waits that survive nested waits). <code>_nav.py</code> keeps the moves; <code>fixed_eef_twist.py</code> now reuses twist's leg loop.",
    2: "On main <code>force</code> was true or false only. Now, on the five moves, it also takes <code>default</code>: the dispatch turns it into <code>None</code>, which the move resolves to the <code>NAV_CHECK_FREE_SPACE_DEFAULT</code> Doppler setting. Any other argument's <code>\"default\"</code> is passed through untouched.",
    3: "On main <code>slide()</code> sent nav_slide/slide and waited. Now: one instant read of the whole distance before moving; a live check on the distance still ahead, re-read from the live pose; near something it slows down (chart 8). On a stop it cancels, drains this leg's own done, measures how far it got, then ends OK at a wall with less than 0.2 m left, or pauses (chart 9) and resumes with the rest. <code>wait=False</code> is watched until its own done (matched by request id) and the call returns even if the move ended before the robot moved. Slide errors carry the distance travelled. <code>bump()</code> passes <code>force=True</code>.",
    4: "Same pattern as slide, in degrees: a read of the whole sweep before moving; a live check on the arc still left to turn, from the live pose; near something it slows down (chart 8). On a stop: a cancel (new nav_twist/cancel_rotate), its response read from <code>node.last</code> if it already came, then done at a wall with under 0.2 m of arc left at the footprint's corner, or a pause on the arc left and a new leg. <code>wait=False</code> is watched until its own response, and a wall stops it with an error.",
    5: "On main <code>twist()</code> sent the steps and waited. Now the checked area depends on the direction (full circle, or 0.5 m ahead of the live pose), read once before moving and live while moving, and it slows near something (chart 8). A stop cancels, reads <code>completed_steps</code> from nav_twist (only if the leg's twist was sent), pauses and re-sends only the steps left. Steps have no known distance, so a wall always pauses a twist. The leg loop is shared with chart 6.",
    6: "On main it sent the twist and compensated the arm, with no check. Now it is checked exactly like twist and runs the same leg loop; the arm compensation follows the capped speed. Each leg re-captures the fixed-eef reference before sending its twist, so a resumed leg starts from where the base stopped; a stop that comes while the leg is still waiting for the arm pauses without waiting for a done.",
    7: "On main the route was driven with no check. Now one instant read of the corridor ahead before the run starts; a live check follows a 1.2 m corridor along the route while it drives, slowing the run through a new speed cap in the Rust actor. A wall stop ends the run OK only within 0.2 m of the route's end (path length left, <code>_route_left_m()</code>); any other stop cancels the run, pauses (chart 9) and re-sends the same route, which the actor resumes from where the robot stopped.",
    8: "New. A tick handler polls <code>is_area_free_instant</code> every 0.3 s while a checked move runs, with the move's padding, so a wall beside the path does not count. Each answer with something in the path sets a speed cap from the distance left before the stop distance (<code>safe_speed()</code>; an obstacle wins a tie with a wall). With no room left twice in a row: done at a wall when under 0.2 m of the move is left, otherwise a pause. Unknown and blocked answers don't slow or stop it.",
    9: "New. The pause shared by every move: post the map image (uploaded even in strict privacy, Stay Over: it is only the map and lidar points; a failure is reported to the operator), retry while stopped, then Resume / Cancel. The retry resumes when the area is free, or when what is there leaves room before its stop distance (the careful reply now carries the distances) -- then already capped to the safe speed.",
    10: "On main the request was a rectangle and the verdict free / occupied / moving / blocked / unknown. Now any polygon, with an optional unpadded path (<code>path_points</code>) outside which walls don't count; a new WALL verdict when lidar returns on the path land on a mapped wall (a mapped wall with no returns stays free: the lidar wins); the blocked-view check is exact for any polygon; the image shows the frame that decided, and an occupied / wall reply carries that frame's distances. The command takes <code>area</code>, or <code>rect</code>, its old name.",
    11: "New. The single newest frame only, no voting, for the live check and the read before moving: unknown when there is no map or the frame is older than 2.5 s. An occupied / wall reply also carries the distance from the footprint to the nearest wall return on the path and the nearest other return (<code>nearest_returns()</code>), the frame's age and the measured frame interval (<code>_frame_interval_s()</code>).",
    12: "On main pose and lidar were each checked for freshness on their own. Now a frame also needs them taken close together: 1 to 2 s apart, tighter the faster the robot moves. Returns inside the robot's own footprint are dropped.",
    13: "On main the only way to stop a twist, slide or rotate was a system abort. Now each has its own cancel; a cancelled twist reports how many steps it completed; a cancel that arrives before the twist it was meant for still stops it; a speed cap from brain holds every velocity step down near an obstacle; and a twist's or slide's done echoes the request id it was sent with.",
}

# (number, anchor, title, when, mermaid body, notes)
CHARTS = [
    (
        0,
        "overview",
        "Overview",
        "Which move reaches which check, and which node answers it. The <code>is_area_free</code> command goes straight to chart 10.",
        """flowchart TD
  A(["slide · rotate · twist · fixed_eef_twist ·<br/>blind_navigate_route commands"]):::trig --> D["<b>OrchestratorNode.execute_command()</b><br/>brain/node.py · force = default · chart 2"]:::bChg
  D --> MV["the moves · charts 3 to 7<br/>one read before moving · chart 11"]:::bChg
  MV --> LC["live check while moving<br/><b>_install_debounced_area_check()</b> · chart 8"]:::bNew
  MV -- "the read before moving hits" --> P
  LC -- "two bad answers in a row" --> P["<b>_pause_and_await_resume()</b> · chart 9<br/>retry, then Resume / Cancel"]:::bNew
  MV == "twist · slide · rotate · cancel" ==> NV["nav_twist · nav_slide nodes · chart 13"]:::nChg
  LC == "is_area_free_instant" ==> SA
  P == "is_area_free · while stopped" ==> SA
  SA["spatial_awareness node<br/>verdicts from buffered lidar frames · chart 12"]:::sChg
""",
        [],
    ),
    (
        1,
        "files",
        "Files",
        "Which file holds what.",
        """flowchart TD
  subgraph BRAIN["brain"]
    ND["<b>node.py</b> · <b>spec.py</b>: force = default · cloud form · image upload<br/><b>commands/spatial_awareness.py</b>: the is_area_free command"]:::bChg
    MOV["<b>_nav.py</b> · <b>fixed_eef_twist.py</b> · <b>blind_route_navigator.py</b><br/>the moves and the twist leg loop"]:::bChg
    AC["<b>area_check.py</b>: force · read before moving ·<br/>live check · pause · wait=False watch"]:::bNew
    HLP["<b>swept_area.py</b>: the polygon each move covers<br/><b>event_wait.py</b>: waits that survive nested waits"]:::bNew
  end
  U["<b>billie_utils/messages/spatial_awareness.py</b><br/>requests · replies · polygon helpers"]:::uChg
  subgraph SAN["spatial_awareness"]
    MAIN["<b>main.py</b>: frames · request handlers · image"]:::sChg
    AFR["<b>area_free.py</b>: the verdict"]:::sChg
  end
  NV["<b>src/nav_twist/</b> · <b>src/nav_slide/</b><br/>drive and cancel the base"]:::nChg
  ND --> MOV
  MOV --> AC
  MOV --> HLP
  AC --> U
  U ==> MAIN
  MAIN --> AFR
  MOV ==> NV
  style BRAIN fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
  style SAN fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [],
    ),
    (
        2,
        "force_default",
        "force = default",
        "One of the five moves arrives with <code>force</code> = <code>default</code>, from the cloud, a Flow or <code>run_code</code>; and the cloud's command form for those moves.",
        """flowchart TD
  subgraph TOP[" "]
    A(["a move command arrives<br/>cloud · Flow · run_code"]):::trig
    F(["the cloud asks for the command list"]):::trig
  end
  A --> E["<b>OrchestratorNode.execute_command()</b><br/>brain/node.py"]:::bChg
  E --> FD{{"force = 'default' on one of the five moves?<br/>yes → force = None"}}:::bDec
  FD --> CO["the arguments are checked as on main · the handler runs"]:::bOld
  CO --> RF["<b>_resolve_force()</b> · area_check.py<br/>None → the NAV_CHECK_FREE_SPACE_DEFAULT setting"]:::bNew
  F --> R["<b>_render_action_spec()</b> · brain/spec.py<br/>force on the five moves: true / false / default"]:::bChg
  style TOP fill:none,stroke:none
""",
        [
            "The five moves are slide, rotate, twist, fixed_eef_twist and blind_navigate_route (<code>FORCE_DEFAULT_COMMANDS</code>, spec.py). Any other argument's <code>\"default\"</code> is passed through untouched."
        ],
    ),
    (
        3,
        "slide",
        "slide",
        "The <code>slide</code> command, a Flow or <code>run_code</code>, and <code>bump()</code>, which passes <code>force=True</code>.",
        """flowchart TD
  A(["slide command · Flow · <b>bump()</b> (force=True)"]):::trig --> S["<b>slide()</b> · brain/navigation/_nav.py<br/><b>_validate_direction()</b> · <b>_resolve_force()</b> (area_check.py)"]:::bChg
  S --> F{{"force?"}}:::bDec
  F -- "no" --> PRE["read before moving: <b>_check_before_move()</b> (area_check.py)<br/>the whole distance: <b>_current_pose()</b> → <b>_slide_swept_points()</b> (swept_area.py)<br/>room left: start slow · speed cap: <b>_twist_governor()</b><br/>no room at a wall, under 0.2 m to go → doesn't start, ends OK"]:::bNew
  PRE -- "no room otherwise" --> P
  PRE -- "room left" --> LEG
  F -. "yes: no checks" .-> LEG
  LEG["per leg: <b>wait_for_current_coordinates()</b> · a new request_id ·<br/>nav_slide/slide · chart 13"]:::bChg
  LEG --> W{{"wait?"}}:::bDec
  W -- "no, checked" --> WU["<b>watch_unwaited_move()</b> (area_check.py) until its own nav_slide/done,<br/><b>_done_of()</b> by request_id; a hit cancels and reports<br/><b>optional_background_wait()</b>: returns once moving, or on that done"]:::bNew
  W -- "yes" --> MON["<b>monitored_move()</b> + live check · chart 8: the distance still ahead,<br/><b>_leg_area_ahead()</b> · <b>_leg_left_m()</b> from the live pose<br/><b>optional_background_wait()</b> (event_wait.py) → on_slide_done"]:::bNew
  MON -- "done" --> OK[["slid: the distance travelled is reported"]]:::ok
  MON -- "no movement / timeout" --> X2[["«class» SlideNoMovementError · «class» SlideTimeoutError<br/>with the distance travelled"]]:::stop
  MON -- "live check stop · chart 8" --> CAN["nav_slide/cancel · this leg's done: node.last, else<br/><b>wait_for_event_via_handler()</b> (sent after the leg) · measure how far it got<br/>at a wall, under 0.2 m to go → ends OK, says how far"]:::bNew
  CAN -- "otherwise" --> P["<b>_pause_and_await_resume()</b> · chart 9<br/>on the distance left"]:::bNew
  P -- "resumed" --> LEG
  P -- "cancelled" --> X[["«class» PausedMoveCancelledError"]]:::stop
""",
        [],
    ),
    (
        4,
        "rotate",
        "rotate",
        "The <code>rotate</code> command, a Flow or <code>run_code</code>.",
        """flowchart TD
  A(["rotate command · Flow"]):::trig --> R["<b>rotate()</b> · brain/navigation/_nav.py<br/><b>_validate_direction()</b> · <b>_resolve_force()</b>"]:::bChg
  R --> F{{"force?"}}:::bDec
  F -- "no" --> PRE["read before moving: <b>_check_before_move()</b><br/>the whole sweep: <b>_rotate_swept_points()</b> → <b>_wrap_angle_rad()</b> · <b>circle_points()</b><br/>the target heading is fixed here · distance left: the corner's arc<br/>no room at a wall, under 0.2 m of arc → doesn't start, ends OK"]:::bNew
  PRE -- "no room otherwise" --> P
  PRE -- "room left, capped" --> W
  F -. "yes: no checks" .-> W
  W{{"wait?"}}:::bDec
  W -- "no" --> NW["one request, <b>request_rotate()</b> · then, checked: <b>watch_unwaited_move()</b><br/>until this request's nav_twist/response; a hit sends cancel_rotate<br/>(a wall too: nothing tracks the turn, so it stops with an error)"]:::bChg
  W -- "yes" --> LEG["per leg: «data» RotateRequest → nav_twist/rotate · chart 13<br/>tick: _sample_rotation() sums the turned degrees"]:::bChg
  LEG --> MON["<b>monitored_move()</b> + live check · chart 8: the arc still left,<br/><b>_leg_sweep_left()</b> · <b>_leg_left_deg()</b> from the live pose<br/><b>wait_for_event_via_handler()</b> nav_twist/response"]:::bNew
  MON -- "response" --> OK[["rotated"]]:::ok
  MON -- "live check stop · chart 8" --> CAN["nav_twist/cancel_rotate · the response from node.last, else waited for ·<br/>degrees left, or the same absolute heading<br/>at a wall, under 0.2 m of arc left → ends OK"]:::bNew
  CAN -- "otherwise" --> P["<b>_pause_and_await_resume()</b> · chart 9<br/>on the arc left, from where it stopped"]:::bNew
  P -- "resumed: a new leg" --> LEG
  P -- "cancelled" --> X[["«class» PausedMoveCancelledError"]]:::stop
""",
        [],
    ),
    (
        5,
        "twist",
        "twist",
        "The <code>twist</code> command, a Flow or <code>run_code</code>.",
        """flowchart TD
  A(["twist command · Flow"]):::trig --> T["<b>twist()</b> · brain/navigation/_nav.py<br/><b>_validate_direction()</b> · <b>_resolve_force()</b>"]:::bChg
  T --> F{{"force?"}}:::bDec
  F -- "no" --> AR["<b>_twist_area()</b> · brain/navigation/_nav.py<br/>left / right: a full circle, <b>_twist_rotation_points()</b><br/>forward / backward: 0.5 m ahead of the live pose, <b>_slide_swept_points()</b>"]:::bNew
  AR --> PRE["read before moving: <b>_check_before_move()</b> · room left: start slow ·<br/>no room, a wall too (no known distance): pause, chart 9<br/>wait=False: <b>watch_unwaited_move()</b> until its own nav_twist/done, <b>_done_of()</b>"]:::bNew
  F -. "yes: no checks" .-> LEGS
  PRE --> LEGS["<b>_run_twist_legs()</b> · brain/navigation/_nav.py · shared with chart 6"]:::bNew
  LEGS --> MON["<b>monitored_move()</b> · live check on the area (wait=True) · chart 8"]:::bNew
  MON --> RL["one leg: <b>send_twist()</b> with the request_id → nav_twist/twist · chart 13<br/>twist_sent() · <b>optional_background_wait()</b> nav_twist/done"]:::bChg
  RL -- "done" --> OK[["twisted"]]:::ok
  MON -- "live check stop · chart 8" --> CS["twist sent? <b>_completed_steps_of_cancelled_twist()</b>:<br/>completed_steps from nav_twist/done · not sent: 0 steps, no wait"]:::bNew
  CS -- "steps left" --> P["<b>_pause_and_await_resume()</b> · chart 9"]:::bNew
  P -- "resumed: only the steps left" --> LEGS
  P -- "cancelled" --> X[["«class» PausedMoveCancelledError"]]:::stop
""",
        [],
    ),
    (
        6,
        "fixed_eef_twist",
        "fixed_eef_twist",
        "The <code>fixed_eef_twist</code> command: the base twists while the arm keeps the end-effector still in the map.",
        """flowchart TD
  A(["fixed_eef_twist command · Flow"]):::trig --> FE["<b>fixed_eef_twist()</b><br/>brain/behaviours/fixed_eef_twist.py · <b>_resolve_force()</b>"]:::bChg
  FE --> F{{"force?"}}:::bDec
  F -- "no" --> AR["<b>_twist_area()</b> → <b>_check_before_move()</b><br/>the same area and read as twist · chart 5"]:::bNew
  F -. "yes: no checks" .-> WARM
  AR --> WARM["the arm's pose · warm the IK solver"]:::bOld
  WARM --> LEGS["<b>_run_twist_legs()</b> · chart 5<br/>live check unless force · the pause and resume of chart 9"]:::bNew
  LEGS --> RL["one leg: capture the fixed-eef reference again ·<br/><b>send_twist()</b> · twist_sent() · compensate the arm until nav_twist/done"]:::bChg
  RL -- "done" --> OK[["twisted, end-effector held"]]:::ok
  RL -- "the arm fails" --> X[["cancel the twist · the command fails"]]:::stop
""",
        [
            "While paused the base is stopped and the arm stops compensating, so the end-effector stays put; the next leg takes its reference from there."
        ],
    ),
    (
        7,
        "blind_navigate_route",
        "blind_navigate_route",
        "The <code>blind_navigate_route</code> command (driving a recorded route with AutoXing's obstacle avoidance off), a Flow or <code>run_code</code>.",
        """flowchart TD
  A(["blind_navigate_route command"]):::trig --> C["<b>commands.navigation.blind_navigate_route()</b><br/>brain/commands/navigation.py · passes force, retry_timeout_s, padding"]:::bChg
  C --> B["<b>blind_route_navigator.blind_navigate_route()</b><br/>brain/navigation/blind_route_navigator.py · holds twist ownership"]:::bChg
  B --> F{{"force?"}}:::bDec
  F -- "no" --> PRE["read before the run: <b>_check_route_before_moving()</b> → <b>_check_before_move()</b><br/>the corridor ahead, <b>_route_lookahead_points()</b> → <b>_route_points()</b> ·<br/><b>_arc_length_at_closest_point()</b> · <b>_polyline_between()</b> · past the end: no read<br/>no room at a wall, under 0.2 m of route left → doesn't start"]:::bNew
  PRE -- "no room otherwise" --> P
  PRE -- "room left" --> L["<b>_install_live_area_check()</b> · chart 8<br/>the same 1.2 m corridor, <b>_corridor_ahead()</b>, re-read<br/>from the live pose, with its unpadded path ·<br/>route left: <b>_route_left_m()</b>, the path length"]:::bNew
  F -. "yes" .-> RUN
  L --> RUN["<b>monitored_move()</b> → follow_route to the rust_host actor<br/>slowed by the actor's set_speed_cap<br/>a wall, under 0.2 m of route left: «class» _StoppedAtWall → ends OK"]:::bChg
  RUN -- "the actor reports the end" --> OK[["route done, or failed with the actor's error"]]:::ok
  RUN -- "any other stop: «class» BlindRouteAreaNotFreeError" --> P["cancel the run · <b>_pause_and_await_resume()</b> · chart 9 (posts the image)<br/><b>_completed_description()</b> · <b>_remaining_description()</b> →<br/><b>_distance_to_final_waypoint()</b>"]:::bNew
  P -- "resumed: the same route again" --> RUN
  P -- "cancelled" --> X[["«class» PausedMoveCancelledError"]]:::stop
""",
        [],
    ),
    (
        8,
        "live",
        "Live check",
        "Every brain tick while a checked move runs: inside the waits of charts 3 to 7, or after a <code>wait=False</code> call returned (<code>watch_unwaited_move()</code>, which cancels the base itself: INFO when done at a wall, ERROR otherwise).",
        """flowchart TD
  A(["brain tick while a checked move runs"]):::trig --> C["<b>_install_debounced_area_check()</b> · its tick handler<br/>brain/navigation/area_check.py · every 0.3 s, one request at a time"]:::bNew
  C --> G["the area still ahead, from the live pose (fixed for twist left / right) ·<br/>with the move's padding: a wall in that margin doesn't count"]:::bNew
  G == "is_area_free_instant · «data» IsAreaFreeInstantRequest →<br/>«data» IsAreaFreeInstantResponse" ==> H["<b>on_is_area_free_instant()</b> · chart 11<br/>spatial_awareness/main.py"]:::sChg
  H == "reply: verdict, distances, frame timing" ==> V{{"verdict<br/>free → cap lifted · blocked, unknown → logged, keep going"}}:::bDec
  V -- "occupied / wall" --> L["<b>speed_limit()</b> → <b>safe_speed()</b><br/>room = distance − the Doppler stop distance (wall / other) ·<br/>an obstacle may close 0.5 m/s while the robot can't see · a tie: the obstacle"]:::bNew
  L --> S{{"safe speed ≥ 0.03 m/s?"}}:::bDec
  S -- "yes" --> CAP["<b>SpeedGovernor.set()</b> → nav_twist/speed_cap, or the route's set_speed_cap · chart 13"]:::bNew
  S -- "no, twice in a row" --> W{{"a wall, and under 0.2 m of the move left?<br/><b>_done_at_wall()</b> · twist: never"}}:::bDec
  W -- "yes" --> OK[["<b>_raise_stopped_at_wall()</b> → «class» _StoppedAtWall:<br/>the move ends OK, <b>wall_stop_message()</b> says how far it got"]]:::ok
  W -- "no" --> P[["<b>_raise_area_check_interrupted()</b> → «class» _AreaCheckInterrupted<br/>(route: «class» BlindRouteAreaNotFreeError) · the move pauses · chart 9"]]:::stop
""",
        [
            "Lidar frames arrive about once a second, so two polls 0.3 s apart can read the same frame: “twice in a row” is not always two independent looks.",
            "An answer without distances (an older spatial_awareness) stops after two occupied / wall answers, as before.",
        ],
    ),
    (
        9,
        "pause",
        "Pause and resume",
        "The read before moving finds no room left (charts 3 to 7), or the live check stops a waiting move (chart 8), and the move is not done: anything but a wall, or a wall with more than 0.2 m of the move still to go.",
        """flowchart TD
  subgraph TOP[" "]
    A1(["the read before moving: no room, not done<br/><b>_check_before_move()</b>"]):::trig
    A2(["the live check stopped a move, not done · chart 8"]):::trig
  end
  A1 --> P["<b>_pause_and_await_resume()</b><br/>brain/navigation/area_check.py"]:::bNew
  A2 --> P
  P --> IMG["<b>post_area_check_image()</b> → <b>OrchestratorNode.upload_image()</b><br/>privacy_safe: uploaded and linked even in strict privacy<br/>a failure: an ERROR message, the pause goes on (an abort still aborts)"]:::bNew
  IMG == "is_area_free · chart 10" ==> SA["<b>on_is_area_free()</b><br/>spatial_awareness/main.py"]:::sChg
  IMG --> RT{{"retry_timeout_s set?<br/>(10 s by default)"}}:::bDec
  RT -- "yes" --> POLL["re-check once a tick while stopped, up to the timeout · Cancel shown"]:::bNew
  POLL == "is_area_free" ==> SA
  POLL -- "free, or room left: <b>speed_limit()</b>" --> GO[["the move goes on with what is left ·<br/>room left: capped first, <b>SpeedGovernor.set()</b>"]]:::ok
  POLL -- "timed out" --> ASK
  RT -- "no" --> ASK["Resume / Cancel · waits for the operator, no time limit"]:::bNew
  ASK -- "Resume" --> GO
  ASK -- "Cancel" --> X[["«class» PausedMoveCancelledError, an «class» AreaNotFreeError<br/>with the progress made"]]:::stop
  style TOP fill:none,stroke:none
""",
        [
            "The image comes from the careful multi-frame check, so it can read MOVING (just stopped) or FREE while the move paused on the instant check."
        ],
    ),
    (
        10,
        "is_area_free",
        "is_area_free",
        "The <code>is_area_free</code> command, and the pause's retry and image (chart 9).",
        """flowchart TD
  subgraph TOP[" "]
    C(["is_area_free command"]):::trig
    PZ(["pause: retry and image · chart 9"]):::trig
  end
  C --> CMD["<b>is_area_free()</b> · brain/commands/spatial_awareness.py<br/>area, or rect (old name): <b>_parse_area()</b> →<br/><b>_parse_polygon()</b> or <b>rectangle_points()</b>"]:::bChg
  CMD --> REQ
  PZ --> REQ["<b>spatial_awareness.is_area_free()</b><br/>billie_utils/messages/spatial_awareness.py"]:::uChg
  REQ == "is_area_free · «data» IsAreaFreeRequest →<br/>«data» IsAreaFreeResponse" ==> H["<b>on_is_area_free()</b><br/>spatial_awareness/main.py"]:::sChg
  H --> CAF["<b>check_area_free()</b> · up to 6 buffered frames<br/>spatial_awareness/area_free.py"]:::sChg
  subgraph AF["area_free.py"]
    ST{{"robot still since the frame before?<br/><b>is_robot_stationary()</b> → <b>_wrap_angle()</b>"}}:::sDec
    OC{{"a return inside, away from mapped walls?<br/><b>is_area_occupied_this_frame()</b> · <b>_points_in_polygon()</b> · <b>_nearest_wall_px()</b>"}}:::sDec
    WL{{"a return on the path, on a wall?<br/><b>is_area_wall_this_frame()</b><br/><b>_points_in_path()</b>"}}:::sDec
    BL{{"something closer hides the area?<br/><b>is_area_blocked()</b> → <b>_nearest_boundary_distance()</b> · <b>_nearest_crossing_distance()</b>"}}:::sDec
  end
  CAF --> ST
  ST -- "yes" --> OC
  OC -- "no" --> WL
  WL -- "no" --> BL
  ST -- "no: excluded, moving" --> VOTE
  OC -- "yes: occupied vote" --> VOTE
  WL -- "yes: wall vote" --> VOTE
  BL -- "yes: excluded, blocked" --> VOTE
  BL -- "no: empty vote" --> VOTE
  VOTE{{"«class» AreaStatus: any occupied → OCCUPIED · else any wall → WALL ·<br/>fewer than 3 usable → MOVING / BLOCKED / UNKNOWN · else FREE"}}:::sDec
  VOTE --> IMG["<b>_render_area_check_image()</b><br/>map crop + the deciding frame's lidar points ·<br/>occupied / wall: that frame's <b>nearest_returns()</b>"]:::sChg
  IMG == "reply · status, image, distances" ==> CMD
  style TOP fill:none,stroke:none
  style AF fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [
            "“On a mapped wall” means within 2 map pixels (<code>WALL_DILATE_PX</code>, about 10 cm). A wall drawn on the map where the lidar sees nothing reads FREE.",
        ],
    ),
    (
        11,
        "instant",
        "is_area_free_instant",
        "The read before moving and every live-check poll (charts 8 and 9).",
        """flowchart TD
  A(["<b>spatial_awareness.is_area_free_instant()</b><br/>billie_utils/messages/spatial_awareness.py"]):::trig == "«data» IsAreaFreeInstantRequest (points, path_points)" ==> H["<b>on_is_area_free_instant()</b><br/>spatial_awareness/main.py"]:::sChg
  H --> I["<b>_instant_area_status()</b> · the newest frame only, no voting<br/>spatial_awareness/main.py · occupied / wall replies add <b>nearest_returns()</b><br/>(area_free.py, walls on the path only) and <b>_frame_interval_s()</b>"]:::sNew
  I --> M{{"a map, and a frame under 2.5 s old?<br/>no → unknown"}}:::sDec
  M -- "yes" --> OC{{"<b>is_area_occupied_this_frame()</b>?"}}:::sDec
  OC -- "yes" --> O[["occupied"]]:::stop
  OC -- "no" --> WL{{"<b>is_area_wall_this_frame()</b>, on the path?"}}:::sDec
  WL -- "yes" --> W[["wall"]]:::stop
  WL -- "no" --> BL{{"<b>is_area_blocked()</b>?<br/>yes → blocked"}}:::sDec
  BL -- "no" --> FR[["free"]]:::ok
""",
        [],
    ),
    (
        12,
        "frames",
        "Lidar frames",
        "Every spatial_awareness tick: the buffer both checks read from.",
        """flowchart TD
  A(["spatial_awareness tick"]):::trig --> T["on_tick: drop frames older than 10 s<br/>spatial_awareness/main.py"]:::sOld
  T --> CF["<b>_current_frame()</b><br/>spatial_awareness/main.py"]:::sChg
  CF --> Q{{"pose and lidar fresh, and taken close together?<br/>allowed gap 1 to 2 s, tighter the faster the robot moves"}}:::sDec
  Q -- "no" --> SKIP["no new frame this tick"]:::sOld
  Q -- "yes" --> D["drop zero-intensity noise ·<br/><b>_drop_self_hits()</b>: returns inside the robot's own footprint"]:::sNew
  D --> BUF[("frame_history: read by charts 10 and 11")]:::sOld
""",
        [],
    ),
    (
        13,
        "base_nodes",
        "nav_twist and nav_slide",
        "The brain's twist, slide and rotate requests, and its cancels (charts 3 to 8).",
        """flowchart TD
  BT(["brain: nav_twist/twist · nav_slide/slide ·<br/>nav_twist/rotate · nav_twist/speed_cap"]):::trig --> RUN["<b>main.on_twist()</b> → <b>reset_twist_for_start()</b> (control.py):<br/>a cancel sent before this twist still wins<br/><b>main.on_slide()</b> (nav_slide) · <b>main.on_rotate()</b> → <b>rotate.on_rotate()</b><br/>(new) speed cap: <b>on_speed_cap()</b> → <b>set_speed_cap()</b>, <b>capped()</b> every step"]:::nChg
  BC(["brain: nav_twist/cancel · nav_twist/cancel_rotate · nav_slide/cancel"]):::trig --> CAN["<b>main.on_cancel()</b> → <b>cancel_twist()</b> (control.py), stamped with the request time<br/>(new) <b>on_cancel_rotate()</b> → <b>cancel_rotate()</b> (control.py)<br/>(new) nav_slide <b>main.on_cancel()</b>: no more bursts, forwards nav_slide/twist_cancel"]:::nChg
  AB(["system abort · teleop mode change"]):::trig --> AA["<b>abort_all()</b> (control.py) · <b>on_teleop_mode()</b><br/>reason Aborted by user"]:::nChg
  CAN --> RUN
  AA --> RUN
  RUN --> D[["done / response: success false, reason Cancelled<br/>a cancelled twist adds completed_steps (whole steps)<br/>a twist's or slide's done echoes its request_id"]]:::stop
""",
        [],
    ),
]


# Callers branch_inventory's name matching gets wrong here, fixed by hand: (file, name) -> (callers to
# add, caller prefixes to drop). It misses a call through a module attribute or an import alias, and
# matches cv2.rotate to _nav.rotate.
_CALLER_FIXES = {
    ("billie/nodes/brain/brain/navigation/blind_route_navigator.py", "blind_navigate_route"): (
        ["billie/nodes/brain/brain/commands/navigation.py::blind_navigate_route"],
        [],
    ),
    ("billie/src/nav_twist/rotate.py", "on_rotate"): (["billie/src/nav_twist/main.py::on_rotate"], []),
    ("billie/nodes/brain/brain/navigation/_nav.py", "rotate"): ([], ["billie/nodes/brain/brain/vision/billie_vision.py::"]),
}


# Functions that come from the fix_boolian_defalt and check_arguments_vals branches, which reach main
# before free_area_v2: the review treats them as already on main and leaves them out.
_ON_MAIN_FIRST = {
    ("billie/nodes/brain/brain/node.py", "ExecutionOptions.from_dict"),
    ("billie/nodes/brain/brain/node.py", "_coerce_arguments"),
    ("billie/nodes/brain/brain/node.py", "_coerce_bool"),
    ("billie/nodes/brain/brain/node.py", "_literal_values"),
    ("billie/nodes/brain/brain/node.py", "_numeric_annotation"),
    ("billie/nodes/brain/brain/node.py", "_unwrap_optional"),
    ("billie/nodes/brain/brain/spec.py", "_option_field_to_message_input"),
    ("billie/nodes/brain/brain/spec.py", "render_type"),
}


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
    """The document's opening section: every change that reaches beyond the feature. Returns: Lines."""
    lines = [
        "## ⚠️ System-wide changes — confirm each one was intended before merging",
        "",
        "These reach beyond the area checks themselves. Each one has to be a deliberate choice.",
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
    lines = [
        "## Review notes",
        "",
        "- **Open question:** with `wait=True` a blocked move waits for the operator with no overall time limit "
        "([chart 9](" + anchor(9) + ")). Should it fail by itself after a limit?",
        "- The pause image comes from the careful check and can disagree with the instant verdict that stopped the move. "
        "Planned for the next version: keep the last 2 lidar frames the live check judged, and render the image from those.",
        "- **Next version:** a wall stop with more than 0.2 m of the move still to go pauses and re-checks today "
        "([chart 8](" + anchor(8) + ")). A wall won't move away, so it should fail at once instead "
        "(`spatial_awareness/README_NEXT_STEPS.md`). `WALL_DONE_MAX_LEFT_M` (when a move counts as having reached "
        "its target) is deliberately a code constant, separate from `NAV_STOP_DISTANCE_WALL_M` (how close to a wall it may get).",
        "- The function table covers Python only (the inventory reads `.py` files). Rust changes, in "
        "`nodes/rust_host/src/actors/blind_route_navigator/`: new `speed_cap.rs` (`SpeedCap`), the `set_speed_cap` request "
        "in `mod.rs`, `Follower::drive_decision()` / `Follower::rotate_decision()` / `PurePursuit::rotate_step()` take the cap; "
        "new message `messages/blind_route_navigator_speed_cap.rs`.",
        "- Braking, command delay and the minimum speed are estimates until the live tests in "
        "`spatial_awareness/free_area.md` §5 run on a Billie with its AutoXing base.",
    ]
    unused = [
        r
        for r in rows
        if not r["used_by"] and r["status"] != "removed" and r["name"] != "main"
        and not r["name"].endswith(("__init__", "__call__"))
    ]
    lines += [f"- {kind_marker(r)} in `{r['file'].replace('billie/', '')}`: {users_text(r)}." for r in unused]
    undrawn = [i for i in range(len(rows)) if i not in charts_of]
    lines += [
        f"- {len(undrawn)} of {len(rows)} functions and classes are not drawn in a chart; each is a helper of one "
        "that is, or a removed one (see its *Called from*).",
        "",
    ]
    return lines


def build(preview: bool = False) -> tuple[str, str]:
    """The review document as markdown, in two parts that each fit GitHub's limit.

    preview: Also make the boxes that name another chart clickable (preview only).
    Returns: (the PR description: system-wide changes, guide, charts and review notes; the PR's first
        comment: every changed function and class).
    """
    with open(INVENTORY) as f:
        rows = [
            r
            for r in json.load(f)["rows"]
            if "/tests/" not in r["file"] and "/test/" not in r["file"] and (r["file"], r["name"]) not in _ON_MAIN_FIRST
        ]
    _fix_callers(rows)
    charts_of: dict[int, list[int]] = {}
    for number, _, _, _, body, _ in CHARTS:
        for i in sorted(chart_items(body, rows)):
            charts_of.setdefault(i, []).append(number)
    lines = [
        "# free_area_v2: what changed and how it flows",
        "",
        "Review guide for the `free_area_v2` branch (compared with `main`). The base moves that bypass AutoXing's "
        "obstacle avoidance (slide, rotate, twist, fixed_eef_twist, blind_navigate_route) now check the area they are "
        "about to cover with the spatial_awareness node's lidar: once before moving, and live while moving. An obstacle "
        "pauses the move, retries, then asks the operator.",
        "",
    ]
    lines += system_wide_section()
    lines += [
        f"**{len(rows)} functions and classes** are added, changed or removed (tests not counted). Every one is "
        "listed in the first comment, [All changed functions and classes](#all-changed-functions-and-classes).",
        "",
        "## How to review",
        "",
        "1. Confirm each system-wide change in the table above.",
        "2. Read [Chart 0 · Overview](#chart-0--overview): which command reaches which check.",
        "3. Open one chart at a time. Each starts from what triggers it; every branch is a real condition in the code.",
        "4. [All changed functions and classes](#all-changed-functions-and-classes) lists every one, file by file.",
        "5. [Review notes](#review-notes) lists what the reading found worth a look.",
        "",
        "## How to read the charts",
        "",
        "| What you see | Meaning |",
        "|---|---|",
        "| Blue / purple / orange / green fill | Runs in the brain / the spatial_awareness node / the nav_twist or nav_slide node / shared `billie_utils` code the brain calls |",
        "| Thick solid / dashed / thin grey border | New on the branch / changed on the branch / unchanged, shown for context |",
        "| `name()` | A function. `module.name()` when the same name exists in two modules |",
        "| `Class.method()` | A method, called on an instance of `Class` |",
        "| `«class» Name` / `«data» Name` | A class / a data class (a message). On a thick arrow: request → reply |",
        "| Rounded pill | What starts the flow |",
        "| Hexagon | A decision, filled like the node it runs in; its arrows name the cases |",
        "| Green box / red box | Goes ahead / stopped, paused or failed |",
        "| Thin / thick / dashed arrow | A call in one node / a request to another node / the other case, the old path |",
        "",
    ]
    for number, _, title, when, body, notes in CHARTS:
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
            wrap_labels(body) + (click_lines(body, number) if preview else "") + styles_for(body),
            "```",
            "",
            "</details>",
            "",
        ]
        targets = sorted({n for label in label_strings(body) for n in mentioned_charts(label, number)})
        if targets:
            lines += ["Goes on in: " + " · ".join(f"[Chart {n}]({anchor(n)})" for n in targets), ""]
        lines += [f"- {link_mentions(note, number)}" for note in notes] + ([""] if notes else [])
    lines += review_notes(rows, charts_of)
    description = "\n".join(lines)
    lines = [
        "## All changed functions and classes",
        "",
        'Grouped by file. *Chart* says where each one is drawn; "—" means it is a helper of a function that is.',
        "",
    ]
    files: list[tuple[str, list[int]]] = []
    for i, r in enumerate(rows):
        if not files or files[-1][0] != r["file"]:
            files.append((r["file"], []))
        files[-1][1].append(i)
    for path, indexes in files:
        new = sum(rows[i]["status"] == "new" for i in indexes)
        lines += [
            "<details>",
            f"<summary><code>{path.replace('billie/', '')}</code> · {len(indexes)} ({new} new, {len(indexes) - new} changed or removed)</summary>",
            "",
            table(rows, indexes, charts_of, with_file=False),
            "",
            "</details>",
            "",
        ]
    return description, "\n".join(lines)


def main() -> None:
    """Writes the PR description, its first comment and the HTML preview into billie-onboard/out/review/."""
    description, comment = build()
    os.makedirs(_OUT_DIR, exist_ok=True)
    for path, text in ((OUT_MD, description), (OUT_COMMENT_MD, comment)):
        with open(path, "w") as f:
            f.write(text)
        size = len(text.encode())
        print(f"{path}: {size} bytes" + ("" if size <= _GITHUB_MAX_BYTES else f" - OVER GitHub's {_GITHUB_MAX_BYTES}"))
    page = preview_html("\n".join(build(preview=True))).replace(
        "<title>Arm Planner Call Map</title>", "<title>Free Area v2 Review</title>"
    )
    with open(OUT_HTML, "w") as f:
        f.write(page)
    print(OUT_HTML)


if __name__ == "__main__":
    sys.exit(main())
