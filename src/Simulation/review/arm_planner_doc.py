"""Builds arm_planner.md, the arm_planner PR review document, and an HTML preview of it.

    python -m Simulation.review.branch_inventory --repo billie-onboard --out output/review/inventory.json
    python -m Simulation.review.arm_planner_doc

The charts are written below by hand, one per trigger, in the convention the document's "How to read the
charts" explains. The tables come from the inventory, so every function and class the branch adds or
changes is listed; a chart lists what it names. Update a chart when the flow it draws changes.
"""

import json
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# The inventory branch_inventory.py wrote, the document (for the PR description) and its preview page.
INVENTORY = os.path.join(_ROOT, "output", "review", "inventory.json")
OUT_MD = os.path.join(_ROOT, "output", "review", "arm_planner.md")
OUT_COMMENT_MD = os.path.join(_ROOT, "output", "review", "arm_planner_functions.md")
# GitHub's limit for a PR description and for one comment, in bytes of UTF-8.
_GITHUB_MAX_BYTES = 65536
OUT_HTML = os.path.join(_ROOT, "output", "review", "arm-planner-review.html")

# Node styles: fill = process (b brain, p planner, u billie_utils), border = change (New / Chg / Old).
STYLES = """  classDef trig fill:#ffffff,stroke:#1d2330,stroke-width:1.5px,color:#1d2330;
  classDef bNew fill:#dbe8fb,stroke:#2f62b0,stroke-width:3px,color:#1d2330;
  classDef bChg fill:#dbe8fb,stroke:#2f62b0,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef bOld fill:#dbe8fb,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef pNew fill:#eadff8,stroke:#6d40b8,stroke-width:3px,color:#1d2330;
  classDef pChg fill:#eadff8,stroke:#6d40b8,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef pOld fill:#eadff8,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef uNew fill:#e1efe4,stroke:#3f7a52,stroke-width:3px,color:#1d2330;
  classDef uChg fill:#e1efe4,stroke:#3f7a52,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef uOld fill:#e1efe4,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef bDec fill:#dbe8fb,stroke:#2f62b0,stroke-width:1.5px,color:#1d2330;
  classDef pDec fill:#eadff8,stroke:#6d40b8,stroke-width:1.5px,color:#1d2330;
  classDef uDec fill:#e1efe4,stroke:#3f7a52,stroke-width:1.5px,color:#1d2330;
  classDef ok fill:#d6efdc,stroke:#2e7d46,stroke-width:1.5px,color:#1d2330;
  classDef stop fill:#f8dada,stroke:#b3261e,stroke-width:1.5px,color:#1d2330;"""

# Callers listed per function before "and N more": keeps the document under GitHub's 65,536-character
# limit for a PR description.
_MAX_CALLERS = 4

# What each chart's flow does differently from main, shown under its "When" line.
SUMMARIES = {
    0: "On main the brain sent a pose to the IK and joints straight to the arm, and nothing knew about obstacles. Now the planner keeps the obstacles and decides every move: single moves are planned around them or refused, batches are shortened near them, and the brain checks every target once more before the arm.",
    1: "On main the planner was mostly one large resolver.py. Now the logic files decide (plan_single.py, plan_batch.py, replay_planner.py) and the solver files only solve (resolver_single.py, resolver_batch.py, resolver_transit.py); the logic reaches the solvers only through resolver.py.",
    2: "A new command. Obstacles in the map or arm frame become capsules plus the floor, which the planner keeps for every planning request; the brain keeps a copy for its last check.",
    3: "On main a pose went to the IK and then to the arm, and joints went straight to the arm. Now a pose always asks the planner (<code>plan_move</code>): solve, check that the IK reached the pose (retry if not), check that the move is clear, and route around the obstacles if not. Joints ask the planner only when obstacles are set. A blocked or unreachable move fails the command.",
    4: "The same request as on main, but the IK now has an obstacle cost (chart 9), and a move between frames that would still enter an obstacle is shortened to its farthest clear point, never refused. A poses replay is first checked whole, before it is cut into batches (chart 6).",
    5: "The same request as on main, with the same two changes as the batches: the IK has the obstacle cost, and the move to the solution is shortened if it would enter an obstacle. The reply says so.",
    6: "On main a replay played as recorded. Now the planner checks a poses replay first and replaces each stretch that enters an obstacle with a route around it, or refuses it. Joints replays are unchanged.",
    7: "New: every target the brain sends is checked against the obstacles, the floor and joint jumps over 90° before it reaches the arm; a failure stops the command.",
    8: "Startup builds the three solvers from their own files, with the obstacle costs, and loads the solvers an earlier run saved, with no Python tracing: about 26 s warm on billie-29, was about 62 s.",
    9: "On main, <code>IKResolver.resolve()</code> already solved a pose to joints from 16 starts around the arm's joints. It minimised the pose error, the joint limits, a pull toward the rest pose and self-collision, and kept the start with the lowest pose error. Now the function itself changed in one call: it passes the current obstacles to the solver. The solvers got two new costs at startup (chart 8): every moving link stays at least a margin from each obstacle capsule and the floor (<code>world_capsule_cost()</code>), and in batches and routes the straight move between frames stays clear (<code>world_segment_cost()</code>). Picking the best start also adds a penalty for entering an obstacle. With no obstacles set, the result is the same as on main.",
    10: "New debug plots (<code>DEBUG_PYROKI_PLANNER</code> only): a move's planned and actual path with the obstacles, drawn in a separate process so the arm never waits.",
}

# (number, anchor, title, when, mermaid body, notes)
CHARTS = [
    (
        0,
        "overview",
        "Overview",
        "Which brain action sends which request to the planner. The planner keeps one copy of the obstacles, which every planning request reads.",
        """flowchart TD
  subgraph CMDS[" "]
    O(["set_obstacles command"]):::trig
    subgraph SINGLE["Single moves · one target, the robot waits"]
      S1(["pose command"]):::trig
      S2(["joints command · fold ·<br/>other single moves"]):::trig
    end
    subgraph BATCH["Batches · a stream of targets, played while solving"]
      B1(["AI action batches"]):::trig
      B2(["replay_policy"]):::trig
      B3(["pick: move_linear()"]):::trig
    end
    SV(["transformations · vision ·<br/>a replay's first frame (IK only)"]):::trig
  end
  subgraph PL["pyroki planner node"]
    R1["<b>on_set_world_obstacles()</b><br/>chart 2"]:::pNew
    S[("obstacles<br/>capsules + floor")]:::pNew
    PM["<b>plan_single.plan_move()</b><br/>chart 3"]:::pNew
    BP["<b>on_batch_plan()</b> → <b>keep_solution_clear()</b><br/>chart 4"]:::pChg
    RP["<b>replay_planner.plan_replay()</b><br/>chart 6"]:::pNew
    OP["<b>on_plan()</b> → <b>keep_solution_clear()</b><br/>chart 5"]:::pChg
  end
  G["the brain queues the frames →<br/><b>arm_motion_guard</b> last safety check · chart 7"]:::uNew
  X(["xarm_writer → arm"]):::trig
  O == "set_world_obstacles" ==> R1 --> S
  S1 == "plan_move · always" ==> PM
  S2 == "plan_move · with obstacles" ==> PM
  B1 == "batch_plan" ==> BP
  B3 == "batch_plan" ==> BP
  B2 == "plan_replay · once, before playing" ==> RP
  B2 == "batch_plan" ==> BP
  SV == "plan" ==> OP
  S -. "read by" .-> PM & BP & OP & RP
  PM == "reply: frames" ==> G
  BP == "reply: frames" ==> G
  G --> X
  style CMDS fill:none,stroke:none
  style SINGLE fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
  style BATCH fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
  style PL fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [],
    ),
    (
        1,
        "files",
        "Planner files",
        "Which planner file holds what. The logic files decide; the solver files only solve; every solver is reached through <code>resolver.py</code>.",
        """flowchart TD
  subgraph REQ[" "]
    H1(["plan_move request"]):::trig
    H2(["plan / batch_plan requests"]):::trig
    H3(["plan_replay request"]):::trig
  end
  H1 --> PS["<b>plan_single.py</b> · chart 3<br/>single moves: solve, check reached, retry,<br/>check clear, route around"]:::pNew
  H2 --> PB["<b>plan_batch.py</b> · charts 4 and 5<br/>shorten moves that enter an obstacle"]:::pNew
  H3 --> RP["<b>replay_planner.py</b> · chart 6<br/>whole-recording reroute"]:::pNew
  PS --> R["<b>resolver.py</b> · «class» IKResolver, the one door<br/><b>IKResolver.resolve()</b> · <b>IKResolver.resolve_batch()</b> · <b>IKResolver.route()</b> · <b>IKResolver.arm_model()</b><br/>shared: <b>load_arm_models()</b>, build + warm-up"]:::pChg
  PB --> R
  RP --> R
  R --> RS["<b>resolver_single.py</b><br/>single IK solver: build, compile, warm-up"]:::pNew
  R --> RB["<b>resolver_batch.py</b><br/>batch IK solver: build, compile, warm-up"]:::pNew
  R --> RT["<b>resolver_transit.py</b><br/>route solver: build, compile, warm-up"]:::pNew
  RS --> WC["<b>world_collision.py</b> · obstacle state and costs · chart 9<br/><b>exported_solvers.py</b> · saved solvers · chart 8"]:::pNew
  RB --> WC
  RT --> WC
  style REQ fill:none,stroke:none
""",
        [],
    ),
    (
        2,
        "obstacles",
        "set_obstacles",
        "A script or operator sends the <code>set_obstacles</code> command (tables, walls, doors, capsules; map or arm frame). The floor is added unless <code>floor=False</code>.",
        """flowchart TD
  A(["set_obstacles command<br/>registered by main() · brain/main.py"]):::trig --> B["<b>set_obstacles()</b><br/>brain/commands/world_obstacles.py"]:::bNew
  B -- "a table given as an AutoXing rect" --> R["<b>_with_rect()</b><br/>brain/commands/world_obstacles.py"]:::bNew
  B --> C["<b>obstacle_in_arm_frame()</b><br/>billie_utils/world_obstacles.py<br/>frame = map: <b>map_to_arm_frame()</b> → <b>planar_transform()</b> (pyroki_world.py)"]:::uNew
  R --> C
  C --> OC["<b>obstacle_capsules()</b> → <b>_board_capsules()</b><br/>billie_utils/world_obstacles.py<br/>table · wall · door: <b>tabletop_capsules()</b> · <b>wall_capsules()</b> (pyroki_world.py) · <b>capsule_row()</b> (world_obstacles.py)"]:::uNew
  OC --> D{{"more than 32 capsules?<br/>yes → refused: ValueError, use fewer or thicker obstacles"}}:::bDec
  D -- "no" --> F["<b>set_current_joints_provider()</b><br/>billie_utils/arm_motion_guard.py"]:::uNew
  F --> G["<b>set_world_obstacles()</b><br/>billie_utils/messages/pyroki_world.py"]:::uNew
  G -- "first time" --> GM["<b>get_collision_model()</b><br/>billie_utils/messages/pyroki_world.py"]:::uNew
  GM == "get_collision_model<br/>«data» GetCollisionModelRequest →<br/>«data» GetCollisionModelResponse" ==> GH["<b>_on_get_collision_model()</b> → <b>export_collision_model()</b><br/>pyroki_planner/world_collision.py"]:::pNew
  GH --> CM["<b>«data» CollisionModel</b><br/><b>CollisionModel.to_json()</b> · <b>CollisionModel.from_json()</b><br/>billie_utils/arm_collision_model.py"]:::uNew
  G == "set_world_obstacles<br/>«data» SetWorldObstaclesRequest →<br/>«data» SetWorldObstaclesResponse" ==> H["<b>on_set_world_obstacles()</b><br/>pyroki_planner/world_collision.py"]:::pNew
  H --> HR["<b>_capsule_rows()</b> → <b>_quat_z_to_direction()</b> · <b>_floor_part()</b><br/>pyroki_planner/world_collision.py"]:::pNew
  HR --> I[("planner copy, read by current_world_capsules()<br/>and current_obstacles()")]:::pNew
  G --> J["brain copy: <b>last_sent_obstacles()</b> · <b>last_sent_floor()</b><br/>billie_utils/messages/pyroki_world.py"]:::uNew
  J --> K["<b>active_obstacles()</b> → <b>«data» ActiveObstacles</b><br/>billie_utils/arm_motion_guard.py"]:::uNew
""",
        [
            '"Obstacles set" in every later chart means <code>active_obstacles()</code> is not None.'
        ],
    ),
    (
        3,
        "single",
        "Single moves",
        "The <code>pose</code> command always; the <code>joints</code> command, fold and every other single move that clears the arm's queue when obstacles are set. The planner part follows four steps: solve, check that it reached the pose, check that the move is clear, go around.",
        """flowchart TD
  subgraph CMDS[" "]
    P(["pose command"]):::trig
    J(["joints command · fold · other single moves"]):::trig
  end
  P --> SP["<b>set_state_with_arm_pose()</b><br/>brain/behaviours/state.py"]:::bChg
  J --> SJ["<b>set_state_with_arm_joints()</b><br/>brain/behaviours/state.py<br/>IK debug plots on: <b>MoveRecorder.begin()</b> (new) · chart 10"]:::bChg
  SJ --> O{{"obstacles set and the queue is cleared?"}}:::bDec
  O -. "no: straight to the arm" .-> ENQ
  SP -- "always" --> REQ["<b>_plan_single_move()</b> → <b>pyroki_world.plan_move()</b><br/>brain/behaviours/state.py · billie_utils/messages/pyroki_world.py"]:::bNew
  O -- "yes, not planned yet" --> REQ
  REQ == "plan_move · «data» PlanMoveRequest" ==> H["<b>on_plan_move()</b> → <b>plan_single.plan_move()</b><br/>pyroki_planner/plan_single.py"]:::pNew
  H --> K{{"goal is a pose?"}}:::pDec
  subgraph PLN["plan_single.py"]
    K
    IK["1 · solve: <b>solve_pose_goal()</b> → <b>IKResolver.resolve()</b><br/>resolver.py · with the obstacles (none is fine)"]:::pChg
    BC{{"inside Billie's base?<br/>yes → base_collision: a warning, the arm stays"}}:::pDec
    R{{"2 · reached the pose?<br/><b>_pose_error()</b> → <b>tcp_poses()</b> · <b>_reaches()</b>"}}:::pDec
    RT["retry: <b>_retry_solutions()</b> from the rest pose,<br/>the zero pose, a 2-step chain · <b>_nearest_turns()</b>"]:::pNew
    R2{{"reached now?"}}:::pDec
    CL{{"3 · straight move clear?<br/><b>plan_joint_move()</b> → <b>check_joint_path()</b>"}}:::pDec
    RTE["4 · route around: <b>IKResolver.route()</b><br/>resolver.py → resolver_transit.py"]:::pNew
    CR{{"route clear?"}}:::pDec
  end
  K -- "yes" --> IK --> BC
  BC -- "no" --> R
  K -- "no: joints" --> CL
  R -- "yes" --> CL
  R -- "no" --> RT --> R2
  R2 -- "yes" --> CL
  CL -- "no" --> RTE --> CR
  CL -- "yes: direct" --> Q
  CR -- "yes: detour" --> Q
  R2 -- "no: unreached" --> ERR
  CR -- "no: blocked" --> ERR
  Q["reply · «data» PlanMoveResponse<br/><b>set_state_with_arm_joints()</b> queues the frames · a pose's plan is passed in as planned=plan"]:::bNew --> ENQ["<b>enqueue_joints()</b><br/>billie_utils/messages/xarm_writer.py · chart 7"]:::uChg
  ERR[["«class» ArmMoveBlockedError: the command fails with the planner's reason"]]:::stop
  style CMDS fill:none,stroke:none
  style PLN fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [
            "The planner never moves the arm: it replies with the frames, and the brain queues them.",
            "A detour's message (“going around: 23 frames…”) is sent to the cloud by <code>_plan_single_move()</code>.",
        ],
    ),
    (
        4,
        "batches",
        "Batches",
        "A stream of targets, solved batch by batch while the arm plays them: AI action batches, pick motions, and a replay's frames.",
        """flowchart TD
  subgraph TOP[" "]
    AI(["AI action batches<br/>ai.py"]):::trig
    RPL(["replay_policy<br/>brain/behaviours/play/replay_policy.py"]):::trig
    PK(["pick: <b>move_linear()</b><br/>pick/motion.py"]):::trig
  end
  AI --> BS["<b>pyroki_node.batch_solve()</b><br/>billie_utils/messages/pyroki_node.py"]:::uOld
  PK --> BS
  RPL --> T{{"transform"}}:::bDec
  T -. "joints: frames one by one, last check only" .-> G
  T -- "poses, obstacles set,<br/>recorded joints on every frame" --> WR["first, the whole recording: <b>reroute_around_obstacles()</b><br/>plan_replay · chart 6<br/>no route → refused before the arm moves"]:::bNew
  T -- "other pose transforms" --> BUF
  WR -- "rerouted or unchanged" --> BUF["«class» BufferingPlayer cuts it into batches"]:::bOld
  BUF --> RBS["<b>pyroki_node.request_batch_solve()</b><br/>billie_utils/messages/pyroki_node.py"]:::uOld
  BS == "batch_plan · waits" ==> OBP
  RBS == "batch_plan · doesn't wait" ==> OBP
  OBP["<b>on_batch_plan()</b><br/>pyroki_planner/main.py"]:::pChg
  IKB["1 · solve: <b>IKResolver.resolve_batch()</b><br/>resolver.py → resolver_batch.py · with the obstacles (none is fine)"]:::pChg
  subgraph PLB["plan_batch.py"]
    CLB{{"2 · every move between the frames clear?<br/><b>keep_solution_clear()</b>"}}:::pDec
    SH["3 · shorten each blocked move to its farthest clear point<br/><b>keep_moves_clear()</b>"]:::pNew
  end
  OBP --> IKB --> CLB
  CLB -- "no" --> SH
  CLB -- "yes" --> OUT
  SH --> OUT
  OUT["reply · «data» BatchSolveResponse<br/>obstacle_adjusted_mask says which moves were shortened"]:::bOld
  OUT ==> PLAY["the brain plays the frames as they come:<br/><b>set_state_with_arm_joints()</b> without clearing the queue (AI, replay) ·<br/><b>enqueue_joints()</b> (pick) · <b>play_joints_batch()</b> (AI reverse)"]:::bOld
  PLAY --> G["last check per target → arm · chart 7"]:::uChg
  style TOP fill:none,stroke:none
  style PLB fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [
            "A batch never refuses: near an obstacle the arm gets as close as it can.",
            "Batches don't check that the IK reached each pose. Near an obstacle a frame is meant to stop short, so it isn't clear yet what such a check should report; the open question is noted in <code>plan_batch.py</code>.",
        ],
    ),
    (
        5,
        "solve",
        "Single IK solve",
        "One IK solve that isn't a move: transformations, vision captures, and a replay's first frame, whose move then goes through chart 3.",
        """flowchart TD
  C(["transformations · vision captures · a replay's first frame"]):::trig --> S1["<b>pyroki_node.solve()</b><br/>billie_utils/messages/pyroki_node.py · waits"]:::uOld
  S1 == "plan" ==> P1["<b>on_plan()</b><br/>pyroki_planner/main.py"]:::pChg
  P1 --> R1["1 · solve: <b>IKResolver.resolve()</b><br/>resolver.py → resolver_single.py · with the obstacles (none is fine)"]:::pChg
  subgraph PB["plan_batch.py"]
    K{{"2 · the move to the solution clear?<br/><b>keep_solution_clear()</b>"}}:::pDec
    SH["3 · shorten it to its farthest clear point<br/><b>keep_moves_clear()</b>"]:::pNew
  end
  R1 --> K
  K -- "no" --> SH
  K -- "yes" --> OUT
  SH --> OUT
  OUT["reply · «data» SolveResponse<br/>is_obstacle_adjusted says it was shortened"]:::bOld
  OUT ==> USE["the caller uses the joints; a replay's first frame is then driven as a single move (chart 3)"]:::bOld
  style PB fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [],
    ),
    (
        6,
        "replay",
        "replay_policy",
        'The <code>replay_policy</code> command. The reroute runs only for <code>transform="poses"</code>, where the recorded joints match the poses the planner checks.',
        """flowchart TD
  RP(["replay_policy command"]):::trig --> RPF["<b>replay_policy()</b><br/>brain/behaviours/play/replay_policy.py"]:::bChg
  RPF --> T{{"transform<br/>joints → <b>arm.replay_joints_batch()</b>: each frame → enqueue_joints(), chart 7"}}:::bDec
  T -- "poses" --> RR["<b>reroute_around_obstacles()</b><br/>brain/behaviours/play/replay_reroute.py"]:::bNew
  RR --> Q1{{"obstacles set and<br/>recorded joints on every frame?"}}:::bDec
  Q1 -. "no" .-> PLAY
  Q1 -- "yes" --> PW["<b>pyroki_world.plan_replay()</b><br/>billie_utils/messages/pyroki_world.py"]:::uNew
  PW == "plan_replay<br/>«data» PlanReplayRequest →<br/>«data» PlanReplayResponse" ==> H["<b>on_plan_replay()</b> → <b>replay_planner.plan_replay()</b><br/>pyroki_planner/replay_planner.py"]:::pNew
  H --> RC["<b>reroute_recording()</b><br/>pyroki_planner/replay_planner.py"]:::pNew
  RC --> CL["<b>_clearances()</b> → <b>_distances()</b><br/>every frame and the moves between them"]:::pNew
  CL --> LD["<b>link_obstacle_distances()</b> · <b>floor_link_distances()</b><br/>billie_utils/arm_collision_model.py"]:::uNew
  CL --> ST["<b>_stretches()</b><br/>the runs of frames that enter an obstacle"]:::pNew
  ST --> Q2{{"what was found"}}:::pDec
  Q2 -- "starts or ends inside an obstacle" --> B1[["blocked"]]:::stop
  Q2 -- "no stretch" --> U[["unchanged"]]:::ok
  Q2 -- "stretches" --> LOOP["per stretch: transit route between the clear frames around it,<br/>then check_joint_path()"]:::pNew
  LOOP --> Q3{{"route clear?"}}:::pDec
  Q3 -- "no" --> B2[["blocked: No route found"]]:::stop
  Q3 -- "yes" --> RS["<b>_resample()</b> → <b>tcp_poses()</b><br/>route frames at the recording's tool speed"]:::pNew
  RS --> RE[["rerouted"]]:::ok
  B1 ==> ERR[["brain: «class» ArmMoveBlockedError"]]:::stop
  B2 ==> ERR
  U ==> PLAY
  RE ==> MAP["brain maps the frames back to states: a route frame takes its source<br/>frame's gripper; brain commands move to the nearest route frame"]:::bNew
  MAP --> PLAY["<b>arm.replay_poses_batch()</b> → «class» BufferingPlayer<br/>batch_plan per batch (chart 4) → enqueue_joints()"]:::bOld
""",
        [],
    ),
    (
        7,
        "guard",
        "Last safety check",
        "Every joint target the brain sends to the arm, whichever chart it came from. It stays in the brain for now (open question).",
        """flowchart TD
  E1(["<b>enqueue_joints()</b><br/>billie_utils/messages/xarm_writer.py<br/>single moves, replay frames"]):::trig --> CM["<b>check_move()</b><br/>billie_utils/arm_motion_guard.py"]:::uNew
  E2(["<b>play_joints_batch()</b><br/>brain/behaviours/player.py<br/>AI batches"]):::trig --> CJS["<b>check_joint_sequence()</b><br/>billie_utils/arm_motion_guard.py"]:::uNew
  CM --> A{{"<b>active_obstacles()</b><br/>obstacles set?"}}:::uDec
  CJS --> A
  A -. "no" .-> SEND
  A -- "yes" --> GM["<b>get_cached_collision_model()</b> (pyroki_world.py) ·<br/><b>current_joints()</b> (arm_motion_guard.py)"]:::uNew
  GM --> SP{{"<b>_raise_if_spinning()</b><br/>a queued step turns a joint more than 90°?"}}:::uDec
  SP -- "yes" --> X[["«class» ArmMoveBlockedError"]]:::stop
  SP -- "no" --> PC["<b>ActiveObstacles.check()</b> → <b>check_joint_path()</b><br/>billie_utils/arm_motion_guard.py · arm_collision_model.py"]:::uNew
  PC --> SJ["<b>sample_joint_path()</b> · <b>link_obstacle_distances()</b> → <b>_segment_distances()</b> ·<br/><b>floor_link_distances()</b> · <b>link_transforms()</b> → <b>_topological_order()</b>, <b>_se3_exp()</b> → <b>_skew()</b><br/>billie_utils/arm_collision_model.py"]:::uNew
  SJ --> C{{"«data» PathCheckResult: clear?"}}:::uDec
  C -- "no" --> RB["<b>_raise_if_blocked()</b><br/>billie_utils/arm_motion_guard.py"]:::uNew
  RB --> X
  C -- "yes" --> SEND[["sent to xarm_writer → arm<br/>target kept for queued_targets()"]]:::ok
""",
        [
            "A move that clears the queue starts from the arm's current joints; a queued step starts from the previous queued target, as the arm will drive it."
        ],
    ),
    (
        8,
        "startup",
        "Planner startup",
        "The pyroki planner node starts (Billie boots, or the node restarts).",
        """flowchart TD
  ST(["pyroki-planner starts"]):::trig --> MN["<b>main()</b><br/>pyroki_planner/main.py"]:::pChg
  MN --> RB["«class» IKResolver: <b>_build_solver()</b> in a background thread<br/>pyroki_planner/resolver.py"]:::pChg
  RB --> LM["<b>load_arm_models()</b> → «data» ArmModels · <b>_startup_inputs()</b> → «data» StartupInputs<br/>resolver.py · with <b>set_movable_links()</b> and <b>stabilize_jaxls_cost_order()</b> (world_collision.py)"]:::pNew
  LM --> B1["<b>resolver_single.build()</b><br/>pyroki_planner/resolver_single.py · costs: chart 9"]:::pNew
  LM --> B2["<b>resolver_batch.build()</b><br/>pyroki_planner/resolver_batch.py · costs: chart 9"]:::pNew
  B1 --> ES["«class» ExportedSolver ik_single · ik_batch<br/>pyroki_planner/exported_solvers.py"]:::pNew
  B2 --> ES
  MN --> WC["<b>register_handlers()</b><br/>world_collision.py · set_world_obstacles · get_collision_model · plan_move · plan_replay"]:::pNew
  WC --> WU["<b>warm_up_in_background()</b> → <b>get_solver()</b> → <b>build_transit_solver()</b><br/>pyroki_planner/resolver_transit.py · after the IK solvers are ready"]:::pNew
  WU --> ES2["«class» ExportedSolver transit-24 · «class» _TransitEndVar · <b>_end_cost()</b>"]:::pNew
  ES --> F{{"<b>ExportedSolver.__init__()</b>: a saved export with the same <b>fingerprint()</b>?<br/><b>_source_digest()</b> · <b>_shapes()</b> · <b>exports_folder()</b>"}}:::pDec
  ES2 --> F
  F -- "yes" --> L["load it: no Python tracing"]:::pNew
  F -. "no" .-> TR["trace on the first call, as before"]:::pOld
  TR --> SV["<b>ExportedSolver.__call__()</b> → <b>ExportedSolver._save()</b><br/>background thread; exit waits for it"]:::pNew
  L --> W["warm-ups: both IK solvers in parallel, <b>_warm_up()</b> (resolver.py),<br/>then the transit solver · debug: <b>_warm_up_near_collision_check()</b>"]:::pNew
  SV --> W
  W --> CC["XLA compile, or load from the JAX cache"]:::pOld
  CC --> RDY[["ready: about 26 s warm on billie-29, was about 62 s"]]:::ok
""",
        [],
    ),
    (
        9,
        "costs",
        "Obstacle costs inside the solvers",
        "Every IK solve (charts 3, 4 and 5) and every route. This is the planner's own obstacle avoidance; chart 7 is the brain's separate check.",
        """flowchart TD
  IN(["a solve: on_plan() · on_batch_plan() · plan_move · a route"]):::trig --> RS["<b>IKResolver.resolve()</b> · <b>IKResolver.resolve_batch()</b> · <b>IKResolver.route()</b><br/>pyroki_planner/resolver.py · changed: pass the obstacles"]:::pChg
  BS["at startup (chart 8): <b>_variables()</b> · <b>_costs()</b> in resolver_single.py and resolver_batch.py ·<br/><b>_obstacle_costs()</b> (resolver_batch.py) · <b>build_transit_solver()</b> (resolver_transit.py)"]:::pNew -.-> WCC
  BS -.-> WSC
  RS --> CW["<b>current_world_capsules()</b><br/>pyroki_planner/world_collision.py · the packed array: 32 capsules × 9 + floor"]:::pNew
  CW --> WV["«class» WorldCapsulesVar: the array as a constant solver input<br/>pyroki_planner/world_collision.py"]:::pNew
  WV --> WCC["<b>world_capsule_cost()</b><br/>each frame: every moving link at least the margin from every capsule"]:::pNew
  WV --> WSC["«class» PrevJointsVar: the arm's joints before a batch's first frame →<br/><b>world_segment_cost()</b> (batch and transit)<br/>the straight move between two frames stays clear"]:::pNew
  WCC --> WR["<b>world_residuals()</b><br/>pyroki_planner/world_collision.py<br/><b>capsules_from_packed()</b> · <b>_obstacle_links()</b> · <b>floor_link_distances()</b>"]:::pNew
  WSC --> WR
  WR --> PICK{{"which of the 16 starts wins?<br/>starts from <b>multi_start_perturbations()</b> (resolver.py)"}}:::pDec
  PICK --> SEL["pose error + <b>world_selection_penalty()</b> · batch: + <b>batch_segment_selection_penalty()</b><br/>in <b>_multi_start()</b> (resolver_single.py, resolver_batch.py)"]:::pNew
  SEL --> OUT[["the solution handed to keep_solution_clear() or plan_move"]]:::ok
""",
        [],
    ),
    (
        10,
        "plots",
        "Debug plots",
        "Only with <code>DEBUG_PYROKI_PLANNER=true</code>. A blocking single move (chart 3) or a pick motion records itself, and its plot renders in a separate process so the arm never waits.",
        """flowchart TD
  M(["set_state_with_arm_joints() · move_linear() (pick/motion.py)"]):::trig --> B["<b>MoveRecorder.begin()</b> → «class» MoveRecorder<br/>brain/behaviours/play/move_plot.py"]:::bNew
  B --> SM["<b>MoveRecorder.sample()</b> while waiting<br/>also from <b>_wait_queue_empty_or_fault()</b> (pick/motion.py)"]:::bNew
  SM --> FIN["<b>MoveRecorder.finish()</b> · <b>queued_targets()</b> · <b>current_activity()</b> → «data» Activity"]:::bNew
  FIN --> RCM["<b>request_collision_model()</b><br/>billie_utils/messages/pyroki_world.py · does not wait"]:::uNew
  RCM --> SUB["<b>_submit()</b> → «data» MoveRecord<br/>brain/behaviours/play/move_plot.py"]:::bNew
  AI(["AI / replay tracking: PoseTracker.dump()"]):::trig --> PT["<b>PoseTracker.dump()</b> · <b>PoseTracker._plot_ai()</b><br/>brain/behaviours/play/pose_tracker.py"]:::bChg
  SUB --> Q["<b>plot_worker.submit()</b> · <b>snapshot()</b> → «data» _Job, at most 20 queued<br/>brain/behaviours/play/plot_worker.py · world_obstacles_plot.py"]:::bNew
  PT --> Q
  Q --> NX["<b>_start_next()</b>: starts the plot process"]:::bNew
  NX --> RJ["plot process: <b>_render_job()</b> · «class» _MessageSink · <b>restore()</b><br/>brain/behaviours/play/plot_worker.py"]:::bNew
  RJ --> RMP["<b>render_move_plot()</b> → <b>_tool_poses()</b> · <b>_summary()</b> · <b>sample_joint_path()</b><br/>brain/behaviours/play/move_plot.py"]:::bNew
  RMP --> PIK["<b>plot_ik_results()</b> → <b>add_virtual_obstacles()</b> → <b>_add_capsules()</b> →<br/><b>_capsule_surface_points()</b> · <b>_add_floor()</b> · plot_ik_results.py · world_obstacles_plot.py"]:::bChg
  PIK --> SINK["<b>_MessageSink.send_cloud_message()</b> · <b>_MessageSink.upload_for_debugging()</b><br/>kept for the brain"]:::bNew
  SINK --> TICK["brain tick: <b>_on_tick()</b> → <b>_deliver()</b> → <b>_read_result()</b> · <b>_upload()</b><br/>brain/behaviours/play/plot_worker.py"]:::bNew
  TICK --> NODE[["<b>OrchestratorNode.send_cloud_message()</b> · <b>OrchestratorNode.upload_for_debugging()</b><br/>brain/node.py · posted under the command's activity"]]:::ok
""",
        [],
    ),
]


# [characters] Longest label line before it is split at a natural break; at Mermaid's 14px font this
# keeps a box under about 360px, so a chart stays readable without scrolling sideways.
_MAX_LINE_CHARS = 46
# Natural breaks to split a long label line at, best first; the break's own spaces are dropped.
_BREAKS = (" · ", " → ", ", ", " ")
# A preferred break wins only within this share of half the line from its middle; else the next kind.
_BALANCE = 0.5


def _visible_length(text: str) -> int:
    """Characters a label line shows (its HTML tags left out). text: The line. Returns: The count."""
    return len(re.sub(r"<[^>]+>", "", text))


def _split_line(line: str) -> list[str]:
    """A label line split at natural breaks until every piece is at most _MAX_LINE_CHARS long.

    Splits only outside HTML tags and outside <b>…</b>, so no tag is ever cut. A line with no usable
    break stays whole.
    line: One line of a label (the text between <br/> tags).
    Returns: The pieces, in order.
    """
    if _visible_length(line) <= _MAX_LINE_CHARS:
        return [line]
    middle = _visible_length(line) / 2
    best_any = None  # the most central usable break of any kind
    for brk in _BREAKS:
        best = None
        for match in re.finditer(re.escape(brk), line):
            before = line[: match.start()]
            if before.count("<b>") != before.count("</b>") or before.count(
                "<"
            ) != before.count(">"):
                continue  # inside a tag or a bold name
            distance = abs(_visible_length(before) - middle)
            if best is None or distance < best[0]:
                best = (distance, match.start(), match.end(), brk)
        if best is not None and best[0] <= middle * _BALANCE:
            break  # the best kind of break near the middle wins
        if best is not None and (best_any is None or best[0] < best_any[0]):
            best_any = best
    else:
        best = best_any  # no kind had a break near the middle: the most central one
    if best is None:
        return [line]
    _, start, end, brk = best
    head = line[:start] + (
        " " + brk.strip() if brk.strip() in ("·", "→") else brk.rstrip()
    )
    return _split_line(head.rstrip()) + _split_line(line[end:].lstrip())


def wrap_labels(body: str) -> str:
    """A chart with every long label line split at a natural break, so the boxes fit their text.

    Mermaid sizes a box to its longest line; a shorter line makes the box narrower and taller instead
    of letting the text run past the edge.
    body: The chart's mermaid text. Returns: The same chart, long label lines split with <br/>.
    """

    def wrap(match: re.Match) -> str:
        """One quoted label with its long lines split. match: A quoted string. Returns: The string."""
        lines = re.split(r"<br/>", match.group(1))
        return (
            '"'
            + "<br/>".join(piece for line in lines for piece in _split_line(line))
            + '"'
        )

    return re.sub(r'"([^"]*)"', wrap, body)


def styles_for(body: str) -> str:
    """The classDef lines of the node styles a chart uses. body: The chart's mermaid text. Returns: The lines."""
    used = set(re.findall(r":::(\w+)", body))
    return "\n".join(line for line in STYLES.split("\n") if line.split()[1] in used)


def label_strings(body: str) -> list[str]:
    """Every quoted label (node or edge) in a mermaid body."""
    return re.findall(r'"([^"]*)"', body)


def chart_items(body: str, rows: list[dict]) -> set[int]:
    """Indexes of the inventory rows a chart names, matched by name and, for ambiguous names, by file."""
    found: set[int] = set()
    simple_count: dict[str, int] = {}
    for r in rows:
        simple_count[r["name"].split(".")[-1]] = (
            simple_count.get(r["name"].split(".")[-1], 0) + 1
        )
    for text in label_strings(body):
        plain = re.sub(
            r"<[^>]+>", " ", text
        )  # a space, so words either side of a tag stay apart
        stems = set(re.findall(r"(\w+)\.py", plain))
        names = re.findall(r"([A-Za-z_]\w*(?:\.\w+)?)\(\)", plain)
        classes = re.findall(r"«(?:class|data)» (\w+)", plain)
        for i, r in enumerate(rows):
            stem = os.path.splitext(os.path.basename(r["file"]))[0]
            simple = r["name"].split(".")[-1]
            for n in names:
                if n == r["name"] and "." in n:  # Class.method
                    found.add(i)
                elif n.split(".")[-1] == simple and r["kind"] == "function":
                    prefix = n.split(".")[0] if "." in n else None
                    if prefix == stem or (
                        prefix is None and (stem in stems or simple_count[simple] == 1)
                    ):
                        found.add(i)
            if r["name"] in classes and r["kind"] in ("class", "data class"):
                found.add(i)
    return found


def kind_marker(r: dict) -> str:
    """How a row's name is written: name() for functions, Class.method(), «class» / «data» for classes."""
    if r["kind"] in ("function", "method"):
        return f"`{r['name']}()`"
    return f"«{'data' if r['kind'] == 'data class' else 'class'}» `{r['name']}`"


def users_text(r: dict) -> str:
    """The 'Called from' cell: callers as module.function, shortened when there are many."""
    if r["name"] == "main":
        return "node entry point"
    users = sorted(
        {
            os.path.splitext(os.path.basename(u.split("::")[0]))[0]
            + "."
            + u.split("::")[1]
            for u in r["used_by"]
        }
    )
    users = [u.replace(".<module>", " (module level)") for u in users]
    text = ", ".join(f"`{u}`" for u in users[:_MAX_CALLERS]) + (
        f" and {len(users) - _MAX_CALLERS} more" if len(users) > _MAX_CALLERS else ""
    )
    return text or "**not called anywhere**"


def table(
    rows: list[dict],
    indexes: list[int],
    charts_of: dict[int, list[int]],
    with_file: bool,
) -> str:
    """A markdown table of inventory rows."""
    head = (
        "| Name | Change | What it does | Called from | Chart |"
        if not with_file
        else "| Name | File | Change | What it does | Called from |"
    )
    out = [
        head,
        "|" + "---|" * head.count("|")[:-0]
        if False
        else "|" + "---|" * (head.count("|") - 1),
    ]
    for i in indexes:
        r = rows[i]
        doc = r["doc"].replace("|", "\\|") or "—"
        charts = ", ".join(str(c) for c in charts_of.get(i, [])) or "—"
        if with_file:
            out.append(
                f"| {kind_marker(r)} | `{r['file'].replace('billie/', '')}` | {r['status']} | {doc} | {users_text(r)} |"
            )
        else:
            out.append(
                f"| {kind_marker(r)} | {r['status']} | {doc} | {users_text(r)} | {charts} |"
            )
    return "\n".join(out)


def anchor(number: int) -> str:
    """The link to a chart's heading, as GitHub names heading anchors.

    number: The chart's number. Returns: "#chart-n--title" for its "## Chart N · title" heading.
    """
    title = next(t for n, _, t, _, _, _ in CHARTS if n == number)
    text = re.sub(r"[^\w\s-]", "", f"chart {number} · {title}".lower())
    return "#" + re.sub(r"\s", "-", text)


# A mention of other charts in a label or a sentence: "chart 3", "charts 2 and 3".
_CHART_MENTION = re.compile(r"\b([Cc]harts?) (\d+(?: and \d+)?)")
# A node definition in a chart: its id and quoted label ([..], ([..]), [[..]], [(..)], {{..}}).
_NODE = re.compile(r'\b(\w+)(?:\[\[|\(\[|\[\(|\{\{|\[)"([^"]*)"')


def mentioned_charts(text: str, number: int) -> list[int]:
    """The other charts a text mentions, in order of first mention.

    text: A label or sentence. number: The chart it belongs to (not listed). Returns: Chart numbers.
    """
    found: list[int] = []
    for match in _CHART_MENTION.finditer(re.sub(r"<[^>]+>", " ", text)):
        for n in map(int, re.findall(r"\d+", match.group(2))):
            if n != number and n not in found:
                found.append(n)
    return found


def link_mentions(text: str, number: int) -> str:
    """A sentence with each mention of another chart turned into a link to it.

    text: The sentence (markdown). number: Its own chart, left unlinked. Returns: The sentence.
    """

    def link(match: re.Match) -> str:
        """One mention linked: "chart 2" whole, "charts 3 and 4" per number. match: A _CHART_MENTION match.

        Returns: The text.
        """
        numbers = [int(n) for n in re.findall(r"\d+", match.group(2))]
        if len(numbers) == 1:
            return (
                match.group(0)
                if numbers[0] == number
                else f"[{match.group(0)}]({anchor(numbers[0])})"
            )
        return f"{match.group(1)} " + re.sub(
            r"\d+",
            lambda d: (
                d.group()
                if int(d.group()) == number
                else f"[{d.group()}]({anchor(int(d.group()))})"
            ),
            match.group(2),
        )

    return _CHART_MENTION.sub(link, text)


def click_lines(body: str, number: int) -> str:
    """Mermaid click lines that make each box naming another chart open that chart (preview only:
    GitHub does not run chart clicks).

    body: The chart's mermaid text. number: The chart's number. Returns: The lines, one per box.
    """
    lines = []
    for node, label in _NODE.findall(body):
        targets = mentioned_charts(label, number)
        if targets:
            lines.append(f'  click {node} href "{anchor(targets[0])}" _self')
    return "\n".join(lines) + ("\n" if lines else "")


def build(preview: bool = False) -> tuple[str, str]:
    """The review document as markdown, in two parts that each fit GitHub's limit.

    preview: Also make the boxes that name another chart clickable (the preview page runs chart clicks;
        GitHub does not, and the PR copy stays smaller without them).
    Returns: (the PR description: guide, charts and review notes; the PR's first comment: every changed
        function and class). GitHub shows both on one page, so links between them work.
    """
    rows_all = json.load(open(INVENTORY))["rows"]
    rows = [r for r in rows_all if "/tests/" not in r["file"]]
    charts_of: dict[int, list[int]] = {}  # inventory row -> the charts that draw it
    for number, _, _, _, body, _ in CHARTS:
        for i in sorted(chart_items(body, rows)):
            charts_of.setdefault(i, []).append(number)
    lines = [
        "# arm_planner: what changed and how it flows",
        "",
        "Review guide for the `arm_planner` branch (compared with `main`). The arm now plans around world obstacles in the "
        "pyroki planner: the brain sends the obstacles once, then asks the planner for every single move, every IK solve and "
        "every pose replay, and checks every move once more before it reaches the arm.",
        "",
        f"**{len(rows)} functions and classes** are added or changed (tests not counted). Every one is listed in "
        "the first comment, [All changed functions and classes](#all-changed-functions-and-classes).",
        "",
        "## How to review",
        "",
        "1. Read [Chart 0 · Overview](#chart-0--overview): which brain action sends which request.",
        "2. Open one chart at a time. Each starts from what triggers it; every branch is a real condition in the code.",
        "3. [All changed functions and classes](#all-changed-functions-and-classes) lists every one, file by file: "
        "what it does, who calls it, and the *Chart* it is drawn in.",
        "4. [Review notes](#review-notes) lists what the inventory found worth a look.",
        "",
        "## How to read the charts",
        "",
        "| What you see | Meaning |",
        "|---|---|",
        "| Blue / purple / green fill | Runs in the brain node / the pyroki planner node / shared `billie_utils` code the brain calls |",
        "| Thick solid / dashed / thin grey border | New on the branch / changed on the branch / unchanged, shown for context |",
        "| `name()` | A function. `module.name()` when the same name exists in two modules (lowercase before the dot is a module) |",
        "| `Class.method()` | A method, called on an instance of `Class` (capitalized before the dot is a class) |",
        "| `«class» Name` | A class: an instance is created or used here |",
        "| `«data» Name` | A data class: a message or record that only carries data. On a thick arrow: request → reply |",
        "| Rounded pill | What starts the flow |",
        "| Hexagon | A decision, filled like the node it runs in; its arrows name the cases |",
        "| Green box / red box | The move goes ahead / refused or failed |",
        "| Thin arrow / thick arrow / dashed arrow | A call inside one node / a request to the other node, with its name / the other case, when the flow takes the old path |",
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
        ]
        chart = (
            wrap_labels(body)
            + (click_lines(body, number) if preview else "")
            + styles_for(body)
        )
        lines += [
            "<details open>" if number == 0 else "<details>",
            f"<summary><b>Open chart {number}</b></summary>",
            "",
            "```mermaid",
            chart,
            "```",
            "",
            "</details>",
            "",
        ]
        targets = sorted(
            {
                n
                for label in label_strings(body)
                for n in mentioned_charts(label, number)
            }
        )
        if targets:
            lines += [
                "Goes on in: "
                + " · ".join(f"[Chart {n}]({anchor(n)})" for n in targets),
                "",
            ]
        for note in notes:
            lines.append(f"- {link_mentions(note, number)}")
        if notes:
            lines.append("")
    lines += review_notes(rows, charts_of)
    description, lines = "\n".join(lines), []
    lines += [
        "## All changed functions and classes",
        "",
        'Grouped by file. *Chart* says where each one is drawn; "—" means it is a helper of a function that is.',
        "",
    ]
    files = []
    for i, r in enumerate(rows):
        if not files or files[-1][0] != r["file"]:
            files.append((r["file"], []))
        files[-1][1].append(i)
    for path, indexes in files:
        new = sum(rows[i]["status"] == "new" for i in indexes)
        lines += [
            "<details>",
            f"<summary><code>{path.replace('billie/', '')}</code> · {len(indexes)} "
            f"({new} new, {len(indexes) - new} changed)</summary>",
            "",
            table(rows, indexes, charts_of, with_file=False),
            "",
            "</details>",
            "",
        ]
    return description, "\n".join(lines)


def review_notes(rows: list[dict], charts_of: dict[int, list[int]]) -> list[str]:
    """The review notes: code nothing calls, and how much no chart draws.

    rows: Inventory rows. charts_of: Row index -> the charts that draw it.
    Returns: Markdown lines.
    """
    unused = [
        r
        for r in rows
        if not r["used_by"]
        and r["name"] != "main"
        and not r["name"].endswith(("__init__", "__call__"))
    ]
    undrawn = [i for i in range(len(rows)) if i not in charts_of]
    lines = ["## Review notes", ""]
    lines += [
        f"- {kind_marker(r)} in `{r['file'].replace('billie/', '')}`: {users_text(r)}."
        for r in unused
    ]
    lines += [
        f"- {len(undrawn)} of {len(rows)} functions and classes are not drawn in a chart; each is a helper of one "
        "that is (see its *Called from*).",
        "",
    ]
    return lines


# The preview's font, given to mermaid as well: it sizes each box with its own font setting, so a page
# font it did not measure with would overflow the boxes and clip their text.
_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'


def _js_string(text: str) -> str:
    """A JavaScript string literal of text, safe inside a <script> block.

    text: The text. Returns: JSON-encoded text, with "</" written "<\\/" so a "</script" in it cannot end
        the block (inside a JS string that escape decodes back to "</").
    """
    return json.dumps(text).replace("</", "<\\/")


def preview_html(markdown: str) -> str:
    """An HTML page that renders the markdown the way GitHub shows it in a PR (marked + mermaid)."""
    return f"""<title>Arm Planner Call Map</title>
<style>
  /* Layout: GitHub-like reading column, so the preview looks like the PR's document. */
  :root {{ --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --rule: #d1d9e0; --code: #f6f8fa; --accent: #0969da; --chart: #fcfcfa; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --rule: #3d444d; --code: #151b23; --accent: #4493f8; --chart: #eceee9; color-scheme: dark; }} }}
  :root[data-theme="dark"] {{ --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --rule: #3d444d; --code: #151b23; --accent: #4493f8; --chart: #eceee9; color-scheme: dark; }}
  body {{ background: var(--bg); color: var(--fg); font-family: {_FONT}; font-size: 15px; line-height: 1.5; }}
  #doc {{ max-width: 1012px; margin: 0 auto; padding-inline: 16px; padding-block: 24px 64px; }}
  h1 {{ font-size: 2em; border-bottom: 1px solid var(--rule); padding-bottom: .3em; }}
  h2 {{ font-size: 1.5em; border-bottom: 1px solid var(--rule); padding-bottom: .3em; margin-top: 1.6em; }}
  a {{ color: var(--accent); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 85%; background: var(--code); padding: .15em .35em; border-radius: 6px; }}
  table {{ border-collapse: collapse; display: block; overflow-x: auto; margin-block: 8px; }}
  th, td {{ border: 1px solid var(--rule); padding: 6px 12px; vertical-align: top; }}
  th {{ background: var(--code); text-align: left; }}
  details {{ margin-block: 8px; }}
  summary {{ cursor: pointer; }}
  .flow {{ background: var(--chart); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; overflow-x: auto; margin-block: 8px; }}
  .flow svg {{ max-width: none; height: auto; }}
  /* Labels in the font and size mermaid measured them with; never clipped if a browser measures short. */
  .flow foreignObject {{ overflow: visible; }}
  .flow foreignObject div, .flow .nodeLabel, .flow .edgeLabel {{ font-family: {_FONT}; font-size: 14px; line-height: 1.5; }}
  .note {{ color: var(--muted); font-size: .9em; }}
</style>
<div id="doc"><p class="note">Loading the review document…</p></div>
<script>window.REVIEW_MD = {_js_string(markdown)};</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<script>
  // Preview of docs/arm_planner.md as a PR shows it: markdown by marked, charts by mermaid.
  (async () => {{
    const doc = document.getElementById("doc");
    const source = window.REVIEW_MD;
    if (!window.marked) {{ doc.textContent = "The markdown library did not load; reload the page."; return; }}
    doc.innerHTML = marked.parse(source, {{ gfm: true }});
    doc.querySelectorAll("h2").forEach((h) => {{
      h.id = h.textContent.trim().toLowerCase().replace(/[^\\w\\s-]/g, "").replace(/\\s/g, "-");
    }});
    // A link to a chart's heading opens its collapsed chart and scrolls to it.
    const openChart = (hash) => {{
      const target = document.getElementById(decodeURIComponent(hash.replace(/^#/, "")));
      if (!target) return false;
      for (let n = target.nextElementSibling; n && n.tagName !== "H2"; n = n.nextElementSibling) {{
        if (n.tagName === "DETAILS") {{ n.open = true; break; }}
      }}
      target.scrollIntoView({{ behavior: "smooth", block: "start" }});
      return true;
    }};
    document.addEventListener("click", (e) => {{
      const a = e.target.closest ? e.target.closest("a") : null;
      const href = a ? (a.getAttribute("href") || a.getAttribute("xlink:href") || "") : "";
      if (href.startsWith("#") && openChart(href)) e.preventDefault();
    }});
    if (!window.mermaid) return;
    if (document.fonts && document.fonts.ready) await document.fonts.ready;  // measure with the drawn font
    mermaid.initialize({{ startOnLoad: false, theme: "base", securityLevel: "antiscript",  // antiscript: chart clicks on, scripts in labels still removed
     
      flowchart: {{ htmlLabels: true, curve: "basis" }},
      themeVariables: {{ fontFamily: {json.dumps(_FONT)}, lineColor: "#4d5562", fontSize: "14px", edgeLabelBackground: "#fcfcfa", primaryTextColor: "#1d2330" }} }});
    // Fits every box whose text is bigger than the box (the browser drew it wider than Mermaid measured):
    // the text wraps inside the box's width, and the box grows taller (its shape scaled in height around
    // its centre, the border kept at its own width). Returns: How many boxes grew.
    const grow = (box) => {{
      let grown = 0;
      box.querySelectorAll("g.node").forEach((node) => {{
        const fo = node.querySelector("foreignObject");
        const label = fo && fo.querySelector("div");
        const shape = node.querySelector(":scope > rect, :scope > polygon, :scope > path, :scope > circle");
        if (!label || !shape) return;
        const width = Number(fo.getAttribute("width")), height = Number(fo.getAttribute("height"));
        if (label.scrollWidth <= width + 1 && label.scrollHeight <= height + 1) return;
        for (const el of [label, ...label.querySelectorAll("*")]) el.style.whiteSpace = "normal";
        // A long name may break only after "_", "/" or "." (a zero-width space there), not mid-word.
        const walker = document.createTreeWalker(label, NodeFilter.SHOW_TEXT);
        for (let t = walker.nextNode(); t; t = walker.nextNode()) t.nodeValue = t.nodeValue.replace(/([_/.])(?=\\w)/g, "$1\\u200b");
        Object.assign(label.style, {{ display: "block", width: `${{width}}px`, maxWidth: `${{width}}px`, overflowWrap: "break-word" }});
        const needH = Math.max(height, label.scrollHeight);
        const sy = (shape.getBBox().height + needH - height) / shape.getBBox().height;
        fo.setAttribute("height", needH);
        fo.parentElement.setAttribute("transform", `translate(${{-width / 2}}, ${{-needH / 2}})`);
        shape.setAttribute("transform", `scale(1, ${{sy}}) ` + (shape.getAttribute("transform") || ""));
        shape.setAttribute("vector-effect", "non-scaling-stroke");
        grown += 1;
      }});
      return grown;
    }};
    // Sizes a chart's drawing area to everything drawn in it (grown boxes included), with a margin.
    const fit = (box) => {{
      const svg = box.querySelector("svg");
      if (!svg) return;
      const bb = svg.getBBox(), pad = 12;
      const w = bb.width + 2 * pad, h = bb.height + 2 * pad;
      svg.setAttribute("viewBox", `${{bb.x - pad}} ${{bb.y - pad}} ${{w}} ${{h}}`);
      svg.setAttribute("width", w);
      svg.setAttribute("height", h);
      Object.assign(svg.style, {{ width: `${{w}}px`, maxWidth: "none", height: `${{h}}px` }});
    }};
    const blocks = Array.from(doc.querySelectorAll("pre > code.language-mermaid"));
    window.REVIEW_GROWN = 0;  // boxes the page had to grow, for checking
    for (const [i, code] of blocks.entries()) {{
      const box = document.createElement("div");
      box.className = "flow";
      code.parentElement.replaceWith(box);
      try {{
        box.innerHTML = (await mermaid.render(`flow-${{i}}`, code.textContent)).svg;
        window.REVIEW_GROWN += grow(box);
        fit(box);
      }} catch (err) {{ box.textContent = `This chart could not be drawn: ${{err.message || err}}`; }}
    }}
  }})();
</script>
"""


def main() -> None:
    """Writes the PR description, its first comment and the HTML preview (both parts) into output/review/."""
    description, comment = build()
    for path, text in ((OUT_MD, description), (OUT_COMMENT_MD, comment)):
        with open(path, "w") as f:
            f.write(text)
        size = len(text.encode())
        print(
            f"{path}: {size} bytes"
            + (
                ""
                if size <= _GITHUB_MAX_BYTES
                else f" - OVER GitHub's {_GITHUB_MAX_BYTES}"
            )
        )
    with open(OUT_HTML, "w") as f:
        f.write(preview_html("\n".join(build(preview=True))))
    print(OUT_HTML)


if __name__ == "__main__":
    sys.exit(main())
