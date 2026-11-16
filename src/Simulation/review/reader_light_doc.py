"""Builds reader_light_task_switch.md, the PR #1378 review document, and an HTML preview of it.

    python -m Simulation.review.branch_inventory --repo billie-onboard \
        --out billie-onboard/out/review/reader_light_task_switch/reader_light_inventory.json
    python -m Simulation.review.reader_light_doc

Same method and convention as arm_planner_doc.py / free_area_doc.py, whose chart-independent helpers
this reuses: the charts are written below by hand, one per trigger; the tables come from the inventory,
so every function and class the branch adds or changes is listed. Update a chart when the flow it draws
changes. Reviewed in the Simulation repo's billie-onboard submodule (branch feat/reader-light-task-switch),
not the person's own billie-onboard checkout, so the branch under review never has to be checked out there.
"""

import json
import os
import re
import sys

import Simulation.review.arm_planner_doc as planner_doc
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
# Reviewed in the submodule checkout (Simulation/billie-onboard), not the sibling repo: the branch under
# review was checked out there instead of in the person's own billie-onboard working copy.
_OUT_DIR = os.path.abspath(os.path.join(_ROOT, "billie-onboard", "out", "review", "reader_light_task_switch"))
INVENTORY = os.path.join(_OUT_DIR, "reader_light_inventory.json")
OUT_MD = os.path.join(_OUT_DIR, "reader_light_task_switch.md")
OUT_COMMENT_MD = os.path.join(_OUT_DIR, "reader_light_task_switch_functions.md")
OUT_HTML = os.path.join(_OUT_DIR, "reader-light-task-switch-review.html")

# Node styles: fill = process (b brain, s spatial_awareness node, u billie_utils), border = change.
STYLES = """  classDef trig fill:#ffffff,stroke:#1d2330,stroke-width:1.5px,color:#1d2330;
  classDef bNew fill:#dbe8fb,stroke:#2f62b0,stroke-width:3px,color:#1d2330;
  classDef bChg fill:#dbe8fb,stroke:#2f62b0,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef bOld fill:#dbe8fb,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef sNew fill:#eadff8,stroke:#6d40b8,stroke-width:3px,color:#1d2330;
  classDef sChg fill:#eadff8,stroke:#6d40b8,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef sOld fill:#eadff8,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef uNew fill:#e1efe4,stroke:#3f7a52,stroke-width:3px,color:#1d2330;
  classDef uChg fill:#e1efe4,stroke:#3f7a52,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef uOld fill:#e1efe4,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef bDec fill:#dbe8fb,stroke:#2f62b0,stroke-width:1.5px,color:#1d2330;
  classDef sDec fill:#eadff8,stroke:#6d40b8,stroke-width:1.5px,color:#1d2330;
  classDef ok fill:#d6efdc,stroke:#2e7d46,stroke-width:1.5px,color:#1d2330;
  classDef stop fill:#f8dada,stroke:#b3261e,stroke-width:1.5px,color:#1d2330;"""

# branch_inventory's name matching misses a call through an instance attribute, never the class name
# itself: ai.py calls self._node.reader_light.take(), and never writes "ReaderLightTracker" as text, so
# _refers_to's "the owner's class name must appear in the file" check fails. Fixed by hand, as in
# free_area_doc.py's _CALLER_FIXES: (file, name) -> (callers to add, caller prefixes to drop).
_CALLER_FIXES = {
    ("billie/nodes/brain/brain/reader_light.py", "ReaderLightTracker.take"): (
        ["billie/nodes/brain/brain/behaviours/ai.py::_Attempt._event_loop"],
        [],
    ),
}


def _fix_callers(rows: list[dict]) -> None:
    """Applies _CALLER_FIXES to the inventory rows, in place. rows: Inventory rows."""
    for r in rows:
        add, drop = _CALLER_FIXES.get((r["file"], r["name"]), ([], []))
        r["used_by"] = sorted({u for u in r["used_by"] if not u.startswith(tuple(drop))} | set(add))


# Every change that reaches beyond the reader-light feature itself (the review convention's item 0):
# (what changed, who it affects, risk). Filled from the whole branch diff against the merge-base.
SYSTEM_WIDE = [
    (
        "<code>dataflow.py</code>: brain gains access to the <code>spatial_awareness/reader_light</code> input, "
        "spatial_awareness gains access to the <code>reader_light</code> output (chart 2)",
        "Every robot's generated <code>dataflow.yml</code>",
        "With an old, un-regenerated <code>dataflow.yml</code> the <code>reader_light</code> output never "
        "reaches brain (dora does not deliver an output the node doesn't declare), so every "
        "<code>reader_light=True</code> attempt ends \"no green before the task stopped moving\". Regenerate "
        "<code>dataflow.yml</code> and deploy brain and spatial_awareness together",
    ),
    (
        "New shared messages in <code>billie_utils.messages.spatial_awareness</code>: "
        "<code>WatchReaderLightRequest</code>, <code>ReaderLightState</code>, <code>watch_reader_light()</code>",
        "brain and the spatial_awareness node, both of which import this module",
        "Deploy <code>billie-utils</code> together with both nodes, as any shared message addition",
    ),
    (
        "<code>OrchestratorNode</code> (brain/node.py) now always holds a <code>ReaderLightTracker</code> and "
        "subscribes to <code>spatial_awareness/reader_light</code> unconditionally — not only when an "
        "<code>ai_task</code> asks for it",
        "Every brain tick, whether or not <code>reader_light</code> is ever used",
        "Low: <code>ReaderLightTracker.on_state()</code> is a no-op while nothing is watching. Listed because "
        "it changes shared dispatch code (<code>node.py</code>)",
    ),
]

# What each chart's flow does differently from main, shown under its "When" line.
SUMMARIES = {
    0: "On main an <code>ai_task</code> sequence always moved from one task to the next the moment the arm "
    "went stationary. Now, with <code>reader_light=True</code>, the first task also watches the card-reader "
    "light: a separate green switches it to the next task (or ends a single task as a success) instead of "
    "waiting on stillness, and going stationary with no green fails the whole attempt. Later tasks still "
    "switch on stillness exactly as before. A second, unrelated change rides along: an intermediate task's "
    "timeout sentence is now the activity's reported error, not a generic \"AI task failed\".",
    1: "Three new/changed files split by node: brain's <code>behaviours/ai.py</code> (the attempt reacting to "
    "the light) and new <code>reader_light.py</code> (what \"the first green\" means); spatial_awareness's "
    "<code>main.py</code> (the watch request, the gripper tick) and new <code>reader_light.py</code> (classify "
    "+ debounce). A new <code>billie_utils</code> message ties the two nodes together.",
    2: "New. spatial_awareness classifies the gripper camera's raw pixels every tick while a watch is on, "
    "debounces a flashing green into real red / green / none pulses, and publishes one <code>reader_light</code> "
    "output per frame. Brain's <code>ReaderLightTracker</code> ignores anything published before the \"none\" "
    "that this watch's own start forces, so a state left over from an earlier watch can never be read as "
    "this one's first green.",
    3: "New. <code>_Attempt.run()</code> starts the watch once the arm has reached its starting joints, so that "
    "move itself is never mistaken for a reaction to the light. Every loop iteration checks for a taken green "
    "before anything else; the three endings for the first task (switch, succeed, fail without a green) all "
    "stop the watch first. A reverse playback is skipped when the light itself ended the attempt. In-flight AI "
    "batches and pyroki solves left over from the task being switched away from are explicitly dropped rather "
    "than applied to the new one.",
    4: "On main every failed attempt reported the same generic \"AI task failed\" as the activity's error, "
    "whatever the cause. Now an intermediate task's timeout keeps its own sentence on the attempt "
    "(<code>timeout_failure</code>), carried across <code>starting_joints</code> retries, and raised as the "
    "activity error if every retry fails. Unrelated to the reader light: this applies to every "
    "<code>ai_task</code> sequence that sets a <code>duration</code>.",
}

# (number, anchor, title, when, mermaid body, notes)
CHARTS = [
    (
        0,
        "overview",
        "Overview",
        "An <code>ai_task</code> command with <code>reader_light=True</code> and a <code>\" | \"</code>-separated "
        "task sequence.",
        """flowchart TD
  A(["ai_task command · reader_light=True<br/>a &quot; | &quot;-separated task sequence"]):::trig --> E["<b>execute_policy()</b> → <b>_AITask.execute()</b><br/>brain/behaviours/ai.py"]:::bChg
  E --> R["<b>_Attempt.run()</b>: starts the watch once the arm<br/>has reached its starting joints"]:::bChg
  R == "watch_reader_light · chart 2" ==> SA["spatial_awareness classifies the gripper camera,<br/>publishes reader_light · chart 2"]:::sNew
  SA == "reader_light · «data» ReaderLightState" ==> BT["brain: «class» ReaderLightTracker keeps<br/>the first green · chart 2"]:::bNew
  R --> LOOP["the event loop reacts to the light · chart 3"]:::bChg
  BT -. "take(): the first green, once" .-> LOOP
  LOOP -- "task 1: green, not the last task" --> NEXT["hold the arm · start the next task · chart 3"]:::bNew
  LOOP -- "green, the last (or only) task" --> DONE[["attempt succeeds"]]:::ok
  LOOP -- "task 1: stationary with no green" --> FAIL[["attempt fails · chart 3"]]:::stop
  LOOP -- "a later task stops moving" --> SAME["advances exactly as before reader_light existed"]:::bOld
  E -. "an intermediate task's duration elapses · chart 4<br/>(independent of reader_light)" .-> TO["its own sentence becomes<br/>the activity's reported error"]:::bChg
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
    AI["<b>behaviours/ai.py</b>: starts/stops the watch,<br/>reacts to the light inside the attempt · chart 3"]:::bChg
    ND["<b>node.py</b>: holds the «class» ReaderLightTracker,<br/>routes spatial_awareness/reader_light to it"]:::bChg
    RL["<b>reader_light.py</b>: «class» ReaderLightTracker —<br/>the first green since this watch's own &quot;none&quot;"]:::bNew
  end
  U["<b>billie_utils/messages/spatial_awareness.py</b><br/>«data» WatchReaderLightRequest · «data» ReaderLightState · watch_reader_light()"]:::uNew
  subgraph SAN["spatial_awareness"]
    MAIN["<b>main.py</b>: the watch request handler,<br/>the gripper tick"]:::sChg
    SRL["<b>reader_light.py</b>: classify_reader_light() ·<br/>«class» LightPulse (debounce) · «class» ReaderLightWatch"]:::sNew
  end
  RL ~~~ U
  AI --> U
  ND --> U
  U ==> MAIN
  MAIN --> SRL
  AI -.-> RL
  ND --> RL
  style BRAIN fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
  style SAN fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [],
    ),
    (
        2,
        "light",
        "Watching and classifying the light",
        "<code>_Attempt._start_reader_light_watch()</code> / <code>_end_reader_light_watch()</code> "
        "(chart 3), and every gripper camera tick while a watch is on.",
        """flowchart TD
  subgraph TOP[" "]
    W(["_start_reader_light_watch() · _end_reader_light_watch()<br/>chart 3"]):::trig
    G(["gripper camera tick"]):::trig
  end
  W == "watch_reader_light · «data» WatchReaderLightRequest" ==> H["<b>on_watch_reader_light()</b><br/>spatial_awareness/main.py"]:::sNew
  H --> HR["reader_light_enabled = request.enabled ·<br/>«class» ReaderLightWatch.reset() → «class» LightPulse.reset()"]:::sNew
  HR --> PUB1["_publish_reader_light(): one &quot;none&quot;, every time —<br/>so a leftover state can't leak into the next watch"]:::sNew
  G --> OC["<b>on_camera_gripper()</b> → <b>_gripper_rgb()</b><br/>spatial_awareness/main.py"]:::sChg
  OC --> EN{{"reader_light_enabled?"}}:::sDec
  EN -- "yes" --> CL["ReaderLightWatch.step() → <b>classify_reader_light()</b><br/><b>_color_counts()</b> · <b>_inner_pixels()</b> (reader_light.py)<br/>raw pixels, not the JPEG: compression smears the LED's color"]:::sNew
  CL --> LP["LightPulse.step() → LightPulse._advance() ·<br/>LightPulse._green_is_separate()<br/>red: at once · a flash: none · a separate green: confirmed"]:::sNew
  LP --> PUB2["_publish_reader_light(): green or none<br/>«data» ReaderLightState"]:::sNew
  EN -- "no" --> SKIP["not watched: only <b>_jpeg_payload()</b><br/>for the usual gripper snapshot"]:::sOld
  PUB1 == "reader_light" ==> RT
  PUB2 == "reader_light" ==> RT["<b>OrchestratorNode._handle_reader_light()</b><br/>brain/node.py"]:::bNew
  RT --> TR["ReaderLightTracker.on_state()<br/>brain/reader_light.py<br/>ignores states before this watch's own first &quot;none&quot;"]:::bNew
  TR -. "take(): the first green, once" .-> EL["the attempt's event loop · chart 3"]:::bOld
  style TOP fill:none,stroke:none
""",
        [
            "<code>on_camera_gripper()</code> still calls <code>state.update_camera_gripper()</code> with the "
            "JPEG on every frame, watched or not — the gripper snapshot other callers read is unaffected.",
            "<code>_encode_gripper_frame()</code> is no longer called by <code>on_camera_gripper()</code> (split "
            "into <code>_gripper_rgb()</code> + <code>_jpeg_payload()</code>, called directly) — it now has no "
            "caller outside its own 3 tests, and its docstring's \"used elsewhere\" is stale.",
        ],
    ),
    (
        3,
        "reacting",
        "Reacting to the light",
        "<code>_Attempt.run()</code> with <code>tp.reader_light</code>: every iteration of "
        "<code>_event_loop()</code>, and the existing <code>_on_ai_batch_response()</code> path that used to "
        "always advance an intermediate task.",
        """flowchart TD
  A(["_Attempt.run() starts · tp.reader_light"]):::trig --> SW["<b>_start_reader_light_watch()</b><br/>ReaderLightTracker.start() first, then watch_reader_light(True) (chart 2), after the move to starting_joints: that move isn't watched<br/>run()'s finally: still watching → _end_reader_light_watch(), also on an abort"]:::bNew
  SW --> LOOP["<b>_event_loop()</b>: every event from safe_next()"]:::bChg
  LOOP --> TAKE{{"watching, and a green taken?<br/>ReaderLightTracker.take()"}}:::bDec
  TAKE -- "yes" --> GL["<b>_on_green_light()</b><br/>_end_reader_light_watch() first · chart 2"]:::bNew
  GL --> LAST{{"the last (or only) task?"}}:::bDec
  LAST -- "yes" --> STOPOK["<b>_stop_for_reader_light()</b>, INFO: stop accepting AI/IK · clear the queues · <b>_hold_arm()</b><br/>→ the attempt succeeds · run() skips the reverse playback"]:::bNew
  LAST -- "no" --> SWITCH["<b>_switch_for_green_light()</b>: drop the old task's in-flight AI batch and pyroki solve ·<br/>_advance_to_next_task() · _hold_arm()"]:::bNew
  SWITCH -- "the next task" --> LOOP
  TAKE -- "no" --> SWOG{{"_stationary_without_green set?<br/>yes → <b>_stop_for_reader_light()</b>, WARNING: no green before the task stopped moving → the attempt fails"}}:::bDec
  SWOG -- "no" --> REST["rest of the loop, unchanged:<br/>pyroki, batches, navigation, task timeout · chart 4"]:::bOld
  REST -- "an AI batch arrives" --> STAT
  subgraph BATCH["_on_ai_batch_response()"]
    STAT{{"an intermediate task goes stationary<br/>(switch_after_stationary, or the completion monitor)"}}:::bDec
    EIT["<b>_end_intermediate_task()</b>"]:::bNew
    COND{{"reader_light on, and (still watching, or this is task 1)?<br/>yes, still watching → _stationary_without_green = True, read on the next event"}}:::bDec
  end
  STAT --> EIT --> COND
  COND -- "no: the watch has stopped, a later task" --> ADV["<b>_advance_to_next_task()</b>: switches as before reader_light existed"]:::bOld
  style BATCH fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
""",
        [
            "<b>_ignore_pyroki_result()</b> drops an IK/pyroki response or error that was requested before the "
            "switch (<code>_drop_next_pyroki_response</code>) or that arrives after <code>_stop_for_reader_light()</code> "
            "(<code>_reader_light_stopped</code>) — used by <code>_on_pyroki_response()</code> and <code>_on_pyroki_error()</code>, "
            "neither redrawn here since only their first line changed.",
            "<code>_on_ai_batch_response()</code>'s own late-response guard (<code>_discard_pending_ai_for_light</code>) "
            "throws away an AI batch that was requested for the task before the green, the same way a stale "
            "request id always was — not redrawn, since the rest of that function is unchanged.",
            "<i>Later tasks never auto-advance</i> was flagged by Cursor Bugbot against an earlier commit "
            "(<code>a3bd9ce5</code>) and is already fixed on this branch (<code>f2912be3</code>, "
            "<code>test_a_later_task_advances_once_the_reader_light_watch_has_stopped</code>) — the "
            "<i>reader_light on, and (still watching, or this is task 1)?</i> decision above is that fix.",
        ],
    ),
    (
        4,
        "timeout",
        "Intermediate task timeout",
        "A tick while an intermediate task's <code>duration</code> has elapsed. Independent of "
        "<code>reader_light</code> — applies to every multi-task sequence with a duration set.",
        """flowchart TD
  A(["tick, a duration is set on the current task"]):::trig --> C["<b>_check_task_timeout()</b><br/>brain/behaviours/ai.py"]:::bChg
  C --> E{{"elapsed ≥ this task's duration?"}}:::bDec
  E -- "no" --> N[["loop continues"]]:::ok
  E -- "yes" --> L{{"the last (or only) task?"}}:::bDec
  L -- "yes" --> S["DEBUG message · event loop returns True"]:::bOld
  S --> V[["attempt succeeds · verify runs, as before"]]:::ok
  L -- "no" --> T["self.timeout_failure = the timeout sentence ·<br/>ERROR cloud message · event loop returns False"]:::bChg
  T --> F[["attempt fails"]]:::stop
  F --> X["<b>_AITask.execute()</b>: kept across starting_joints retries —<br/>the last attempt's timeout_failure wins"]:::bChg
  X --> R{{"every attempt failed?"}}:::bDec
  R -- "yes" --> RAISE[["raise RuntimeError(timeout_failure or &quot;AI task failed&quot;)<br/>the activity's reported error"]]:::stop
  R -- "no: a later attempt succeeded" --> OK2[["AI task completed"]]:::ok
""",
        [],
    ),
]


# [characters] Narrower label lines for the widest charts: GitHub's 880px column shrinks a wide chart's
# text with it, and these two came out at 44% and 80% of full size with the default 46.
_NARROW_LINE_CHARS = {1: 30, 3: 34}


def wrap_for(body: str, number: int) -> str:
    """A chart's labels wrapped at its own line length (_NARROW_LINE_CHARS, else the shared default).

    body: The chart's mermaid text. number: The chart's number.
    Returns: The chart with long label lines split, as wrap_labels() does.
    """
    default = planner_doc._MAX_LINE_CHARS
    planner_doc._MAX_LINE_CHARS = _NARROW_LINE_CHARS.get(number, default)
    try:
        return wrap_labels(body)
    finally:
        planner_doc._MAX_LINE_CHARS = default


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
        "These reach beyond the reader-light feature itself. Each one has to be a deliberate choice.",
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
        "- **Already fixed on this branch, not open issues:** Cursor Bugbot's review (against commit "
        "`a3bd9ce5`, before several fix commits) flagged a dump-writer-thread deadlock and a silently-defaulted "
        "`READER_LIGHT_DUMP_DIR` env var in `spatial_awareness/reader_light.py` — that whole frame-dump feature "
        "was later removed outright (`25d63a5e`), so neither exists on the branch tip. It also flagged "
        "\"later tasks never auto-advance\" — fixed by `f2912be3`, see [Chart 3](" + anchor(3) + ")'s notes.",
        "- **Still true, low risk:** `_end_reader_light_watch()` logs and continues if the `watch_reader_light(False)` "
        "request itself fails, so spatial_awareness could in theory keep watching after brain has moved on — the "
        "next `_start_reader_light_watch()` resets it regardless, and a failed fire-and-forget request here has "
        "nothing else to retry against.",
        "- The ROI and HSV thresholds in `spatial_awareness/reader_light.py` (`_ROI`, `_RED_HUE`, `_GREEN_HUE`, "
        "`_MAX_PIXELS`, …) are tuned to one camera's framing of the gripper and the reader LED; a lens, mount or "
        "gripper change would need them re-tuned, same as the free_area branch's area-check constants.",
        "- `_stop_for_reader_light()`'s `_hold_arm()` runs `state.set_state_with_arm_joints(..., wait=True)` from "
        "inside the event loop handling the green/no-green event — same blocking-hold pattern the rest of the "
        "codebase already uses, not new to this branch.",
    ]
    unused = [
        r
        for r in rows
        if not r["used_by"]
        and r["status"] != "removed"
        and r["name"] != "main"
        and not r["name"].endswith(("__init__", "__call__"))
    ]
    lines += [
        f"- {kind_marker(r)} in `{r['file'].replace('billie/', '')}`: {users_text(r)}"
        + (f" ({r['tests']} tests call it directly)." if r["tests"] else ".")
        for r in unused
    ]
    undrawn = [i for i in range(len(rows)) if i not in charts_of]
    lines += [
        f"- {len(undrawn)} of {len(rows)} functions and classes are not drawn in a chart; each is a small "
        "helper of one that is (constructors, `__init__`s and the like), or a leaf of the classification math "
        "covered by chart 2's description.",
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
        rows = [r for r in json.load(f)["rows"] if "/tests/" not in r["file"] and "/test/" not in r["file"]]
    _fix_callers(rows)
    charts_of: dict[int, list[int]] = {}
    for number, _, _, _, body, _ in CHARTS:
        for i in sorted(chart_items(body, rows)):
            charts_of.setdefault(i, []).append(number)
    lines = [
        "# reader_light_task_switch (PR #1378): what changed and how it flows",
        "",
        "Review guide for `feat/reader-light-task-switch` (compared with `main`). An `ai_task` can now watch "
        "the card-reader light on the gripper camera: a separate green on the first task of a ` | `-separated "
        "sequence holds the arm and starts the next task (or ends a single task as a success) instead of "
        "waiting for the arm to go stationary. A second, independent change rides along: an intermediate "
        "task's timeout now reports its own sentence as the activity's error.",
        "",
    ]
    lines += system_wide_section()
    lines += [
        f"**{len(rows)} functions and classes** are added, changed or removed (tests not counted). Every one "
        "is listed in the first comment, [All changed functions and classes](#all-changed-functions-and-classes).",
        "",
        "## How to review",
        "",
        "1. Confirm each system-wide change in the table above.",
        "2. Read [Chart 0 · Overview](#chart-0--overview): the whole feature in one picture.",
        "3. Open one chart at a time. Each starts from what triggers it; every branch is a real condition in the code.",
        "4. [All changed functions and classes](#all-changed-functions-and-classes) lists every one, file by file.",
        "5. [Review notes](#review-notes) lists what the reading found, including which Bugbot findings are already fixed.",
        "",
        "## How to read the charts",
        "",
        "| What you see | Meaning |",
        "|---|---|",
        "| Blue / purple / green fill | Runs in the brain node / the spatial_awareness node / shared `billie_utils` code both import |",
        "| Thick solid / dashed / thin grey border | New on the branch / changed on the branch / unchanged, shown for context |",
        "| `name()` | A function. `module.name()` when the same name exists in two modules |",
        "| `Class.method()` | A method, called on an instance of `Class` |",
        "| `«class» Name` / `«data» Name` | A class / a data class (a message). On a thick arrow: request → reply |",
        "| Rounded pill | What starts the flow |",
        "| Hexagon | A decision, filled like the node it runs in; its arrows name the cases |",
        "| Green box / red box | Succeeds / fails |",
        "| Thin / thick / dashed arrow | A call in one node / a request to the other node / the other case, an old or unchanged path |",
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
            wrap_for(body, number) + (click_lines(body, number) if preview else "") + styles_for(body),
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
    """Writes the PR description, its first comment and the HTML preview into the submodule's out/review/."""
    description, comment = build()
    os.makedirs(_OUT_DIR, exist_ok=True)
    for path, text in ((OUT_MD, description), (OUT_COMMENT_MD, comment)):
        with open(path, "w") as f:
            f.write(text)
        size = len(text.encode())
        print(f"{path}: {size} bytes" + ("" if size <= _GITHUB_MAX_BYTES else f" - OVER GitHub's {_GITHUB_MAX_BYTES}"))
    page = preview_html("\n".join(build(preview=True))).replace(
        "<title>Arm Planner Call Map</title>", "<title>Reader Light Task Switch Review</title>"
    )
    with open(OUT_HTML, "w") as f:
        f.write(page)
    print(OUT_HTML)


if __name__ == "__main__":
    sys.exit(main())
