"""Builds door_verify_duration.md, the PR #1394 review document, and an HTML preview of it.

    python -m Simulation.review.branch_inventory --repo billie-onboard \
        --base fd5bd56bfda75ac6af108ca1bfc80dab962ced50 \
        --out billie-onboard/out/review/door_verify_duration/door_verify_inventory.json
    python -m Simulation.review.door_verify_duration_doc

--base is the PR's own base (the tag v0.0.886-reader-light-trt-door), not origin/main: that tag is not
yet merged to main, so the default merge-base would pull in the whole reader-light/trt-door stack instead
of just this PR's five commits. Same method and convention as arm_planner_doc.py / reader_light_doc.py,
whose chart-independent helpers this reuses: the charts are written below by hand, one per trigger; the
tables come from the inventory. Reviewed in the Simulation repo's billie-onboard submodule (branch
feat/door-verify-duration), not the person's own billie-onboard checkout (on a different branch), so the
branch under review never has to be checked out there.
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

# branch_inventory's name matching misses that a dataclass field can share a function's name: node.py's
# ExecutionOptions.door_verify field (not a call) reads as a reference to door_verify() scoped to
# "ExecutionOptions", the same false-positive shape as free_area_doc.py's _CALLER_FIXES.
_CALLER_FIXES = {
    ("billie/nodes/brain/brain/commands/spatial_awareness.py", "door_verify"): (
        [],
        ["billie/nodes/brain/brain/node.py::ExecutionOptions"],
    ),
}


def _fix_callers(rows: list[dict]) -> None:
    """Applies _CALLER_FIXES to the inventory rows, in place. rows: Inventory rows."""
    for r in rows:
        add, drop = _CALLER_FIXES.get((r["file"], r["name"]), ([], []))
        r["used_by"] = sorted({u for u in r["used_by"] if not u.startswith(tuple(drop))} | set(add))


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Reviewed in the submodule checkout (Simulation/billie-onboard): the branch under review was checked
# out there instead of the person's own billie-onboard working copy, which is on a different branch.
_OUT_DIR = os.path.abspath(os.path.join(_ROOT, "billie-onboard", "out", "review", "door_verify_duration"))
INVENTORY = os.path.join(_OUT_DIR, "door_verify_inventory.json")
OUT_MD = os.path.join(_OUT_DIR, "door_verify_duration.md")
OUT_COMMENT_MD = os.path.join(_OUT_DIR, "door_verify_duration_functions.md")
OUT_HTML = os.path.join(_OUT_DIR, "door-verify-duration-review.html")

# Node styles: fill = process (b brain, u shared billie_utils). Border = change status.
# No spatial_awareness-node color: this PR never touches that node's own files, only brain and billie_utils.
STYLES = """  classDef trig fill:#ffffff,stroke:#1d2330,stroke-width:1.5px,color:#1d2330;
  classDef bNew fill:#dbe8fb,stroke:#2f62b0,stroke-width:3px,color:#1d2330;
  classDef bChg fill:#dbe8fb,stroke:#2f62b0,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef bOld fill:#dbe8fb,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef uNew fill:#e1efe4,stroke:#3f7a52,stroke-width:3px,color:#1d2330;
  classDef uChg fill:#e1efe4,stroke:#3f7a52,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef bDec fill:#dbe8fb,stroke:#2f62b0,stroke-width:1.5px,color:#1d2330;
  classDef ok fill:#d6efdc,stroke:#2e7d46,stroke-width:1.5px,color:#1d2330;
  classDef stop fill:#f8dada,stroke:#b3261e,stroke-width:1.5px,color:#1d2330;"""

# Every change that reaches beyond the door_verify-duration feature itself (the review convention's
# item 0): (what changed, who it affects, risk). Filled from the whole branch diff against the merge-base
# (fd5bd56b, the PR's own base tag), not from memory.
SYSTEM_WIDE = [
    (
        "Every <code>door_verify</code> failure text now starts with <code>&quot;door_verify: &quot;</code> "
        "(was <code>&quot;Door verification failed: …&quot;</code>) — the text that lands in "
        "<code>activity_error</code>",
        "Every existing command that uses the <code>door_verify</code> execution option or the standalone "
        "<code>door_verify</code> command, not only <code>ai_task</code> with a new <code>duration</code>",
        "Low in this repo (nothing greps for the old text), but any cloud dashboard or alert that matches "
        "activity errors on <code>&quot;Door verification failed&quot;</code> stops matching",
    ),
    (
        "A <code>door_verify</code> lidar read (the standalone command and the after-command check) now "
        "fails after 10s with no answer, instead of waiting forever (<code>door_status()</code> gains an "
        "optional <code>timeout</code>; door_verify always passes 10s)",
        "Every existing <code>door_verify</code> use, whether or not it sets <code>duration</code>",
        "Intended per the PR description, but a command that relied on an unbounded wait (e.g. a very slow "
        "door) now fails instead of eventually succeeding",
    ),
    (
        "<code>validate_door_verify_option()</code> now runs before the command's own <code>condition</code> "
        "check in <code>execute_command()</code> (chart 2) — it used to run after, so a bad "
        "<code>door_verify</code> was previously skipped together with the command when the condition failed",
        "Any command combining <code>door_verify</code> with <code>condition</code>",
        "A command that depended on the old skip-together behaviour (a bad door_verify silently never "
        "evaluated because its condition was false) now fails validation up front instead",
    ),
    (
        "<code>door_verify</code> validation is stricter: a boolean <code>angle</code> (e.g. "
        "<code>angle: true</code>, previously accepted as <code>1.0</code> via <code>float(True)</code>) "
        "is now rejected, and non-numeric or negative values raise up front",
        "Any existing command config that passed a boolean or otherwise non-numeric <code>angle</code>/"
        "<code>duration</code>",
        "Low — no such config exists in this repo — but a live task definition on the cloud side could have one",
    ),
]

# What each chart's flow does differently from main, shown under its "When" line.
SUMMARIES = {
    0: "On main, <code>door_verify</code> only ever checked the door once, after the whole command ran. "
    "Now an <code>ai_task</code> can set a <code>duration</code> so each ` | `-separated task is held to its "
    "own deadline during the run — read once when the task's time runs out, or when the task ends first — "
    "while the after-command check still runs as a final verification. Every other command's "
    "<code>door_verify</code> keeps checking once, after the command, but with two independent changes that "
    "apply to it too: a 10s lidar timeout instead of an unbounded wait, and validation moved ahead of the "
    "<code>condition</code> check.",
    1: "Four changed files split by what they do: <code>billie_utils</code>'s message module gains a "
    "non-blocking door request; brain's <code>commands/spatial_awareness.py</code> holds the shared "
    "validation, watch and after-command check; <code>behaviours/ai.py</code> holds the new in-loop, "
    "per-task checks; <code>node.py</code> wires the two together with one new attribute.",
    2: "New branching inside <code>execute_command()</code>: a task count for <code>ai_task</code> commands, "
    "and a validation call now placed before <code>condition</code> instead of after it. "
    "<code>validate_door_verify_option()</code> itself gained the per-task (`&quot; | &quot;`) number lists "
    "and the stricter type checks in <code>_door_verify_values()</code>.",
    3: "<code>door_verify_attempt()</code> now also decides what <code>node.door_verify_check</code> holds "
    "during the command (chart 4 reads it), and restores the outer command's check afterwards even if the "
    "command raised. The after-command check (<code>door_verify_option()</code> → <code>door_verify()</code>) "
    "is unchanged in shape but now reads the last angle above 0 when <code>angle</code> is a per-task list, "
    "fails on a 10s lidar timeout, and prefixes every failure with <code>door_verify:</code>.",
    4: "New. While an <code>ai_task</code> carries its own <code>door_verify</code> duration, every tick can "
    "queue and send one non-blocking door request per task, and judge the reply on a later tick. The event "
    "loop itself never blocks: a reply arrives through a callback and is only judged on the next tick, so a "
    "slow or missing lidar answer cannot hang the loop or swallow <code>$stop</code> and navigation events.",
    5: "New. Once the loop returns, the attempt either drops any check still in flight (it failed for another "
    "reason) or judges everything not yet judged — blocking is safe here since the loop has ended. A door "
    "miss in any <code>starting_joints</code> attempt makes the whole <code>ai_task</code> command retriable, "
    "and its activity error becomes the door sentence instead of the generic <code>&quot;AI task failed&quot;</code>.",
}

# (number, anchor, title, when, mermaid body, notes)
CHARTS = [
    (
        0,
        "overview",
        "Overview",
        "A command runs with a <code>door_verify</code> execution option — most commonly "
        "<code>ai_task</code> with <code>duration: &quot;0 | 6 | 10&quot;</code>.",
        """flowchart TD
  A(["command with a door_verify option<br/>e.g. ai_task, duration: &quot;0 | 6 | 10&quot;"]):::trig --> V["validate_door_verify_option(): before anything<br/>moves, now ahead of condition · chart 2"]:::bChg
  V --> ATT["door_verify_attempt() wraps the command · chart 3<br/>brain/commands/spatial_awareness.py"]:::bChg
  ATT -- "duration set" --> SET["node.door_verify_check = the resolved spec"]:::bNew
  ATT -- "no duration" --> CLR["node.door_verify_check = None"]:::bOld
  SET --> BODY
  CLR --> BODY["the command body runs"]:::bOld
  BODY -. "ai_task reads node.door_verify_check" .-> LOOP["each task's door deadline,<br/>checked on every tick · chart 4"]:::bNew
  LOOP -- "a task's door is short" --> HOLD[["hold the arm · attempt fails · chart 5"]]:::stop
  LOOP -- "every task reached its angle in time" --> END["attempt ends · chart 5 judges<br/>whatever was never sent or answered"]:::bNew
  BODY --> AFTER
  END --> AFTER["the after-command check: door_verify_option()<br/>· chart 3 — the last angle above 0"]:::bChg
  HOLD --> OUT1[["RetriableError(&quot;door_verify: …&quot;) → attempts retries it<br/>activity error = the door sentence"]]:::stop
  AFTER -- "miss" --> OUT1
  AFTER -- "ok" --> OUT2[["command succeeds"]]:::ok
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
    SPA["<b>commands/spatial_awareness.py</b><br/>door_verify() · validate_door_verify_option()<br/>door_verify_attempt() · door_verify_option()<br/>begin_door_verify() · door_verify_final_angle()"]:::bChg
    AI["<b>behaviours/ai.py</b><br/>execute_policy(): per-task door_angles/door_deadlines via<br/>parse_ai_task_durations(), stored on «data» _TaskParams<br/>_Attempt: the in-loop checks · charts 4, 5<br/>_AITask.execute(): the activity error"]:::bChg
    ND["<b>node.py</b><br/>ExecutionOptions.door_verify (doc only)<br/>OrchestratorNode.door_verify_check<br/>execute_command(): validates before condition"]:::bChg
  end
  U["<b>billie_utils/messages/spatial_awareness.py</b><br/>request_door_status() (new, non-blocking)<br/>door_status() gains an optional timeout"]:::uChg
  SPA ==> U
  AI -.-> U
  ND -.-> SPA
  style BRAIN fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
""",
        [],
    ),
    (
        2,
        "validating",
        "Validating before anything runs",
        "<code>OrchestratorNode.execute_command()</code>, whenever <code>options.door_verify</code> is set — "
        "for every command, not only <code>ai_task</code>.",
        """flowchart TD
  A(["OrchestratorNode.execute_command(): options.door_verify is not None"]):::trig --> AI{{"cmd.name == &quot;ai_task&quot;?"}}:::bDec
  AI -- "yes" --> CNT["<b>count_ai_tasks()</b>: arguments → task_count,<br/>or None if the arguments don't say"]:::bNew
  AI -- "no" --> NONE["task_count = None"]:::bOld
  CNT --> VAL
  NONE --> VAL["<b>validate_door_verify_option()</b>: spec, command=cmd.name, task_count<br/>moved ahead of the condition check · was after it"]:::bChg
  VAL --> ANG["<b>_door_verify_values()</b>: angle, …, rejects bool / non-numeric /<br/>negative, splits a string on &quot; | &quot; per task"]:::bNew
  ANG --> MULTI{{"more than one angle value?"}}:::bDec
  MULTI -- "yes" --> DREQ{{"a duration is given, and<br/>at least one angle &gt; 0?"}}:::bDec
  MULTI -- "no" --> POS{{"the single angle &gt; 0?"}}:::bDec
  DREQ -- "no" --> FAIL1[["ValueError: rejected up front"]]:::stop
  POS -- "no" --> FAIL1
  DREQ -- "yes" --> DUR
  POS -- "yes" --> DUR{{"duration given?"}}:::bDec
  DUR -- "yes" --> AITASK{{"command == ai_task, and per-task<br/>counts match task_count?"}}:::bDec
  AITASK -- "no" --> FAIL1
  AITASK -- "yes" --> OK[["validated"]]:::ok
  DUR -- "no" --> OK
  OK --> COND["the command's own condition check runs next<br/>(can move the arm) — unchanged"]:::bOld
""",
        [
            "<code>door</code> and <code>keep</code> checks are unchanged (a door number must be an int, "
            "<code>keep</code> must be <code>true</code>/<code>false</code>) and are not redrawn here.",
        ],
    ),
    (
        3,
        "watching",
        "Watching the command, and the after-command check",
        "<code>door_verify_attempt(node, spec)</code>, the context manager every command with "
        "<code>door_verify</code> runs inside.",
        """flowchart TD
  A(["door_verify_attempt(node, spec): wraps the command body"]):::trig --> SAVE["outer_check = node.door_verify_check<br/>restored in finally, even if the body raises"]:::bNew
  SAVE --> DIR{{"spec.direction given?"}}:::bDec
  DIR -- "yes" --> BEGIN["begin_door_verify(): door_watcher(direction) →<br/>watched_door"]:::bOld
  DIR -- "no" --> SKIP["watched_door = None<br/>an already-watched door is left alone"]:::bOld
  BEGIN --> MERGE["check_spec = spec, with door=watched_door,<br/>direction=None"]:::bOld
  SKIP --> MERGE2["check_spec = spec"]:::bOld
  MERGE --> DUR{{"spec.duration given?"}}:::bDec
  MERGE2 --> DUR
  DUR -- "yes" --> SETCHK["node.door_verify_check = check_spec<br/>ai_task's event loop reads it · chart 4"]:::bNew
  DUR -- "no" --> CLRCHK["node.door_verify_check = None<br/>keeps an outer command's check out"]:::bNew
  SETCHK --> RUN
  CLRCHK --> RUN["yield: the command body runs"]:::bOld
  RUN --> FINAL["<b>door_verify_option()</b> → door_verify_final_angle():<br/>the last angle above 0 when angle is a per-task list"]:::bChg
  FINAL --> DS["<b>_door_status_in_time()</b>: door_status(timeout=10s)<br/>TimeoutError → RetriableError(&quot;door_verify: …&quot;)"]:::bNew
  DS --> MISS{{"angle / age / finite, as before?"}}:::bDec
  MISS -- "miss" --> ERR[["RetriableError(&quot;door_verify: …&quot;)<br/>every failure text now starts this way"]]:::stop
  MISS -- "ok" --> PASS[["check passes"]]:::ok
  ERR --> REST["finally: node.door_verify_check = outer_check ·<br/>drop the watch unless spec.keep (only if started here)"]:::bNew
  PASS --> REST
""",
        [
            "An inner command's <code>door_verify</code> (e.g. a nested <code>ai_task</code>) saves and "
            "restores the outer one through the same <code>outer_check</code>/<code>finally</code>, so a "
            "nested call can never leak its check spec into, or lose, the command that called it.",
            "<code>begin_door_verify()</code>'s own error message gains the <code>door_verify:</code> prefix; "
            "its door_watcher() call is otherwise unchanged.",
        ],
    ),
    (
        4,
        "in-loop",
        "The in-loop check, once per task",
        "A <code>tick</code> event inside <code>_Attempt._event_loop()</code>, while "
        "<code>self._tp.door_deadlines</code> is set (an <code>ai_task</code> whose own "
        "<code>door_verify</code> option carries a <code>duration</code>).",
        """flowchart TD
  A(["tick · self._tp.door_deadlines is not None"]):::trig --> CHK["<b>_Attempt._check_door_deadline()</b>"]:::bNew
  CHK --> TO{{"a request in flight ≥ 10s?"}}:::bDec
  TO -- "yes" --> TOF["<b>_Attempt._fail_door()</b>: &quot;no answer from<br/>the lidar within 10s&quot;"]:::bNew
  TO -- "no" --> JUDGE
  TOF --> JUDGE["<b>_Attempt._judge_door_answer()</b>: a stored reply? judge it via<br/><b>_Attempt._judge_door()</b> against its queued (task, angle) —<br/>_fail_door() keeps only the first miss"]:::bNew
  JUDGE --> FAILED{{"self.door_failure now set?"}}:::bDec
  FAILED -- "yes" --> HOLD["<b>_Attempt._fail_attempt_on_door()</b>: ERROR message · _hold_arm()<br/>→ event loop returns False"]:::stop
  FAILED -- "no" --> DL{{"this task's deadline elapsed?"}}:::bDec
  DL -- "yes" --> Q["<b>_Attempt._queue_door_check()</b>:<br/>&quot;Ns after the task started&quot;"]:::bNew
  DL -- "no" --> SEND
  Q --> SEND{{"nothing in flight, and<br/>the queue has a check?"}}:::bDec
  SEND -- "yes" --> REQ["<b>request_door_status()</b>: non-blocking send<br/>billie_utils/messages/spatial_awareness.py"]:::uNew
  SEND -- "no" --> CONT[["loop continues"]]:::ok
  REQ -. "reply, whenever it arrives" .-> CB["<b>_Attempt._on_door_status()</b>: event, node, request_id —<br/>a stale request_id (attempt moved on) is dropped"]:::bNew
  CB --> STORE["kept as self._door_answer,<br/>judged on the next tick above"]:::bNew
""",
        [
            "<code>_advance_to_next_task()</code> (an existing function, changed) also calls "
            "<code>_queue_door_check(&quot;when the task ended&quot;)</code> before moving the task index, "
            "and <code>_start_door_deadline()</code> to reset the clock for the next task — queuing does no "
            "I/O, so it is safe to call from the batch-response handler, not only from a tick.",
            "<code>_door_deadline()</code> and <code>_door_number()</code> are small readers "
            "(<code>self._tp.door_deadlines[self._task_index]</code>, <code>self._tp.door_check[\"door\"]</code>) "
            "used throughout this chart and chart 5; not drawn as their own boxes.",
        ],
    ),
    (
        5,
        "ending",
        "Ending the attempt, and the activity error",
        "<code>_Attempt.run()</code>, once its event loop returns — and, above that, "
        "<code>_AITask.execute()</code> across every <code>starting_joints</code> attempt.",
        """flowchart TD
  A(["_Attempt.run(): the event loop has returned"]):::trig --> S{{"succeeded_so_far?"}}:::bDec
  S -- "no" --> DROP["<b>_Attempt._drop_door_checks()</b>: forget anything not judged<br/>a reply that still arrives is then ignored"]:::bNew
  DROP --> FAIL1
  S -- "yes" --> FIN["<b>_Attempt._finish_door_deadline()</b>"]:::bNew
  FIN --> INFLIGHT{{"a check still in flight?"}}:::bDec
  INFLIGHT -- "yes" --> WAIT["node.wait_for_response(remaining of 10s)<br/>safe to block: the loop has ended"]:::bNew
  INFLIGHT -- "no" --> QUEUE
  WAIT --> QUEUE["queues the last task's own check · chart 4 —<br/>any earlier task's check never sent is read now,<br/>labelled &quot;(read when the attempt ended)&quot;"]:::bNew
  QUEUE --> PEND{{"anything pending, and<br/>no failure yet?"}}:::bDec
  PEND -- "yes" --> ONE["one blocking door_status(timeout=10s)<br/>judges every pending check against that one reading"]:::bNew
  PEND -- "no" --> DONE2
  ONE --> DONE2{{"self.door_failure set?"}}:::bDec
  DONE2 -- "yes" --> HOLD2["holds the arm and reports the door failure · chart 4"]:::stop
  DONE2 -- "no" --> OK2[["returns True: the attempt stands"]]:::ok
  HOLD2 --> FAIL1["succeeded_so_far = False"]:::bOld
  OK2 --> DONE3[["AI task completed"]]:::ok
  FAIL1 --> EXEC["<b>_AITask.execute()</b>: door_failure = this attempt's,<br/>or an earlier attempt's — the first miss found wins,<br/>kept across starting_joints retries"]:::bChg
  EXEC --> ANY{{"every attempt failed,<br/>and door_failure set?"}}:::bDec
  ANY -- "yes" --> RE[["raise RetriableError(&quot;door_verify: …&quot;)<br/>attempts retries it · activity error = the door sentence"]]:::stop
  ANY -- "no: no door failure" --> RT[["raise RuntimeError(&quot;AI task failed&quot;)<br/>unchanged"]]:::stop
""",
        [
            "<code>_hold_arm()</code> is the same function the card-reader-light feature already uses "
            "(<code>_stop_for_reader_light()</code>, <code>_switch_for_green_light()</code>); this PR only "
            "generalises its error message from &quot;…for the reader light&quot; to &quot;Cannot hold the "
            "arm: …&quot; and adds a third caller, <code>_fail_attempt_on_door()</code>.",
            "A failed hold (<code>_hold_arm()</code> itself raising) is logged but does not replace "
            "<code>door_failure</code>: the command still ends up retriable with the door sentence as its "
            "activity error, not a generic arm-hold error.",
        ],
    ),
]


# [characters] Narrower label lines for the widest charts: GitHub's 880px column shrinks a wide chart's
# text with it.
_NARROW_LINE_CHARS = {1: 34, 2: 34, 4: 34, 5: 34}


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
        "These reach beyond the new `duration` feature itself: every existing `door_verify` use is affected. "
        "Each one has to be a deliberate choice.",
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
        "- **Testing gaps the PR itself discloses:** the full `./test.sh`, the real event loop end to end "
        "with a door check, an actual activity record with `attempts` retrying it, and a run on the robot are "
        "all listed as not yet covered.",
        "- `_judge_door()` picks the highest-numbered watched door when `door` is `None`, the same rule the "
        "existing standalone `door_verify()` uses (`max(reports, key=...)`) — consistent, not redrawn twice.",
        "- `DOOR_REPLY_TIMEOUT_S` (behaviours/ai.py) and `DOOR_STATUS_TIMEOUT_S` (commands/spatial_awareness.py) "
        "are the same 10s value, defined twice in two files rather than shared from one place.",
        "- `_fail_door()` keeps only the **first** miss per attempt; a later, different miss (e.g. the lidar "
        "times out after an earlier task already came up short) is never reported, which matches \"the first "
        "failure is the one reported\" in its own docstring.",
        "- **Inventory tool noise, not real callers:** `branch_inventory`'s name matching is not type-aware. "
        "`ExecutionOptions.door_verify` (a dataclass field, same name as the `door_verify()` function) showed "
        "up as a caller of it and is removed by hand (`_CALLER_FIXES`, as in `reader_light_doc.py`). "
        "`__init__`'s generic name still matches unrelated classes' constructors in the table below (e.g. "
        "`OrchestratorNode.__init__`'s \"Called from\") — a known limitation, not specific to this branch.",
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
        f"- {len(undrawn)} of {len(rows)} functions and classes are not drawn in a chart: "
        "`_Attempt.__init__`/`_Attempt._reset_door_checks`/`_Attempt._start_door_deadline` (per-attempt "
        "setup, called out in charts 4 and 5's notes instead of their own boxes), the small readers "
        "`_Attempt._door_deadline()` and `_Attempt._door_number()`, `_ai_task_failure()` (assembles the "
        "activity error string chart 5 already shows the outcome of), `ExecutionOptions` and "
        "`OrchestratorNode.__init__` (doc-only / a new attribute, not new logic), and the unchanged bulk of "
        "`_Attempt._event_loop()` and `_Attempt._hold_arm()` (only their door-check call site, or error "
        "message, changed — both are drawn there, charts 4 and 5).",
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
        "# door_verify_duration (PR #1394): what changed and how it flows",
        "",
        "Review guide for `feat/door-verify-duration` (compared with its own base, the "
        "`v0.0.886-reader-light-trt-door` tag — not yet on `main`). `door_verify` can now check the door "
        "**during** an `ai_task`, task by task, with a `duration` per ` | `-separated task, instead of only "
        "once after the whole command.",
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
        "5. [Review notes](#review-notes) lists what the reading found.",
        "",
        "## How to read the charts",
        "",
        "| What you see | Meaning |",
        "|---|---|",
        "| Blue / green fill | Runs in the brain node / shared `billie_utils` code brain imports |",
        "| Thick solid / dashed / thin grey border | New on the branch / changed on the branch / unchanged, shown for context |",
        "| `name()` | A function. `module.name()` when the same name exists in two modules |",
        "| `Class.method()` | A method, called on an instance of `Class` |",
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
        "<title>Arm Planner Call Map</title>", "<title>Door Verify Duration Review</title>"
    )
    with open(OUT_HTML, "w") as f:
        f.write(page)
    print(OUT_HTML)


if __name__ == "__main__":
    sys.exit(main())
