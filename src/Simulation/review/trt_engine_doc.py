"""Builds trt_engine_load.md, the review document for billie-onboard PR #1321 together with its
vendored Isaac-GR00T PR #44, and an HTML preview of it.

    python -m Simulation.review.branch_inventory --repo <billie-onboard worktree at PR #1321> \
        --base <merge-base with origin/main> \
        --out ../billie-onboard/out/review/trt_engine_load/trt_engine_load_inventory.json
    python -m Simulation.review.branch_inventory --repo <Isaac-GR00T worktree at PR #44> \
        --base <merge-base with origin/main> --subdir . \
        --out ../billie-onboard/out/review/trt_engine_load/gr00t_pr44_inventory.json
    python -m Simulation.review.trt_engine_doc

Two repos, one feature: billie-onboard PR #1321 (feat/standalone-trt-engine-load) adds a standalone
TensorRT ENGINE bundle as a VLA weight source; Isaac-GR00T PR #44 (feat/trt-engine-standalone-bundle,
vendored at billie/vendor/Isaac-GR00T) builds and validates that bundle. The charts are written below
by hand, one per trigger; the tables come from both inventories, so every function and class either
branch adds or changes is listed. Update a chart when the flow it draws changes.
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
    users_text,
    wrap_labels,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# trt_engine_load is developed in billie-onboard, so its review lives there, in out/ (ignored, never pushed).
_OUT_DIR = os.path.abspath(os.path.join(_ROOT, "..", "billie-onboard", "out", "review", "trt_engine_load"))
INVENTORY = os.path.join(_OUT_DIR, "trt_engine_load_inventory.json")
GR00T_INVENTORY = os.path.join(_OUT_DIR, "gr00t_pr44_inventory.json")
OUT_MD = os.path.join(_OUT_DIR, "trt_engine_load.md")
OUT_COMMENT_MD = os.path.join(_OUT_DIR, "trt_engine_load_functions.md")
OUT_HTML = os.path.join(_OUT_DIR, "trt-engine-load-review.html")

# Node styles: fill = process, border = change (New / Chg / Old).
#   u = billie_utils (ai_node.py, vla_reload.py): shared library code the ai-gr00t node calls.
#   b = ai_gr00t: this node's own package (gr00t.py, engine_policy.py, inpainting_policy.py, ...).
#   p = Isaac-GR00T (vendored): the gr00t package that runs inside the same process.
#   d = Isaac-GR00T deployment_scripts: the build/publish pipeline, run offline (a workstation or CI),
#       never on the robot.
STYLES = """  classDef trig fill:#ffffff,stroke:#1d2330,stroke-width:1.5px,color:#1d2330;
  classDef uNew fill:#e1efe4,stroke:#3f7a52,stroke-width:3px,color:#1d2330;
  classDef uChg fill:#e1efe4,stroke:#3f7a52,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef uOld fill:#e1efe4,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef bNew fill:#dbe8fb,stroke:#2f62b0,stroke-width:3px,color:#1d2330;
  classDef bChg fill:#dbe8fb,stroke:#2f62b0,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef bOld fill:#dbe8fb,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef pNew fill:#eadff8,stroke:#6d40b8,stroke-width:3px,color:#1d2330;
  classDef pChg fill:#eadff8,stroke:#6d40b8,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef pOld fill:#eadff8,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef dNew fill:#fbe7d2,stroke:#a95a14,stroke-width:3px,color:#1d2330;
  classDef dChg fill:#fbe7d2,stroke:#a95a14,stroke-width:2px,stroke-dasharray:6 3,color:#1d2330;
  classDef dOld fill:#fbe7d2,stroke:#9aa0a6,stroke-width:1px,color:#1d2330;
  classDef uDec fill:#e1efe4,stroke:#3f7a52,stroke-width:1.5px,color:#1d2330;
  classDef bDec fill:#dbe8fb,stroke:#2f62b0,stroke-width:1.5px,color:#1d2330;
  classDef pDec fill:#eadff8,stroke:#6d40b8,stroke-width:1.5px,color:#1d2330;
  classDef dDec fill:#fbe7d2,stroke:#a95a14,stroke-width:1.5px,color:#1d2330;
  classDef ref fill:#f3f3f1,stroke:#b9b9b4,stroke-width:1px,stroke-dasharray:2 2,color:#8a8a86;
  classDef ok fill:#d6efdc,stroke:#2e7d46,stroke-width:1.5px,color:#1d2330;
  classDef stop fill:#f8dada,stroke:#b3261e,stroke-width:1.5px,color:#1d2330;"""

# Every change that reaches beyond this feature itself, first in the document (review convention item 0).
SYSTEM_WIDE = [
    (
        "Any <code>&lt;model&gt;-ENGINE</code> Hugging Face repo published <i>before</i> this change has only "
        "the <code>*.engine</code> files and <code>engine_shapes.env</code>: it is missing <code>config.json</code>, "
        "<code>trt_auxiliary_weights.pt</code> and <code>engine_manifest.json</code>, all required by "
        "<code>validate_engine_bundle()</code> (chart 8). Loading it — cold boot or <code>override_vla_model</code> — "
        "now fails before any GPU work starts (chart 4)",
        "Every robot whose <code>VLA_MODEL</code> or an override target is an existing <code>-ENGINE</code> id",
        "<b>Must regenerate first:</b> re-run <code>bellboy_tensorrt_deploy.sh</code> (chart 7) and re-upload before "
        "that id can be loaded again. Documented in <code>deployment_scripts/README.md</code>; nothing warns the "
        "operator before they try to switch to a stale one",
    ),
    (
        "<b>Default policy class changed for every Gr00t model.</b> On main every model was built as "
        "<code>Gr00tInpaintingPolicy</code>. Now <code>_policy_cls()</code> builds the vendor <code>Gr00tPolicy</code> "
        "unless <code>USE_MODEL_SIDE_RTC=true</code> (default false) (chart 5). Together with it, "
        "<code>predict()</code> dropped its per-batch <code>reset_action_buffer()</code> call (chart 6): with the vendor "
        "policy there is no inpainting buffer to reset",
        "Every robot running a Gr00t model, PT or ENGINE, with model-side RTC off (the default)",
        "Should match main's output: with RTC off, main reset the buffer before every batch, so the inpainting policy "
        "never had a prefix. Confirm on a robot that PT inference is unchanged before merging",
    ),
    (
        "<code>Gr00tInpaintingPolicy</code> passes <code>prefix_actions</code> / <code>num_prefix_steps</code> only "
        "when a prefix exists (chart 6)",
        "Models with a sibling TensorRT overlay (<code>VLA_MODEL_TRT</code>) and RTC on",
        "<b>Likely a fix:</b> on main the overlay's Isaac <code>action_head_tensorrt_forward(self, backbone_output, "
        "action_input)</code> takes no prefix kwargs, so reading the code, every overlay prediction raised "
        "<code>TypeError</code>. Not confirmed on a robot",
    ),
    (
        "<code>AIRunner._reload_model()</code> rollback path: a failed reload that rolls back to the prior model now "
        "also calls <code>_apply_reload_env(prior_source)</code> (chart 3), restoring <code>VLA_MODEL</code> and "
        "clearing <code>VLA_MODEL_TRT</code> when the prior source was an ENGINE id",
        "Every <code>override_vla_model</code> whose reload fails and rolls back",
        "<b>Regression fix, not a new regression:</b> on main this call was missing on the rollback-success path, "
        "so <code>VLA_MODEL</code> stayed pointed at the failed target after a successful rollback",
    ),
    (
        "Isaac-GR00T CI (<code>.github/actions/setup-venv/action.yml</code>, <code>pyproject.toml</code>): pins "
        "<code>ruff==0.15.21</code> and narrows the default lint rule set to <code>E4/E7/E9/F</code>",
        "Every PR and lint run in the Isaac-GR00T repo",
        "Low: CI-only, unrelated to the TRT bundle feature it shipped alongside",
    ),
    (
        "<code>hf_deploy.py upload-engine</code> now always passes <code>ignore_patterns=[\"*.log\"]</code>",
        "Anyone who inspected build logs from the uploaded HF repo instead of the local build directory",
        "Low: logs stay on the build machine, just stop being uploaded",
    ),
    (
        "billie-onboard's vendored Isaac-GR00T submodule pin (<code>8f6227e4</code>) sits a few commits behind "
        "PR #44's current head (<code>14cdbe8</code>): the gap is a video-frame-seek-tolerance fix to "
        "<code>gr00t/utils/video.py</code> (PR #48, unrelated feature) plus CI/test-only commits. The TRT-bundle "
        "feature code itself (<code>policy.py</code>, <code>deploy_utils.py</code>, <code>hf_deploy.py</code>, "
        "<code>export_onnx.py</code>) is byte-for-byte identical at both points",
        "No functional effect today",
        "Low: worth a note so the submodule bump doesn't silently skip the seek-tolerance fix when this merges",
    ),
]

# What each chart's flow does differently from main, shown under its "When" line.
SUMMARIES = {
    0: "On main every VLA model, Gr00t included, loaded from a PyTorch safetensors checkpoint (plus an optional sibling TensorRT overlay set with <code>model_id_trt</code>). Now a model id ending <code>-ENGINE</code> is its own weight source: a standalone bundle with the engines, the config and the few learned weights TensorRT does not cover, built and checked by a separate pipeline in the vendored Isaac-GR00T repo.",
    1: "New files on both sides: <code>vla_reload.py</code> (billie_utils) names the cargo class; <code>engine_policy.py</code> and <code>trt_inpainting_forward.py</code> (ai_gr00t) hold the engine attach and the TensorRT inpainting forward; <code>deploy_utils.py</code> (Isaac-GR00T) gains the bundling and the validation.",
    2: "On main <code>Gr00tModel.__init__</code> always loaded the PT checkpoint. Now it resolves the id's cargo class first and takes exactly one of two paths (charts 4 and 5). <code>AIRunner</code> also remembers the source it started with, for the reload's rollback (chart 3).",
    3: "On main a reload freed the old model and built the new one, checking only that the id was cached. Now <code>preflight_reload</code> runs first, with the live model still loaded: for an ENGINE id it validates the whole bundle (chart 8) and can refuse the switch. The env is now set from the source: an ENGINE id clears <code>VLA_MODEL_TRT</code>, and a rollback resets <code>VLA_MODEL</code> to the prior model (on main it kept the failed id).",
    4: "New. The ENGINE id's local cache is validated as a complete bundle (chart 8) before anything is built. The PyTorch skeleton then comes from <code>config.json</code> plus <code>trt_auxiliary_weights.pt</code>, with no safetensors checkpoint. The TensorRT engines attach through the same helper the sibling overlay uses (chart 5), which now also keeps the export's <code>VIT_PATCHES</code>.",
    5: "The same steps as main's <code>__init__</code> body, moved into <code>_init_from_pytorch()</code>, with three differences. The policy class now depends on <code>USE_MODEL_SIDE_RTC</code>: off (the default) builds the vendor <code>Gr00tPolicy</code>, where main always built <code>Gr00tInpaintingPolicy</code>. The overlay's engines attach through the new helper (chart 4). The vendored <code>_load_model()</code> moved its horizon alignment into the shared <code>_align_action_horizon()</code>.",
    6: "On main <code>predict()</code> reset the action buffer itself when RTC was off, and always called the inpainting method with <code>executed_count</code>. Now RTC off calls the vendor policy's method with no <code>executed_count</code>. RTC on calls the inpainting method, which passes prefix kwargs only when a prefix exists. With engines attached and RTC on, the head is the new TensorRT inpainting forward, which re-pins the prefix after each Euler step.",
    7: "On main the script went from building the engines straight to the upload. Now it saves <code>trt_auxiliary_weights.pt</code> during the ONNX export, then bundles <code>config.json</code>, <code>experiment_cfg/</code>, the weights and a manifest into the engine directory. It validates that directory, uploads it without <code>*.log</code> files, and validates the downloaded copy before the PyTorch vs TensorRT comparison. The dataset download now resolves each video's <code>original_key</code>.",
    8: "New. The one check both the robot (chart 4, before swapping a live model) and the build pipeline (chart 7, before and after publishing) run against an ENGINE bundle: every required file is present, the shapes and auxiliary-weight files are internally consistent, and the manifest (when present) matches what is actually on disk.",
}

# (number, anchor, title, when, mermaid body, notes)
CHARTS = [
    (
        0,
        "overview",
        "Overview",
        "A general map of the whole feature: every way a Gr00t model's weights get loaded or rebuilt (node boot, an <code>override_vla_model</code> request), how a batch is predicted, and the offline build pipeline that produces an ENGINE repo. Each box names the chart that draws it.",
        """flowchart TD
  subgraph TOP[" "]
    BOOT(["node starts"]):::trig
    OVR(["override_vla_model"]):::trig
    NB(["next_batch"]):::trig
    BLD(["bellboy_tensorrt_deploy.sh<br/>offline, not on the robot"]):::trig
  end
  BOOT --> INIT["<b>Gr00tModel.__init__()</b><br/>ai_gr00t/gr00t.py · chart 2"]:::bChg
  OVR --> RELOAD["<b>AIRunner._reload_model()</b><br/>billie_utils/ai_node.py · chart 3"]:::uChg
  RELOAD -. "preflight, then rebuild" .-> INIT
  INIT --> RSRC{{"<b>resolve_vla_weight_source()</b><br/>vla_reload.py · cargo_class?"}}:::uDec
  RSRC -- "trt_engine" --> ENG["<b>Gr00tModel._init_from_trt_engine()</b><br/>standalone ENGINE bundle · chart 4"]:::bNew
  RSRC -- "pt" --> PT["<b>Gr00tModel._init_from_pytorch()</b><br/>checkpoint + optional sibling overlay · chart 5"]:::bNew
  ENG -. "check the local cache" .-> VAL
  NB --> PRED["<b>Gr00tModel.predict()</b><br/>chart 6"]:::bChg
  BLD --> PIPE["export → build → bundle → validate → publish<br/>deployment_scripts/ (Isaac-GR00T) · chart 7"]:::dChg
  PIPE -. "bundle checks" .-> VAL["<b>validate_engine_bundle()</b><br/>deploy_utils.py · chart 8"]:::pNew
  style TOP fill:none,stroke:none
""",
        [],
    ),
    (
        1,
        "files",
        "Files",
        "Which file holds what, on the robot and in the offline build pipeline.",
        """flowchart TD
  subgraph AIU["billie-onboard · billie_utils"]
    AN["<b>ai_node.py</b>: «class» AIRunner hot reload ·<br/>the Model.preflight_reload hook<br/><b>vla_reload.py</b> (new): «data» VlaWeightSource,<br/>which cargo class a model id is"]:::uChg
  end
  subgraph AIG["billie-onboard · ai_gr00t node"]
    GR["<b>gr00t.py</b>: «class» Gr00tModel · two load paths · predict<br/><b>engine_policy.py</b> (new): attach the engines<br/><b>inpainting_policy.py</b>: prefix kwargs opt-in<br/><b>trt_inpainting_forward.py</b> (new): TRT inpainting head"]:::bChg
  end
  subgraph GRV["Isaac-GR00T, vendored · on the robot"]
    POL["<b>gr00t/model/policy.py</b>: «class» Gr00tPolicy<br/>built from an ENGINE bundle<br/><b>deployment_scripts/deploy_utils.py</b>:<br/>bundle and validate an ENGINE directory"]:::pChg
  end
  subgraph DEP["Isaac-GR00T · offline build pipeline"]
    SH["<b>bellboy_tensorrt_deploy.sh</b>: the whole pipeline<br/><b>export_onnx.py</b>: + the auxiliary weights<br/><b>hf_deploy.py</b>: bundle-engine ·<br/>validate-engine-bundle · upload-engine"]:::dChg
  end
  AN --> GR
  GR -. "build the policy · check the bundle" .-> POL
  SH -- "bundle · validate" --> POL
  style AIU fill:#f1f6ef,stroke:#3f7a52,color:#3f7a52
  style AIG fill:#f1f6fe,stroke:#2f62b0,color:#2f62b0
  style GRV fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
  style DEP fill:#fdf3e9,stroke:#a95a14,color:#a95a14
""",
        [
            "<code>deploy_utils.py</code> is purple, not orange: the robot imports it at load time (chart 8) as well as the build pipeline using it."
        ],
    ),
    (
        2,
        "cold_start",
        "Cold start",
        "The ai-gr00t node starts: <code>main.py</code> builds <code>Gr00tModel()</code> from the Doppler <code>VLA_MODEL</code> / <code>VLA_MODEL_TRT</code> values (unchanged: read at import time as the constructor's defaults).",
        """flowchart TD
  A(["ai-gr00t node starts<br/>main.py → Gr00tModel()"]):::trig --> I["<b>Gr00tModel.__init__()</b><br/>ai_gr00t/gr00t.py"]:::bChg
  I --> R["<b>resolve_vla_weight_source()</b> → «data» VlaWeightSource<br/>billie_utils/vla_reload.py · an empty id → ValueError"]:::uNew
  R --> D{{"<b>is_trt_engine_model_id()</b><br/>the id ends with -ENGINE?"}}:::uDec
  D -- "yes: cargo_class = trt_engine" --> E["standalone ENGINE bundle · chart 4"]:::ref
  D -- "no: cargo_class = pt" --> P["PyTorch checkpoint · chart 5"]:::ref
  E --> RUN
  P --> RUN["<b>AIRunner.__init__()</b> · billie_utils/ai_node.py<br/><b>resolve_vla_weight_source_from_runtime()</b>:<br/>the model's model_id, else VLA_MODEL → _active_weight_source"]:::uChg
  RUN --> OK[["ready: next_batch · override_vla_model (chart 3)"]]:::ok
""",
        [
            "The two load paths are exclusive: one model id takes exactly one of charts 4 and 5.",
        ],
    ),
    (
        3,
        "hot_reload",
        "Hot reload · override_vla_model",
        "An <code>override_vla_model</code> request while the node runs. The switch itself runs on the GPU thread.",
        """flowchart TD
  A(["override_vla_model request"]):::trig --> H["<b>AIRunner._handle_override_vla_model_request()</b><br/>billie_utils/ai_node.py · accept_override()"]:::uOld
  H --> C["<b>AIRunner._compute_override_vla_model()</b><br/>GPU thread"]:::uOld
  C --> SAME{{"this id already running?<br/>yes → switched, nothing reloaded"}}:::uDec
  SAME -- "no" --> SW{{"<b>VlaModelLifecycle.switch_rejection_reason()</b><br/>cached locally? (unchanged: no bundle check here)"}}:::uDec
  SW -- "no" --> REJ[["rejected: download_vla_model first"]]:::stop
  SW -- "yes" --> RM["<b>AIRunner._reload_model()</b><br/>new_source resolved as in chart 2 ·<br/>prior_source = _active_weight_source"]:::uChg
  RM --> PF["model_class.preflight_reload, before any teardown:<br/><b>Gr00tModel.preflight_reload()</b> · ai_gr00t/gr00t.py<br/>overrides <b>Model.preflight_reload()</b>, a no-op (new)"]:::bNew
  PF --> PD{{"new_source is trt_engine?"}}:::bDec
  PD -- "yes" --> VREF["the bundle check · chart 4 → chart 8"]:::ref
  PD -. "no: returns" .-> REL
  VREF -- "invalid: raises" --> FAIL[["refused: the live model stays loaded, never released"]]:::stop
  VREF -- "valid" --> REL["<b>AIRunner._release_model()</b>: close() + empty the CUDA cache"]:::uOld
  REL --> NEW["model_class(**<b>AIRunner._model_reload_kwargs()</b>)<br/>both cargo classes: model_id_trt = None<br/>→ Gr00tModel.__init__, chart 2's branch"]:::uChg
  NEW -- "built" --> APPLY["<b>AIRunner._apply_reload_env()</b>: VLA_MODEL = the new id ·<br/>trt_engine → VLA_MODEL_TRT removed"]:::uNew
  APPLY --> DONE[["switched · _active_weight_source = new_source"]]:::ok
  NEW -- "raised" --> RB["release both · rebuild from prior_source (no preflight)"]:::uChg
  RB -- "rolled back" --> ROLLED[["the switch fails, the prior model runs ·<br/>(new) VLA_MODEL reset to prior_source"]]:::stop
  RB -- "the rollback fails too" --> FATAL[["«class» FatalModelReloadError: the node exits to restart"]]:::stop
""",
        [
            "Teardown happens only after the preflight passes, so a broken ENGINE bundle can't leave the node without a model. A PT target is not pre-checked: as on main, a failure there is caught by the rollback.",
        ],
    ),
    (
        4,
        "trt_engine",
        "Load a standalone ENGINE bundle",
        "<code>Gr00tModel.__init__</code> finds <code>cargo_class = trt_engine</code>: on cold start (chart 2) or a reload (chart 3).",
        """flowchart TD
  A(["Gr00tModel.__init__ · cargo_class = trt_engine<br/>charts 2 and 3"]):::trig --> E["<b>Gr00tModel._init_from_trt_engine()</b><br/>ai_gr00t/gr00t.py"]:::bNew
  E --> V["<b>Gr00tModel.validate_local_engine_bundle()</b><br/>also the reload's preflight · snapshot_download(local_files_only)<br/>deploy_utils imported only here: TensorRT stack not needed for PT"]:::bNew
  V == "the local snapshot" ==> VB["the bundle check · chart 8"]:::ref
  VB -- "not cached / invalid: raises" --> FAIL[["cold start: the node doesn't start ·<br/>reload: refused before teardown (chart 3)"]]:::stop
  VB -- "valid" --> MB["history params, action stats, data config ·<br/>device, policy class · chart 5"]:::ref
  MB --> FEB
  subgraph POLV["gr00t/model/policy.py (vendored)"]
    FEB["<b>Gr00tPolicy.from_engine_bundle()</b> →<br/><b>Gr00tPolicy.__init__()</b> with engine_bundle=True"]:::pChg
    LME["<b>Gr00tPolicy._load_model_from_engine_bundle()</b><br/>config.json → GR00T_N1_5_Config → GR00T_N1_5 (no safetensors) ·<br/>trt_auxiliary_weights.pt → action_head, strict=False<br/>an unexpected key → ValueError"]:::pNew
    ALN["<b>Gr00tPolicy._align_action_horizon()</b><br/>the modality config's horizon differs →<br/>rebuild the action head · shared with chart 5"]:::pNew
  end
  FEB --> LME --> ALN
  ALN --> SE["<b>gr00t._setup_tensorrt_engines()</b> · gr00t.py<br/>shared with the sibling overlay, chart 5"]:::bNew
  SE --> STE["<b>engine_policy.setup_tensorrt_engines()</b><br/>VIT_PATCHES from engine_shapes.env set before, restored after<br/>Isaac's setup_tensorrt_engines (vendored, unchanged)"]:::bNew
  STE --> RTC{{"USE_MODEL_SIDE_RTC?"}}:::bDec
  RTC -- "yes" --> REBIND["action_head.get_action rebound to the<br/>TensorRT inpainting forward · chart 6"]:::ref
  RTC -. "no: the vendor TensorRT get_action stays" .-> POP
  REBIND --> POP["modality, metadata, tasks · chart 5"]:::ref
  POP --> OK[["ready: self.policy runs on the engines"]]:::ok
  style POLV fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [
            "No PyTorch checkpoint is downloaded or loaded on this path: <code>config.json</code> plus the small <code>trt_auxiliary_weights.pt</code> (<code>future_tokens</code>, <code>position_embedding</code>) give the skeleton the engines plug into.",
        ],
    ),
    (
        5,
        "pytorch",
        "Load the PyTorch checkpoint",
        "<code>Gr00tModel.__init__</code> finds <code>cargo_class = pt</code> (chart 2 or 3): the path every model took on main, now its own method. <code>model_id_trt</code> may attach a sibling repo's TensorRT engines on top.",
        """flowchart TD
  A(["Gr00tModel.__init__ · cargo_class = pt<br/>charts 2 and 3"]):::trig --> P["<b>Gr00tModel._init_from_pytorch()</b><br/>ai_gr00t/gr00t.py · was the body of __init__"]:::bNew
  P --> SD["snapshot_download(model_id, local_files_only)"]:::bOld
  SD --> MB["<b>Gr00tModel._load_modality_bundle()</b><br/>load_history_params() · load_action_stats() ·<br/>BellboyDataConfig → modality config + transform"]:::bNew
  MB --> DC{{"<b>_policy_cls()</b>: USE_MODEL_SIDE_RTC?<br/>(new: on main always Gr00tInpaintingPolicy)"}}:::bDec
  DC -- "yes" --> CI["«class» Gr00tInpaintingPolicy"]:::bOld
  DC -- "no (the default)" --> CV["vendor «class» Gr00tPolicy"]:::pOld
  CI --> CTOR
  CV --> CTOR["policy_cls(model_path, …, device=<b>Gr00tModel._policy_device()</b>)"]:::bChg
  CTOR -. "engine_bundle=False" .-> LM["<b>Gr00tPolicy._load_model()</b> · gr00t/model/policy.py<br/>safetensors, then the horizon alignment of chart 4"]:::pChg
  LM --> OV{{"model_id_trt given?"}}:::bDec
  OV -- "no" --> POP
  OV -- "yes: must be model_id + -ENGINE" --> SDT["snapshot_download(model_id_trt, local_files_only)"]:::bOld
  SDT --> SE["attach the engines · chart 4"]:::ref
  SE --> POP["<b>Gr00tModel._populate_contract_from_policy()</b><br/>modality · metadata · tasks · history params"]:::bNew
  POP --> OK[["ready"]]:::ok
""",
        [
            "Only this path layers a sibling <code>-ENGINE</code> repo on a PT checkpoint (the old mechanism). A hot reload never takes it: <code>_model_reload_kwargs()</code> always passes <code>model_id_trt=None</code> (chart 3).",
        ],
    ),
    (
        6,
        "predict",
        "predict()",
        "Every <code>next_batch</code> request, whichever path (chart 4 or 5) built the policy.",
        """flowchart TD
  A(["next_batch request"]):::trig --> PR["<b>Gr00tModel.predict()</b> · ai_gr00t/gr00t.py<br/>(removed: its own action-buffer reset)"]:::bChg
  PR --> RTC{{"USE_MODEL_SIDE_RTC?"}}:::bDec
  RTC -- "no (the default)" --> GA2["vendor Gr00tPolicy's action call, without executed_count<br/>PyTorch or Isaac TensorRT head (unchanged)"]:::pOld
  RTC -- "yes" --> GA1["<b>Gr00tInpaintingPolicy._get_action_from_normalized_input()</b><br/>ai_gr00t/inpainting_policy.py · with executed_count"]:::bChg
  GA1 --> HK{{"a prefix? (prefix_actions and num_prefix > 0)<br/>no → no prefix kwargs (new: were always passed)"}}:::bDec
  HK --> HD{{"engines attached? (chart 4 or the overlay)"}}:::bDec
  HD -- "yes" --> TRT["<b>action_head_trt_inpainting_forward()</b><br/>ai_gr00t/trt_inpainting_forward.py · Isaac's engine graph,<br/>the prefix re-pinned after every Euler step"]:::bNew
  HD -- "no" --> PTH["PyTorch inpainting action head (unchanged)"]:::bOld
  GA2 --> OUT
  TRT --> OUT
  PTH --> OUT[["the batch → AIRunner._prepare_output()"]]:::ok
""",
        [
            "The action buffer is now reset only by <code>AIRunner._begin_inference_episode()</code> (unchanged), once per task: system-wide change 2.",
        ],
    ),
    (
        7,
        "build_pipeline",
        "Build and publish an ENGINE bundle",
        "Run offline (a workstation or CI), never on the robot: <code>bellboy_tensorrt_deploy.sh &lt;MODEL_NAME&gt; &lt;DATASET_NAME&gt;</code>.",
        """flowchart TD
  A(["bellboy_tensorrt_deploy.sh<br/>deployment_scripts/ (changed)"]):::trig --> DM["hf_deploy.py download-model (unchanged)"]:::dOld
  DM --> DD["<b>download_dataset()</b> · hf_deploy.py<br/>video paths from each key's original_key"]:::dChg
  DD --> ONNX["<b>run_groot_inference()</b> · export_onnx.py<br/>+ <b>save_trt_auxiliary_weights()</b> (deploy_utils.py) →<br/>trt_auxiliary_weights.pt; add_pos_embed without the module → AttributeError"]:::dChg
  ONNX --> BE["build_engine.sh (unchanged): *.engine + engine_shapes.env"]:::dOld
  BE --> CLI["<b>hf_deploy.main()</b>: new bundle-engine and<br/>validate-engine-bundle subcommands"]:::dChg
  CLI --> BUNDLE["<b>bundle_engine()</b> → <b>bundle_engine_artifacts()</b><br/>deploy_utils.py"]:::dNew
  subgraph BA["deploy_utils.py"]
    REQ{{"config.json · metadata.json · trt_auxiliary_weights.pt<br/>present, engine_shapes.env complete?"}}:::pDec
    COPY["copy config.json, experiment_cfg/ (replaced whole),<br/>trt_auxiliary_weights.pt into the engine dir ·<br/>weights checked against the config as in chart 8"]:::pNew
    MAN["write engine_manifest.json: source repo,<br/>bundled files, auxiliary keys, engine shapes"]:::pNew
  end
  BUNDLE --> REQ
  REQ -- "no" --> X1[["FileNotFoundError / ValueError"]]:::stop
  REQ -- "yes" --> COPY --> MAN
  MAN --> VB1["its own check of the result · chart 8"]:::ref
  VB1 --> VAL["<b>validate_engine()</b> · validate-engine-bundle<br/>on the engine dir · chart 8"]:::dNew
  VAL --> UP["<b>upload_engine()</b> · hf_deploy.py<br/>to MODEL-ENGINE · *.log files no longer uploaded"]:::dChg
  UP --> DL["download-model of the uploaded repo (unchanged)"]:::dOld
  DL --> VB2["validate-engine-bundle on the download · chart 8"]:::ref
  VB2 --> CMP["gr00t_inference.py: PyTorch vs TensorRT (unchanged)"]:::dOld
  CMP --> DONE[["ENGINE repo published and checked"]]:::ok
  style BA fill:#f6f1fd,stroke:#6d40b8,color:#6d40b8
""",
        [
            "Any validation failure stops the script (<code>set -e</code>-style): a bundle that fails chart 8 is never uploaded.",
        ],
    ),
    (
        8,
        "validate",
        "validate_engine_bundle",
        "Run by the robot before an ENGINE model is used (chart 4: cold start and the reload's preflight), and by the build pipeline after bundling, before upload and after download (chart 7).",
        """flowchart TD
  A(["chart 4 · chart 7"]):::trig --> V["<b>validate_engine_bundle()</b><br/>deployment_scripts/deploy_utils.py"]:::pNew
  V --> F1{{"a directory, with config.json · engine_shapes.env ·<br/>trt_auxiliary_weights.pt · experiment_cfg/metadata.json?"}}:::pDec
  F1 -- "no" --> X1[["FileNotFoundError, naming the missing files"]]:::stop
  F1 -- "yes" --> F2{{"all 7 engines? vit · llm · vlln_vl_self_attention ·<br/>DiT · state_encoder · action_encoder · action_decoder"}}:::pDec
  F2 -- "no" --> X2[["FileNotFoundError"]]:::stop
  F2 -- "yes" --> S{{"<b>_validate_engine_shapes()</b><br/>all 11 shape keys in engine_shapes.env?"}}:::pDec
  S -- "no" --> X3[["ValueError"]]:::stop
  S -- "yes" --> AW{{"<b>_validate_auxiliary_weights()</b>: future_tokens shape matches<br/>action_head_cfg · position_embedding there exactly when<br/>add_pos_embed (default True), shape matches max_seq_len (default 1024)?"}}:::pDec
  AW -- "no" --> X4[["ValueError"]]:::stop
  AW -- "yes" --> MF{{"metadata.json parses · engine_manifest.json present?"}}:::pDec
  MF -- "no manifest" --> OK
  MF -- "yes" --> MC{{"<b>_collect_bundle_files()</b>: the files on disk<br/>(not *.log, .gitattributes, the manifest) = its bundled_files?"}}:::pDec
  MC -- "no" --> X5[["ValueError: missing / extra files listed"]]:::stop
  MC -- "yes" --> OK[["valid: engine shapes + auxiliary weight keys returned"]]:::ok
""",
        [
            "A bundle with no <code>engine_manifest.json</code> passes without the file-list comparison.",
        ],
    ),
]


# [characters] Label line length for the charts GitHub would otherwise shrink below ~85% text size.
_NARROW_LINE_CHARS = {0: 30, 6: 40, 8: 34}


def wrapped(body: str, number: int) -> str:
    """A chart's labels split to fit its boxes, narrower for the wide charts (_NARROW_LINE_CHARS).

    body: The chart's mermaid text. number: Its number. Returns: The wrapped text.
    """
    default = arm_planner_doc._MAX_LINE_CHARS
    arm_planner_doc._MAX_LINE_CHARS = _NARROW_LINE_CHARS.get(number, default)
    try:
        return wrap_labels(body)
    finally:
        arm_planner_doc._MAX_LINE_CHARS = default


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
        """One mention linked. match: A _CHART_MENTION match. Returns: The text."""
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


def _repo_section(title: str, rows: list[dict], indexes_by_file: list[tuple[str, list[int]]], charts_of: dict) -> list[str]:
    """One repo's <details> blocks for the function-table comment. Returns: markdown lines."""
    lines = [f"### {title}", ""]
    for path, indexes in indexes_by_file:
        new = sum(rows[i]["status"] == "new" for i in indexes)
        lines += [
            "<details>",
            f"<summary><code>{path}</code> · {len(indexes)} ({new} new, {len(indexes) - new} changed or removed)</summary>",
            "",
            table(rows, indexes, charts_of, with_file=False),
            "",
            "</details>",
            "",
        ]
    return lines


def system_wide_section() -> list[str]:
    """The document's opening section: every change that reaches beyond the feature. Returns: Lines."""
    lines = [
        "## ⚠️ System-wide changes — confirm each one was intended before merging",
        "",
        "These reach beyond the standalone-ENGINE feature itself. Each one has to be a deliberate choice.",
        "",
        "| # | What changed | Who it affects | Risk |",
        "|---|---|---|---|",
    ]
    for i, (what, who, risk) in enumerate(SYSTEM_WIDE, 1):
        lines.append(f"| {i} | {link_mentions(what, -1)} | {who} | {link_mentions(risk, -1)} |")
    return lines + [""]


# Callers branch_inventory gets wrong here, fixed by hand: (file, name) -> (callers to add, caller prefixes
# to drop). It misses calls through getattr() and calls across the two repos, and it matches any
# super().__init__() to Gr00tPolicy.__init__.
_CALLER_FIXES = {
    ("billie/billie-utils/billie_utils/ai_node.py", "Model.preflight_reload"): (
        ["billie/billie-utils/billie_utils/ai_node.py::AIRunner._reload_model (getattr)"], []),
    ("billie/nodes/ai-gr00t/ai_gr00t/gr00t.py", "Gr00tModel.preflight_reload"): (
        ["billie/billie-utils/billie_utils/ai_node.py::AIRunner._reload_model (getattr)"], []),
    ("billie/billie-utils/billie_utils/ai_node.py", "AIRunner.__init__"): (
        ["billie/nodes/ai-gr00t/ai_gr00t/main.py::main"], []),
    ("billie/nodes/ai-gr00t/ai_gr00t/gr00t.py", "Gr00tModel.__init__"): (
        ["billie/nodes/ai-gr00t/ai_gr00t/main.py::main",
         "billie/billie-utils/billie_utils/ai_node.py::AIRunner._reload_model"], []),
    ("billie/nodes/ai-gr00t/ai_gr00t/gr00t.py", "Gr00tModel.predict"): (
        ["billie/billie-utils/billie_utils/ai_node.py::AIRunner._compute_next_batch"], []),
    ("gr00t/model/policy.py", "Gr00tPolicy.from_engine_bundle"): (
        ["billie/nodes/ai-gr00t/ai_gr00t/gr00t.py::Gr00tModel._init_from_trt_engine"], []),
    ("gr00t/model/policy.py", "Gr00tPolicy.__init__"): (
        ["gr00t/model/policy.py::Gr00tPolicy.from_engine_bundle",
         "billie/nodes/ai-gr00t/ai_gr00t/gr00t.py::Gr00tModel._init_from_pytorch"],
        ["deployment_scripts/export_onnx.py::"]),
    ("deployment_scripts/deploy_utils.py", "validate_engine_bundle"): (
        ["billie/nodes/ai-gr00t/ai_gr00t/gr00t.py::Gr00tModel.validate_local_engine_bundle"], []),
}


def _fix_callers(rows: list[dict]) -> None:
    """Applies _CALLER_FIXES and drops Isaac-GR00T test callers ("tests/..."), in place. rows: Inventory rows."""
    for r in rows:
        add, drop = _CALLER_FIXES.get((r["file"], r["name"]), ([], []))
        kept = {u for u in r["used_by"] if not u.startswith(("tests/", *drop))}
        r["used_by"] = sorted(kept | set(add))


def review_notes(rows: list[dict], charts_of: dict[int, list[int]]) -> list[str]:
    """The review notes: what the reading found, code nothing calls, and how much no chart draws.

    rows: Inventory rows of both repos. charts_of: Row index -> the charts that draw it. Returns: Markdown lines.
    """
    lines = [
        "## Review notes",
        "",
        "- **Not tested on a robot by this review:** the default policy class change (system-wide change 2) and the "
        "sibling-overlay fix (change 3) are read from the code only. Worth one PT run and one ENGINE run on a Billie "
        "before merging.",
        "- Nothing detects a stale <code>-ENGINE</code> cache before an operator switches to it: "
        "<code>switch_rejection_reason()</code> only checks the id is cached, so the switch is accepted and then refused "
        "by the preflight ([chart 3](" + anchor(3) + ")). The live model is safe; the operator only learns at that point.",
        "- `trt_auxiliary_weights.pt` is loaded with `strict=False` and only *unexpected* keys fail "
        "([chart 4](" + anchor(4) + ")); missing keys are printed and tolerated. [Chart 8](" + anchor(8) + ") checks "
        "both expected keys, so a bundle that passed validation can't be missing them.",
        "- A bundle with no `engine_manifest.json` skips the file-list comparison ([chart 8](" + anchor(8) + ")).",
        "- *Called from* for calls across the two repos (e.g. `gr00t.py` → `Gr00tPolicy.from_engine_bundle()`) and "
        "through `getattr()` (the preflight hook) was added by hand; the inventory sees only calls inside one repo.",
    ]
    unused = [
        r for r in rows
        if not r["used_by"] and r["status"] != "removed" and r["name"] != "main"
        and not r["name"].endswith(("__init__", "__call__"))
    ]
    lines += [f"- {kind_marker(r)} in `{r['file']}`: {users_text(r)}." for r in unused]
    undrawn = [i for i in range(len(rows)) if i not in charts_of]
    if undrawn:
        lines.append(f"- {len(undrawn)} of {len(rows)} functions and classes are not drawn in a chart; see their *Called from*.")
    else:
        lines.append(f"- All {len(rows)} changed functions and classes are drawn in a chart.")
    return lines + [""]


def build(preview: bool = False) -> tuple[str, str]:
    """The review document as markdown, in two parts that each fit GitHub's limit.

    preview: Also make the boxes that name another chart clickable (preview only).
    Returns: (the PR description: system-wide changes, guide, charts and review notes; the PR's first
        comment: every changed function and class, grouped by repo then file).
    """
    with open(INVENTORY) as f:
        billie_rows = [r for r in json.load(f)["rows"] if "/tests/" not in r["file"]]
    with open(GR00T_INVENTORY) as f:
        gr00t_rows = [r for r in json.load(f)["rows"] if "/tests/" not in r["file"] and not r["file"].startswith("tests/")]
    rows = billie_rows + gr00t_rows
    _fix_callers(rows)
    charts_of: dict[int, list[int]] = {}
    for number, _, _, _, body, _ in CHARTS:
        for i in sorted(chart_items(body, rows)):
            charts_of.setdefault(i, []).append(number)

    lines = [
        "# trt_engine_load: what changed and how it flows",
        "",
        "Review guide for billie-onboard PR [#1321](https://github.com/bellboy-robotics/billie-onboard/pull/1321) "
        "`feat/standalone-trt-engine-load`, together with its vendored Isaac-GR00T PR "
        "[#44](https://github.com/bellboy-robotics/Isaac-GR00T/pull/44) `feat/trt-engine-standalone-bundle` "
        "(both compared with their own `main`). A model id ending `-ENGINE` is now a standalone TensorRT bundle: "
        "the robot loads it with no PyTorch checkpoint at all, and a separate offline pipeline builds, bundles and "
        "validates it.",
        "",
    ]
    lines += system_wide_section()
    lines += [
        f"**{len(rows)} functions and classes** are added, changed or removed across both repos (tests not counted). "
        "Every one is listed in the first comment, [All changed functions and classes](#all-changed-functions-and-classes).",
        "",
        "## How to review",
        "",
        "1. Confirm each system-wide change in the table above.",
        f"2. Read [Chart 0 · Overview]({anchor(0)}): every way a model's weights get loaded.",
        "3. Open one chart at a time. Each starts from what triggers it; every branch is a real condition in the code.",
        "4. [All changed functions and classes](#all-changed-functions-and-classes) lists every one, repo by repo, file by file.",
        "5. [Review notes](#review-notes) lists what the reading found worth a look.",
        "",
        "## How to read the charts",
        "",
        "| What you see | Meaning |",
        "|---|---|",
        "| Green / blue / purple / orange fill | Shared `billie_utils` code / the ai-gr00t node's own package / "
        "the vendored Isaac-GR00T package (runs on the robot) / Isaac-GR00T's `deployment_scripts` build pipeline "
        "(runs offline, never on the robot) |",
        "| Thick solid / dashed / thin grey border | New on the branch / changed on the branch / unchanged, shown for context |",
        "| Light grey dashed box (`«ref»`) | This chart's function is drawn fully in another chart; this box just points there |",
        "| `name()` | A function. `module.name()` when the same name exists in two modules |",
        "| `Class.method()` | A method, called on an instance of `Class` |",
        "| `«class» Name` / `«data» Name` | A class / a data class (a message or record) |",
        "| Rounded pill | What starts the flow |",
        "| Hexagon | A decision, filled like the node it runs in; its arrows name the cases |",
        "| Green box / red box | Succeeds / refused or failed |",
        "| Thin / thick / dashed arrow | A call / a cross-process or cross-repo step / the other case, an old or unchanged path |",
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
            wrapped(body, number) + (click_lines(body, number) if preview else "") + styles_for(body),
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
        'Grouped by repo, then file. *Chart* says where each one is drawn; "—" means it is a helper of a function '
        "that is, a test, or a removed one.",
        "",
    ]

    def grouped(rows: list[dict]) -> list[tuple[str, list[int]]]:
        files: list[tuple[str, list[int]]] = []
        offset = len(billie_rows) if rows is gr00t_rows else 0
        for j, r in enumerate(rows):
            if not files or files[-1][0] != r["file"]:
                files.append((r["file"], []))
            files[-1][1].append(j + offset)
        return files

    lines += _repo_section(
        "billie-onboard · PR #1321", rows, grouped(billie_rows), charts_of
    )
    lines += _repo_section(
        "Isaac-GR00T (vendored) · PR #44", rows, grouped(gr00t_rows), charts_of
    )
    return description, "\n".join(lines)


def styles_for(body: str) -> str:
    """The classDef lines of the node styles a chart uses. body: The chart's mermaid text. Returns: The lines."""
    used = set(re.findall(r":::(\w+)", body))
    return "\n".join(line for line in STYLES.split("\n") if line.split()[1] in used)


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
        "<title>Arm Planner Call Map</title>", "<title>TRT Engine Load Review</title>"
    )
    with open(OUT_HTML, "w") as f:
        f.write(page)
    print(OUT_HTML)


if __name__ == "__main__":
    sys.exit(main())
