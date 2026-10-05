"""World editor: a browser page to place Billie on the map, put objects around it, pick the joints and
poses to test, and run them through the planner.

    python -m Simulation.world_sim.editor [post_across_path.json] [--robot bellboy@billie-29.bellboy ...]

The page (editor.html) is served by this process, which reuses the plan and view steps:
- Billie is drawn from the planner's merged URDF; clearance comes from the planner's collision model
  (cached in output/world_sim/arm_model.json after the first build, so editing works while it compiles)
  and from the URDF's real collision meshes (mesh_check.py), which the capsules do not always cover;
- objects are drawn from world.object_capsules, so the page shows the capsules the planner gets;
- Run plans the edited scenario on this machine and/or robots over ssh (runs.py), archives each
  run's results and reports in output/world_sim/runs/ and plays the path on the page;
- random objects and motions come from randomize.py.
Scenarios are saved to src/Simulation/world_sim/scenarios/, ready for plan.py and robot.py.
"""

import argparse
import glob
import json
import logging
import os
import re
import threading
import traceback
import webbrowser
import xml.etree.ElementTree as ET
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

import numpy as np
from billie_utils.messages.pyroki_world_poc import DEFAULT_ARM_BASE_HEIGHT_M, MAX_WORLD_CAPSULES, WORLD_COL_MARGIN_M
from billie_utils.world_collision_check_poc import CollisionModel, link_obstacle_distances, link_transforms
from scipy.spatial.transform import Rotation

from Simulation.world_sim import robot, runs
from Simulation.world_sim.analysis import EEF_LINKS, Run, dense_path, link_distances, load_run, planner_gripper_box
from Simulation.world_sim.mesh_check import EXACT_WITHIN_M, link_mesh_distances, urdf_path
from Simulation.world_sim.randomize import random_motions, random_objects
from Simulation.world_sim.pushables import push_events, simulate_pushes
from Simulation.world_sim.world import PUSHABLE_TYPES, hinge_of, object_capsules, scenario_from_data

_HERE = os.path.dirname(os.path.abspath(__file__))
EDITOR_HTML = os.path.join(_HERE, "editor.html")
SCENARIOS_DIR = os.path.join(_HERE, "scenarios")
OUTPUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "output", "world_sim"))
MODEL_CACHE = os.path.join(OUTPUT_DIR, "arm_model.json")
# Local port of the page; any free port works (--port).
DEFAULT_PORT = 8765
# Decimals of the arm poses sent to the page (0.1mm, ~1e-4 quaternion): enough to draw, keeps replies small.
_DECIMALS = 4
# Scenario file names the page may read or write: one plain name inside SCENARIOS_DIR.
_FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.json$")


class EditorState:
    """What the request handlers share: the arm model and the planner, built in the background."""

    def __init__(self, use_planner: bool, robots: list[str]):
        """Loads the cached arm model and starts the planner build.

        use_planner: Build the planner (JAX) for Run and Solve IK; without it the page edits and previews only.
        robots: ssh destinations of the robots offered as run targets.
        """
        self.robots = robots
        self.model: CollisionModel | None = _cached_model()
        self.planner = None  # SimPlanner once built
        self.status = "building" if use_planner else "off"  # planner state shown on the page
        self.lock = threading.Lock()  # the planner runs one call at a time
        self.meshes: list[str] = []  # mesh files of the URDF sent to the page, by URL index
        self._model_ready = threading.Event()
        if self.model is not None:
            self._model_ready.set()
        elif not use_planner:
            raise SystemExit("No cached arm model yet: run the editor once without --no-planner.")
        if use_planner:
            threading.Thread(target=self._build_planner, daemon=True).start()

    def _build_planner(self) -> None:
        """Builds the planner (compile or JAX cache load) and caches its arm model for later sessions."""
        try:
            from Simulation.world_sim.planner import SimPlanner  # noqa: PLC0415 (imports JAX)

            self.planner = SimPlanner()
            self.model = self.planner.model
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(MODEL_CACHE, "w") as f:
                f.write(self.model.to_json())
            self.status = "ready"
        except Exception as e:  # shown on the page; editing goes on with the cached model
            logging.exception("Planner build failed")
            self.status = f"error: {e}"
        finally:
            self._model_ready.set()

    def require_model(self) -> CollisionModel:
        """The arm model, waiting for the planner build if no cached one exists.

        Returns: The model. Raises: RuntimeError if the planner build failed without a cached model.
        """
        self._model_ready.wait()
        if self.model is None:
            raise RuntimeError(f"No arm model: the planner build failed ({self.status})")
        return self.model

    def require_planner(self):
        """The built planner.

        Returns: The SimPlanner. Raises: RuntimeError while it is building, or if it is off or failed.
        """
        if self.planner is None:
            raise RuntimeError(f"The planner is not ready ({self.status})")
        return self.planner


def _cached_model() -> CollisionModel | None:
    """The arm model saved by an earlier session, or the one inside any plan results file.

    Returns: The model, or None if this machine never built the planner.
    """
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE) as f:
            return CollisionModel.from_json(f.read())
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "model_json" in data:
            return CollisionModel.from_json(data["model_json"])
    return None


def joint_limits_deg() -> list[list[float]]:
    """Limits of the 6 arm joints from the planner's merged URDF.

    Returns: 6 [lower, upper] pairs in degrees.
    """
    limits = [[-360.0, 360.0] for _ in range(6)]  # replaced below; every arm joint has a limit
    for joint in ET.parse(urdf_path()).getroot().findall("joint"):
        name, limit = joint.get("name", ""), joint.find("limit")
        if limit is not None and re.fullmatch(r"joint[1-6]", name):
            limits[int(name[-1]) - 1] = [np.rad2deg(float(limit.get(k))) for k in ("lower", "upper")]
    return limits


def robot_urdf(state: EditorState) -> bytes:
    """The merged URDF for the page, its mesh files renamed to mesh/<index><ext> URLs (served at /mesh/).

    state: The editor state; its mesh list is replaced by this URDF's mesh files (the only files /mesh serves).
    Returns: The URDF XML.
    """
    tree = ET.parse(urdf_path())
    meshes = []
    for mesh in tree.getroot().iter("mesh"):
        path = mesh.get("filename", "")
        if path not in meshes:
            meshes.append(path)
        mesh.set("filename", f"mesh/{meshes.index(path)}{os.path.splitext(path)[1]}")  # relative to /urdf
    state.meshes = meshes
    return ET.tostring(tree.getroot())


def mesh_file(state: EditorState, url_path: str) -> bytes:
    """One mesh file named in the URDF last sent to the page.

    state: The editor state. url_path: "/mesh/<index><ext>" as robot_urdf wrote it.
    Returns: The file's bytes. Raises: FileNotFoundError for any other path.
    """
    match = re.fullmatch(r"/mesh/(\d+)\.\w+", url_path)
    if match is None or int(match.group(1)) >= len(state.meshes):
        raise FileNotFoundError(url_path)
    with open(state.meshes[int(match.group(1))], "rb") as f:
        return f.read()


def model_info(state: EditorState, _query: dict) -> dict:
    """What the page needs to draw Billie: link capsules, joint limits and defaults for a new scenario.

    state: The editor state. _query: Unused.
    Returns: {"links": [{"name", "radius", "height", "movable"}], "joint_limits_deg": 6 [lo, hi],
        "arm_to_base": default [x_mm, y_mm, yaw_deg], "floor_z_mm", "max_capsules", "margin_mm", "planner",
        "robots": robot ssh destinations, "robot_checkout": default planner code checkout on the robots}.
    """
    model = state.require_model()
    # The arm base in the robot frame, read from the URDF's mobile base link under the arm. The
    # robot's ARM_TO_BASE_CALIBRATION may differ slightly; a scenario can set its own.
    arm_to_base = [0.0, 0.0, 0.0]
    if "mobile_base_link" in model.link_names:
        T_arm_from_base = link_transforms(model, np.zeros((1, 6)))[0, model.link_names.index("mobile_base_link")]
        T_base_from_arm = np.linalg.inv(T_arm_from_base)
        yaw = np.rad2deg(np.arctan2(T_base_from_arm[1, 0], T_base_from_arm[0, 0]))
        arm_to_base = [round(T_base_from_arm[0, 3] * 1000.0, 1), round(T_base_from_arm[1, 3] * 1000.0, 1), round(yaw, 2)]
    links = [{"name": n, "radius": float(r), "height": float(h), "movable": bool(m)}
             for n, r, h, m in zip(model.link_names, model.capsule_radius, model.capsule_height,
                                   model.movable_links)]  # fmt: skip
    return {
        "links": links,
        "joint_limits_deg": joint_limits_deg(),
        "arm_to_base": arm_to_base,
        "floor_z_mm": -DEFAULT_ARM_BASE_HEIGHT_M * 1000.0,
        "max_capsules": MAX_WORLD_CAPSULES,
        "margin_mm": WORLD_COL_MARGIN_M * 1000.0,
        "planner": state.status,
        "robots": state.robots,
        "robot_checkout": robot.DEFAULT_ROBOT_CHECKOUT,
    }


def arm_poses(model: CollisionModel, joints_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where every link capsule and the TCP are, for drawing the arm.

    model: The arm model.
    joints_deg: (S, 6) configurations, degrees.
    Returns: (capsule poses (S, L, 7) [x, y, z m, qx, qy, qz, qw] - the capsule is centered on its
        frame and runs along its Z; TCP poses (S, 6) [x, y, z mm, rx, ry, rz rad]), xArm base frame.
    """
    T = link_transforms(model, np.deg2rad(np.atleast_2d(joints_deg)))
    capsule = T @ model.capsule_local
    S, L = capsule.shape[:2]
    quat = Rotation.from_matrix(capsule[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(S, L, 4)
    poses = np.concatenate([capsule[..., :3, 3], quat], axis=-1).round(_DECIMALS)
    tcp = T[:, model.link_names.index("link_tcp")]
    tcp_pose = np.concatenate([tcp[:, :3, 3] * 1000.0, Rotation.from_matrix(tcp[:, :3, :3]).as_rotvec()], axis=1)
    return poses, tcp_pose


def world_view(state: EditorState, data: dict) -> dict:
    """Every object's capsules in its own frame, for drawing, and the planner's capsule budget.

    state: The editor state. data: The scenario as edited on the page.
    Returns: {"objects": per object {"capsules": [[sx, sy, sz, ex, ey, ez, r] meters], "error": str or None,
        "spacing_mm": distance between neighbouring capsule centers (null for one capsule), and for doors
        and handles "hinge": {"point_m", "axis" (unit, positive = its direction), "min_deg", "max_deg"} in
        the same frame}, "avoid_capsules": capsules the avoid objects need, "max_capsules": planner slots}.
    """
    floor_z_m = data.get("floor_z_mm", -DEFAULT_ARM_BASE_HEIGHT_M * 1000.0) / 1000.0
    objects, n_avoid = [], 0
    for spec in data.get("objects", []):
        try:
            # A map object's z is its height above the floor: the map frame is a frame with the floor at 0.
            floor = 0.0 if spec.get("frame") == "map" else floor_z_m
            starts, ends, radii = object_capsules(spec, floor)
            hinge = None
            if spec["type"] in PUSHABLE_TYPES:
                point, axis, min_deg, max_deg = hinge_of(spec, floor)
                hinge = {"point_m": point.round(_DECIMALS), "axis": axis.round(6), "min_deg": min_deg, "max_deg": max_deg}
            rows, error = np.column_stack([starts, ends, radii]).round(_DECIMALS).tolist(), None
            centers = (starts + ends) / 2.0
            spacing = round(float(np.linalg.norm(centers[1] - centers[0])) * 1000.0, 1) if len(radii) > 1 else None
        except (KeyError, ValueError, TypeError, AssertionError) as e:
            rows, error, spacing, hinge = [], f"{type(e).__name__}: {e}", None, None
        if spec.get("role", "avoid") == "avoid" and spec["type"] not in PUSHABLE_TYPES:
            n_avoid += len(rows)
        objects.append({"capsules": rows, "error": error, "spacing_mm": spacing, "hinge": hinge})
    return {"objects": objects, "avoid_capsules": n_avoid, "max_capsules": MAX_WORLD_CAPSULES}


def mesh_clearance(model: CollisionModel, joints_deg: np.ndarray, objects: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Distance from every link's real collision mesh to the nearest avoid object (see mesh_check).

    model: Arm model posing the meshes (the editor's, or a run's).
    joints_deg: (S, 6) configurations, degrees.
    objects: Avoid objects as (name, starts_m (M, 3), ends_m (M, 3), radii_m (M,)).
    Returns: ((S, L) mm, +inf for links without a mesh or no objects; (S, L) index into objects of the
        nearest one, -1 for none).
    """
    joints = np.atleast_2d(joints_deg)
    shape = (len(joints), len(model.link_names))
    if not any(len(o[3]) for o in objects):
        return np.full(shape, np.inf), np.full(shape, -1)
    # One pass over every object's capsules; owner maps the nearest capsule back to its object.
    starts = np.concatenate([np.reshape(o[1], (-1, 3)) for o in objects])
    ends = np.concatenate([np.reshape(o[2], (-1, 3)) for o in objects])
    radii = np.concatenate([np.reshape(o[3], -1) for o in objects])
    owner = np.concatenate([np.full(len(o[3]), i) for i, o in enumerate(objects)])
    nearest = np.zeros(shape, dtype=np.int64)
    distance = link_mesh_distances(model, joints, starts, ends, radii, nearest=nearest) * 1000.0
    return distance, np.where(np.isfinite(distance), owner[nearest], -1)


def clearance_summary(model: CollisionModel, capsule_mm: np.ndarray, mesh_mm: np.ndarray, mesh_object: np.ndarray,
                      names: list[str]) -> list[dict]:  # fmt: skip
    """Per configuration, the closest link by planner capsule and by real mesh, for the page's labels.

    model: The arm model (link names). capsule_mm, mesh_mm: (S, L) distances, +inf for none.
    mesh_object: (S, L) nearest object index per link for the mesh distances, -1 for none.
    names: Object names, indexed by mesh_object.
    Returns: (S,) {"capsule_mm", "capsule_link": closest by capsule; "mesh_mm", "mesh_link", "mesh_object":
        closest by real mesh and its object, "mesh_far": True when mesh_mm is only the lower bound
        mesh_check.EXACT_WITHIN_M (every link farther; mesh_link is then not the closest one for sure);
        "mesh_link_capsule_mm": that link's capsule distance}; numbers rounded to 0.1mm, null when there is no
        distance.
    """
    out, far_mm = [], EXACT_WITHIN_M * 1000.0
    for capsule, mesh, obj in zip(capsule_mm, mesh_mm, mesh_object):
        c, m = int(np.argmin(capsule)), int(np.argmin(mesh))
        far = bool(np.isfinite(mesh[m]) and mesh[m] >= far_mm)
        out.append({"capsule_mm": _mm(capsule[c]), "capsule_link": model.link_names[c],
                    "mesh_mm": _mm(far_mm if far else mesh[m]), "mesh_far": far, "mesh_link": model.link_names[m],
                    "mesh_object": names[obj[m]] if obj[m] >= 0 else None,
                    "mesh_link_capsule_mm": _mm(capsule[m])})  # fmt: skip
    return out


def _mm(value: float) -> float | None:
    """A distance for the page.

    value: mm, +inf for no distance. Returns: Rounded to 0.1mm, None for no distance.
    """
    return round(float(value), 1) if np.isfinite(value) else None


def _mm_rows(values: np.ndarray) -> list[list[float | None]]:
    """(S, L) distances in mm as JSON rows.

    values: (S, L) mm, +inf for no distance. Returns: Rows of _mm values.
    """
    return [[_mm(c) for c in row] for row in values]


def forward_kinematics(state: EditorState, body: dict) -> dict:
    """The arm at given joints, and how close each link is to the scenario's avoid objects.

    state: The editor state.
    body: {"joints": (S, 6) degrees, "scenario": the edited scenario, optional (no clearance without it)}.
    Returns: {"poses": (S, L, 7) see arm_poses, "tcp": (S, 6) TCP poses, "clearance_mm": (S, L) planner
        capsule distance to the nearest avoid object (negative inside, null for static links or no objects),
        "mesh_clearance_mm": (S, L) the same for the real collision meshes, "summary": (S,) see
        clearance_summary, "error"}.
    """
    model = state.require_model()
    joints = np.asarray(body["joints"], dtype=np.float64).reshape(-1, 6)
    poses, tcp = arm_poses(model, joints)
    clearance = mesh = np.full(poses.shape[:2], np.inf)
    objects, error = [], None
    try:
        if body.get("scenario") is not None:
            world = scenario_from_data(body["scenario"], "editor")["world"]
            starts, ends, radii = world.capsules("avoid")
            if len(radii):
                clearance = link_obstacle_distances(model, np.deg2rad(joints), starts, ends, radii).min(axis=2)
            objects = [(o.name, o.starts_m, o.ends_m, o.radii_m) for o in world.objects if o.role == "avoid"]
    except (KeyError, ValueError, TypeError, AssertionError) as e:
        error = f"{type(e).__name__}: {e}"
    clearance = np.where(model.movable_links[None], clearance, np.inf) * 1000.0
    mesh, mesh_object = mesh_clearance(model, joints, objects)
    return {"poses": poses, "tcp": tcp.round(_DECIMALS), "clearance_mm": _mm_rows(clearance), "mesh_clearance_mm": _mm_rows(mesh),
            "summary": clearance_summary(model, clearance, mesh, mesh_object, [o[0] for o in objects]),
            "error": error}  # fmt: skip


def solve_ik(state: EditorState, body: dict) -> dict:
    """The planner's IK for one pose among the scenario's avoid objects, like the `pose` command.

    state: The editor state.
    body: {"scenario": the edited scenario, "pose": [x, y, z mm, rx, ry, rz rad], "joints": (6,) seed degrees}.
    Returns: {"joints": (6,) solution degrees, "error_mm": TCP distance to the target, "base_collision": bool}.
    """
    planner = state.require_planner()
    world = scenario_from_data(body["scenario"], "editor")["world"]
    pose = np.asarray(body["pose"], dtype=np.float64)
    with state.lock:
        planner.set_obstacles(*world.capsules("avoid"))
        solution, base_collision = planner.solve(np.asarray(body["joints"], dtype=np.float64), pose)
    reached = planner.tcp_poses(solution[None])[0]
    error_mm = float(np.linalg.norm(reached[:3] - pose[:3]))
    return {"joints": solution.round(2), "error_mm": round(error_mm, 1), "base_collision": base_collision}


def plan_locally(state: EditorState, scenario_data: dict, flags: dict) -> dict:
    """Plans a scenario with this machine's planner, like plan.py.

    state: The editor state.
    scenario_data: The scenario JSON.
    flags: {"keep_going", "continue_on_block", "compare_no_obstacles"}, the plan.py flags.
    Returns: The plan results (plan.run_scenario's dict plus "name", and "no_obstacles" when compared).
    Raises: RuntimeError while the planner is not ready.
    """
    planner = state.require_planner()
    from Simulation.world_sim.plan import run_scenario, run_without_obstacles  # noqa: PLC0415 (imports JAX)

    scenario = scenario_from_data(scenario_data, "editor")
    args = (planner, scenario["world"], scenario["start_joints_deg"], scenario["commands"])
    options = {"continue_on_block": bool(flags.get("continue_on_block")), "stop_on_error": not flags.get("keep_going")}
    with state.lock:
        results = run_scenario(*args, **options)
        if flags.get("compare_no_obstacles"):
            results["no_obstacles"] = run_without_obstacles(*args, **options)
    results["name"] = scenario["name"]
    return results


def start_run(state: EditorState, body: dict) -> dict:
    """Starts a run of the edited scenario on the chosen machines (see runs.py).

    state: The editor state.
    body: {"scenario", "file", "note", "keep_going", "continue_on_block", "compare_no_obstacles" (also plan
        without the avoid objects), "targets": "local" and/or ssh
        destinations, "robot_checkout": billie-onboard checkout on the robots ("" = local code)}.
    Returns: The run's meta; poll it with GET /api/run?id=.
    """
    if not body.get("targets"):
        raise ValueError("Choose at least one machine to run on")
    flags = {k: bool(body.get(k)) for k in ("keep_going", "continue_on_block", "compare_no_obstacles")}
    return runs.start_run(body["scenario"], body.get("file", ""), body.get("note", ""), flags, body["targets"],
                          body.get("robot_checkout", robot.DEFAULT_ROBOT_CHECKOUT),
                          lambda data, f: plan_locally(state, data, f))  # fmt: skip


def run_status(state: EditorState, query: dict) -> dict:
    """One run's meta with its machines' progress.

    state: Unused. query: {"id": run id}. Returns: The meta (see runs.py).
    """
    return runs.get_run(query["id"])


def run_history(state: EditorState, _query: dict) -> dict:
    """Every archived run, newest first, without the machines' logs.

    state, _query: Unused. Returns: {"runs": metas}.
    """
    metas = runs.list_runs()
    return {"runs": [{**m, "targets": {n: {k: v for k, v in t.items() if k != "log"} for n, t in m["targets"].items()}}
                     for m in metas]}  # fmt: skip


def delete_run(state: EditorState, body: dict) -> dict:
    """Deletes an archived run.

    state: Unused. body: {"id": run id}. Returns: {"deleted": the id}.
    """
    runs.delete_run(body["id"])
    return {"deleted": body["id"]}


def clean_history(state: EditorState, _body: dict) -> dict:
    """Deletes every finished run from the archive (History's Clean).

    state, _body: Unused. Returns: {"deleted": the deleted run ids}.
    """
    return {"deleted": runs.delete_finished_runs()}


def run_playback(state: EditorState, query: dict) -> dict:
    """The arm path of one machine of a run, for the page's player.

    state: Unused. query: {"run": run id, "target": machine name}.
    Returns: playback(run) of that machine's results.
    """
    return playback(load_run(runs.results_path(query["run"], query["target"])))


def random_content(state: EditorState, body: dict) -> dict:
    """Random objects or motions for the edited scenario (see randomize.py).

    state: The editor state.
    body: {"scenario", "what": "object" | "joints" | "pose", "count", "type": object type or null,
        "seed": int or null for a new one}.
    Returns: {"objects": [...]} or {"commands": [...]}, and "seed" (draws the same again).
    """
    seed = body.get("seed")
    seed = int(seed) if seed not in (None, "") else int(np.random.default_rng().integers(1_000_000))
    rng, count, model = np.random.default_rng(seed), int(body.get("count", 1)), state.require_model()
    if body["what"] == "object":
        return {"objects": random_objects(body["scenario"], model, rng, count, body.get("type")), "seed": seed}
    commands = random_motions(body["scenario"], model, joint_limits_deg(), rng, count, body["what"])
    return {"commands": commands, "seed": seed}


def playback(run: Run) -> dict:
    """The run's arm path sampled for the page's player, with each link's clearance state.

    run: The planned run.
    Returns: {"joints_deg": (S, 6) the arm's joints, "poses": (S, L, 7) capsule poses (see arm_poses),
        "capsules": per link {"radius", "height"} of the run's model (m), "link_state": (S, L) 0 clear /
        1 inside the planner margin / 2 inside an avoid object, the worse of the planner capsule and the
        real mesh (mesh_check), "summary": (S,) see clearance_summary, "tcp_mm": (S, 3), "owner": (S,)
        target index per sample (-1 = start), "targets": per target {"command", "kind", "blocked"},
        "commands": per command {"cmd", "ok", "error", "seconds"}, "events": operator messages,
        "warnings": run-level notes like the report's (planner gripper box, real-mesh contacts it reads clear,
        blocked pushes), "pushes": see _playback_pushes}.
    """
    samples, owner = dense_path(run)
    distance = link_distances(run, samples, "avoid") * 1000.0
    avoid = [(o["name"], o["starts_m"], o["ends_m"], o["radii_m"]) for o in run.objects if o["role"] == "avoid"]
    mesh, mesh_object = mesh_clearance(run.model, samples, avoid)
    warnings = []
    if planner_gripper_box(run):
        warnings.append(f"The planner on {run.machine.get('host', '?')} modelled the gripper as its bounding box (its "
                        "gripper mesh was missing there): its gripper clearances miss the camera bracket.")  # fmt: skip
    hidden = np.flatnonzero((mesh.min(axis=1) < 0) & (distance.min(axis=1) >= 0) & (owner >= 0))
    if len(hidden):
        commands = ", ".join(str(c + 1) for c in sorted({int(run.command[owner[k]]) for k in hidden}))
        warnings.append(f"The real arm mesh enters an avoid object where the planner model reads clear: {len(hidden)} "
                        f"samples of cmd {commands}, deepest {mesh[hidden].min():.0f} mm.")  # fmt: skip
    worst = np.minimum(distance, mesh)
    link_state = np.where(worst < 0, 2, np.where(worst < WORLD_COL_MARGIN_M * 1000.0, 1, 0))
    poses, tcp = arm_poses(run.model, samples)
    pushes, push_warnings = _playback_pushes(run, owner)
    return {
        "joints_deg": samples.round(2),
        "poses": poses,
        "capsules": [{"radius": float(r), "height": float(h)} for r, h in zip(run.model.capsule_radius, run.model.capsule_height)],
        "link_state": link_state,
        "summary": clearance_summary(run.model, distance, mesh, mesh_object, [o[0] for o in avoid]),
        "tcp_mm": tcp[:, :3].round(1),
        "owner": owner,
        "targets": [{"command": int(c), "kind": k, "blocked": b} for c, k, b in zip(run.command, run.kind, run.blocked)],
        "commands": [{"cmd": c["spec"]["cmd"], "ok": c["ok"], "error": c["error"], "seconds": round(c["seconds"], 2)}
                     for c in run.commands],  # fmt: skip
        "events": run.events,
        "warnings": warnings + push_warnings,
        "pushes": pushes,
    }


def _playback_pushes(run: Run, owner: np.ndarray) -> tuple[dict | None, list[str]]:
    """How the run's doors and handles turned along the playback path (pushables.py).

    run: The planned run. owner: (S,) target index per playback sample.
    Returns: (None without doors or handles, else {"objects": index of each in the scenario's objects,
        "angles_deg": (S, P), "blocked_mm": (S, P), "pusher": (S, P) link name or None, "events": see
        pushables.push_events}; warnings about blocked pushes and pushes by links other than the EEF).
    """
    pushes = simulate_pushes(run.model, run.objects, run.start_deg, run.joints_deg)
    if pushes is None:
        return None, []
    events = push_events(pushes, owner, run.command)
    warnings = []
    for e in events:
        where = f"{e['object']} (cmd {e['command'] + 1}, targets {e['first_target']}-{e['last_target']})"
        if e["blocked_mm"] > 0:
            warnings.append(f"{where}: pushed past its limit, the arm goes {e['blocked_mm']:.0f} mm into it (a collision).")
        others = [link for link in e["links"] if link not in EEF_LINKS]
        if others:
            warnings.append(f"{where}: pushed by {', '.join(others)}, not only by the end effector.")
    names = [o["name"] for o in run.objects]
    pushes["objects"] = [names.index(n) for n in pushes.pop("names")]
    pushes["events"] = events
    return pushes, warnings


def _scenario_path(name: str) -> str:
    """The scenarios-folder path of a file name from the page.

    name: File name like "my_test.json" (".json" is added if missing).
    Returns: Absolute path inside SCENARIOS_DIR. Raises: ValueError for names with folders or odd characters.
    """
    name = name if name.endswith(".json") else f"{name}.json"
    if not _FILE_NAME.match(name):
        raise ValueError(f"Scenario file name {name!r}: use letters, digits, '_', '-' and '.' only")
    return os.path.join(SCENARIOS_DIR, name)


def list_scenarios(state: EditorState, _query: dict) -> dict:
    """The scenario files the page can open.

    state, _query: Unused. Returns: {"files": sorted file names in SCENARIOS_DIR}.
    """
    return {"files": sorted(os.path.basename(p) for p in glob.glob(os.path.join(SCENARIOS_DIR, "*.json")))}


def read_scenario(state: EditorState, query: dict) -> dict:
    """One scenario file, as written.

    state: Unused. query: {"file": file name in SCENARIOS_DIR}.
    Returns: {"file": the file name, "scenario": its parsed JSON}.
    """
    with open(_scenario_path(query["file"])) as f:
        return {"file": os.path.basename(_scenario_path(query["file"])), "scenario": json.load(f)}


def write_scenario(state: EditorState, body: dict) -> dict:
    """Saves the edited scenario; one object or command per line, like the hand-written scenarios.

    state: Unused. body: {"file": file name in SCENARIOS_DIR, "scenario": the scenario}.
    Returns: {"file": the saved file name, "path": its absolute path}.
    """
    path = _scenario_path(body["file"])
    data = body["scenario"]
    lines = []
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            items = ",\n".join(f"    {json.dumps(v)}" for v in value)
            lines.append(f'  "{key}": [\n{items}\n  ]')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value)}")
    os.makedirs(SCENARIOS_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")
    return {"file": os.path.basename(path), "path": path}


def planner_status(state: EditorState, _query: dict) -> dict:
    """The planner state for the page's status badge.

    state: The editor state. _query: Unused. Returns: {"planner": "building" / "ready" / "off" / "error: ..."}.
    """
    return {"planner": state.status}


# (HTTP method, path) -> handler(state, query or JSON body) -> JSON-serializable reply.
_ROUTES: dict[tuple[str, str], Callable[[EditorState, dict], dict]] = {
    ("GET", "/api/model"): model_info,
    ("GET", "/api/status"): planner_status,
    ("GET", "/api/scenarios"): list_scenarios,
    ("GET", "/api/scenario"): read_scenario,
    ("POST", "/api/scenario"): write_scenario,
    ("POST", "/api/world"): world_view,
    ("POST", "/api/fk"): forward_kinematics,
    ("POST", "/api/ik"): solve_ik,
    ("POST", "/api/run"): start_run,
    ("GET", "/api/run"): run_status,
    ("GET", "/api/runs"): run_history,
    ("POST", "/api/runs/delete"): delete_run,
    ("POST", "/api/runs/clean"): clean_history,
    ("GET", "/api/playback"): run_playback,
    ("POST", "/api/random"): random_content,
}


def _json_default(value):
    """Lets json.dumps write NumPy arrays and scalars.

    value: An object json cannot serialize itself. Returns: A plain list or number.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class _Handler(BaseHTTPRequestHandler):
    """Serves the page, the reports in OUTPUT_DIR and the _ROUTES API."""

    state: EditorState  # set by serve()

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        """Handles a GET request."""
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        """Handles a POST request."""
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        """Answers one request: the page, a report file, or an API route as JSON (errors as {"error"}).

        method: "GET" or "POST".
        """
        url = urlparse(self.path)
        try:
            if method == "GET" and url.path == "/urdf":
                return self._send(HTTPStatus.OK, "application/xml", robot_urdf(self.state))
            if method == "GET" and url.path.startswith("/mesh/"):
                return self._send(HTTPStatus.OK, "application/octet-stream", mesh_file(self.state, url.path))
            if method == "GET" and url.path == "/":
                return self._send(HTTPStatus.OK, "text/html; charset=utf-8", open(EDITOR_HTML, "rb").read())
            if method == "GET" and url.path.startswith("/runs/"):
                return self._send_run_file(url.path)
            route = _ROUTES.get((method, url.path))
            if route is None:
                return self._send_json({"error": f"no route {method} {url.path}"}, HTTPStatus.NOT_FOUND)
            if method == "POST":
                arg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            else:
                arg = {k: v[0] for k, v in parse_qs(url.query).items()}
            self._send_json(route(self.state, arg))
        except Exception as e:  # any failure goes back to the page as a message
            expected = isinstance(e, (RuntimeError, ValueError, KeyError, FileNotFoundError))
            logging.warning(f"{method} {url.path} failed: {e}" if expected else traceback.format_exc())
            self._send_json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_run_file(self, url_path: str) -> None:
        """Sends an archived run file: /runs/index.html or /runs/<run id>/<file>.html|.json.

        url_path: The request path.
        """
        parts = url_path.strip("/").split("/")[1:]
        safe = len(parts) in (1, 2) and all(re.fullmatch(r"[A-Za-z0-9_.+-]+", p) and p not in (".", "..") for p in parts)
        path = os.path.join(runs.RUNS_DIR, *parts) if safe else ""
        if not path.endswith((".html", ".json")) or not os.path.isfile(path):
            return self._send_json({"error": "no such file"}, HTTPStatus.NOT_FOUND)
        kind = "text/html; charset=utf-8" if path.endswith(".html") else "application/json"
        with open(path, "rb") as f:
            self._send(HTTPStatus.OK, kind, f.read())

    def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Sends a JSON reply.

        data: The reply (NumPy allowed). status: HTTP status.
        """
        self._send(status, "application/json", json.dumps(data, default=_json_default).encode())

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        """Sends a complete reply.

        status: HTTP status. content_type: Its Content-Type. body: The bytes to send.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (http.server signature)
        """Keeps the per-request access log out of the console (debug level only).

        format, args: http.server's log message.
        """
        logging.debug(format % args)


def main() -> None:
    """Command line: starts the planner build and the page server, and opens the page."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", help="scenario file name in scenarios/ to open first")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port of the page")
    parser.add_argument("--no-planner", action="store_true", help="edit and preview only (no JAX; needs a cached arm model)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    parser.add_argument("--robot", action="append", help=f"robot ssh destination offered as a run target "
                        f"(repeat for several; default {robot.DEFAULT_HOST}); more can be added on the page")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _Handler.state = EditorState(use_planner=not args.no_planner, robots=args.robot or [robot.DEFAULT_HOST])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    url = f"http://127.0.0.1:{args.port}/" + (f"?file={args.scenario}" if args.scenario else "")
    print(f"World editor: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
