#!/bin/bash
# Runs the world_sim plan step inside a robot's `billie` container with the live pyroki planner's
# environment (same arm, URDF, flags and GPU settings), using the code synced by robot.py.
# Usage (inside the container): robot_run_plan.sh <synced dir> <plan.py arguments...>
set -euo pipefail
DIR="$1"
shift

PID=$(pgrep -f "bin/pyroki-planner" | head -1 || true)
if [ -z "$PID" ]; then
    echo "pyroki-planner is not running in this container; its environment is needed" >&2
    exit 1
fi
while IFS= read -r -d '' kv; do
    export "$kv"
done <"/proc/$PID/environ"

# A robot with no physical arm (no URDF, e.g. billie-29) gets the copy robot.py synced from a developer env;
# all xArm6 URDFs share one structure (only calibration differs), so planner timing is unaffected.
if [ ! -f "$BILLIE_ENVDIR/urdf/$XARM_SN.urdf" ] && [ -f "$DIR/env/urdf/$XARM_SN.urdf" ]; then
    echo "No $XARM_SN.urdf in $BILLIE_ENVDIR/urdf; using the synced copy in $DIR/env/urdf" >&2
    export BILLIE_ENVDIR="$DIR/env"
    # robot.py syncs the gripper mesh with this copy; without it the planner would use its gripper box.
    if [ ! -f "$BILLIE_ENVDIR/urdf/meshes/ee-rome-v0-lod.stl" ]; then
        echo "No gripper mesh $BILLIE_ENVDIR/urdf/meshes/ee-rome-v0-lod.stl; re-run robot.py to sync it" >&2
        exit 1
    fi
elif [ ! -f "$BILLIE_ENVDIR/urdf/meshes/ee-rome-v0-lod.stl" ]; then
    # The robot's own env, as its live planner sees it: plan with the same box, but say so up front.
    echo "WARNING: no gripper mesh in $BILLIE_ENVDIR/urdf/meshes; the planner uses its gripper box" >&2
fi

# The synced code wins over the image's /app code (the image may lack the world-collision POC).
export PYTHONPATH="$DIR/src:$DIR/billie/nodes/pyroki-planner:$DIR/billie/billie-utils:$DIR/pyroki/src"
# Own JAX cache, so the robot's planner cache is never touched.
export PYROKI_JAX_CACHE_DIR="$DIR/jax-cache"
export WORLD_SIM_RECORDINGS="$DIR/recordings"
cd "$DIR"
# WORLD_SIM_MODULE (docker exec -e) picks another entry point, e.g. Simulation.world_sim.sweep.
exec /app/nodes/pyroki-planner/.venv/bin/python -m "${WORLD_SIM_MODULE:-Simulation.world_sim.plan}" "$@"
