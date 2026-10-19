"""The room field page with an Update button and a history (python -m Simulation.world_sim.esdf_view --host ...).

A small local server (127.0.0.1 only). Update fetches the robot's live occupancy and robot state (read only),
builds the views (esdf_view.build_entry) and keeps them in output/world_sim/occupancy/<robot>/history/<time>/:
the snapshot, robot.json, the 3D figure (figure.json), the slices page and info.json. The page lists every update,
newest first, with its pose, joints and base as copyable vectors; clicking one shows its 3D map.
"""

import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from plotly.offline import get_plotlyjs_version

from Simulation.world_sim import esdf_view, occupancy_view, robot

# The page: the Update button and status, the history on the left, the 3D map on the right.
_PAGE = """<!doctype html><html><head><title>{robot} room field</title>{head}
<style>
#bar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 6px; }}
#main {{ display: flex; gap: 8px; }} #list {{ width: 430px; height: 90vh; overflow-y: auto; }}
#plot {{ flex: 1; height: 90vh; }}
.entry {{ border: 1px solid #ddd; border-radius: 6px; padding: 6px; margin-bottom: 6px; cursor: pointer; }}
.entry.shown {{ border-color: #1976d2; background: #f3f8ff; }} .when {{ font-weight: 600; }}
</style>
<script src="https://cdn.plot.ly/plotly-{plotly}.min.js"></script></head><body>
<div id="bar"><b>{robot}</b><button id="update">Update</button><span id="status"></span></div>
<div id="main"><div id="list"></div><div id="plot"></div></div>
<script>
let shown = null;
async function show(id) {{
  document.getElementById("status").textContent = "loading " + id + "...";
  const figure = await (await fetch("/figure/" + id)).json();
  await Plotly.react("plot", figure.data, figure.layout);
  shown = id; document.querySelectorAll(".entry").forEach(e => e.classList.toggle("shown", e.dataset.id === id));
  document.getElementById("status").textContent = "";
}}
async function load() {{
  const entries = await (await fetch("/history")).json();
  const list = document.getElementById("list"); list.innerHTML = "";
  for (const entry of entries) {{
    const div = document.createElement("div"); div.className = "entry"; div.dataset.id = entry.id;
    div.innerHTML = `<div class="when">${{entry.time}} <small>(${{entry.frames ?? "?"}} frames)</small>
      <a href="/slices/${{entry.id}}" target="_blank" onclick="event.stopPropagation()">slices</a></div>` + entry.html;
    div.onclick = () => show(entry.id);
    list.appendChild(div);
  }}
  return entries;
}}
document.getElementById("update").onclick = async () => {{
  const button = document.getElementById("update"); button.disabled = true;
  document.getElementById("status").textContent = "updating from the robot...";
  const reply = await fetch("/update", {{method: "POST"}});
  const entry = await reply.json(); button.disabled = false;
  if (!reply.ok) {{ document.getElementById("status").textContent = "update failed: " + entry.error; return; }}
  await load(); await show(entry.id);
}};
load().then(entries => {{ if (entries.length) show(entries[0].id); }});
</script></body></html>"""


def history_dir(host: str) -> str:
    """Where a robot's updates are kept. host: ssh destination. Returns: The folder."""
    return os.path.join(robot._REPO, "output", "world_sim", "occupancy", robot.robot_name(host), "history")


def update(host: str, args: argparse.Namespace) -> dict:
    """Fetches the robot's live map and state, builds the views and keeps them as a new history entry.

    host: ssh destination. args: esdf_view's command line.
    Returns: The entry's info (id, time, frames, vectors, notes).
    """
    entry_id = time.strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(history_dir(host), entry_id)
    path, _ = occupancy_view.fetch(host, folder)
    figure, info = esdf_view.build_entry(path, folder, args)
    with open(os.path.join(folder, "figure.json"), "w") as f:
        f.write(figure.to_json())
    info = {"id": entry_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    with open(os.path.join(folder, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def history(host: str) -> list[dict]:
    """The kept updates, newest first, each with its info block's HTML. host: ssh destination. Returns: Infos."""
    root = history_dir(host)
    entries = []
    for entry_id in sorted(os.listdir(root) if os.path.isdir(root) else [], reverse=True):
        path = os.path.join(root, entry_id, "info.json")
        if os.path.exists(path):
            with open(path) as f:
                info = json.load(f)
            entries.append({**info, "html": esdf_view.info_html(info)})
    return entries


def _handler(host: str, args: argparse.Namespace) -> type[BaseHTTPRequestHandler]:
    """The request handler for one robot. host: ssh destination. args: esdf_view's command line."""
    busy = threading.Lock()  # one update at a time

    class Handler(BaseHTTPRequestHandler):
        """GET /, /history, /figure/<id>, /slices/<id>; POST /update."""

        def _send(self, status: int, body: bytes, kind: str) -> None:
            """Sends one answer. status: HTTP status. body: The content. kind: Its content type."""
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _entry_file(self, prefix: str, name: str) -> str | None:
            """A kept entry's file from a /<prefix>/<id> path, or None if there is none."""
            entry_id = os.path.basename(unquote(self.path[len(prefix) :]))
            path = os.path.join(history_dir(host), entry_id, name)
            return path if entry_id and os.path.exists(path) else None

        def do_GET(self) -> None:
            """The page, the history, an entry's figure or its slices page."""
            if self.path == "/":
                page = _PAGE.format(robot=robot.robot_name(host), head=esdf_view.PAGE_HEAD, plotly=get_plotlyjs_version())
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif self.path == "/history":
                self._send(200, json.dumps(history(host)).encode(), "application/json")
            elif self.path.startswith("/figure/") and (path := self._entry_file("/figure/", "figure.json")):
                with open(path, "rb") as f:
                    self._send(200, f.read(), "application/json")
            elif self.path.startswith("/slices/") and (path := self._entry_file("/slices/", "esdf_slices.html")):
                with open(path, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            """/update: a new entry from the robot now."""
            if self.path != "/update":
                self._send(404, b"not found", "text/plain")
                return
            with busy:
                try:
                    info = update(host, args)
                except Exception as e:  # noqa: BLE001 - the page shows why the update failed
                    self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")
                    return
            self._send(200, json.dumps(info).encode(), "application/json")

        def log_message(self, format: str, *values) -> None:
            """Quiet: only the updates print their own lines."""

    return Handler


def serve(args: argparse.Namespace) -> None:
    """Serves the page for args.host on 127.0.0.1:args.port and opens it; the first update runs if there is no history.

    args: esdf_view's command line. Runs until Ctrl-C.
    """
    if not history(args.host):
        update(args.host, args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _handler(args.host, args))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving {url} (Update fetches {args.host} again; Ctrl-C stops)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
