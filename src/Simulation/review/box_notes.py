"""A short note per chart box, shown when the box is clicked in a review's HTML preview (GitHub cannot run it).

Most notes come from the code and the chart itself: a box's functions give their docstring's first line; a decision
lists its cases from its arrows; a trigger gives its chart's "When" line; an outcome says what leads to it; a box that
points to another chart says so. A box with nothing to read from takes its note from the review's hand-written ones.
"""

import ast
import glob
import html
import json
import os
import re
from dataclasses import dataclass, field

# A node definition in a mermaid line: its id, opening bracket, label and class (pill, hexagon, box).
_DEF = re.compile(r'\b(\w+)(\(\[|\{\{|\[)"((?:[^"\\]|\\.)*)"(?:\]\)|\}\}|\])(?::::(\w+))?')
# An edge with a label written before the arrow head: A -- "x" --> B, A == "x" ==> B, A -. "x" .-> B.
_EDGE_MID = re.compile(r'^\s*(\w+)\s*(?:--|==|-\.)\s*"((?:[^"\\]|\\.)*)"\s*(?:-->|==>|\.->)\s*(\w+)')
# An edge without a label, or with a |"label"| after the arrow: A --> B, A ==> B, A -.-> B, A -->|"x"| B.
_EDGE_PIPE = re.compile(r'^\s*(\w+)\s*(?:-->|==>|-\.->)\s*(?:\|"((?:[^"\\]|\\.)*)"\|\s*)?(\w+)')
# A bold function or method name in a label: <b>name()</b>, <b>Class.method()</b>, <b>module.name()</b>.
_BOLD_CALL = re.compile(r"<b>([A-Za-z_][\w.]*)\(\)")
# Folders whose Python files never hold the reviewed code.
_SKIP = ("/vendor/", "/.venv/", "/tests/", "/test/", "/node_modules/")
# [chars] A note line taken from a label is cut here.
_SHORT = 60
# [chars] A docstring sentence longer than this is cut at its last clause break before it.
_SENTENCE = 140
# Functions described per box at most (a box naming more lists the first ones).
_MAX_CALLS = 4


@dataclass
class Box:
    """One chart box: its id, label (HTML), class, and the arrows into and out of it."""

    node: str
    label: str
    cls: str
    into: list[tuple[str, str]] = field(default_factory=list)  # (source box id, arrow label)
    out: list[tuple[str, str]] = field(default_factory=list)  # (target box id, arrow label)


def parse_chart(body: str) -> dict[str, Box]:
    """The boxes of a mermaid flowchart and their arrows.

    body: The chart's mermaid text. Returns: Box id -> box (subgraph frames left out).
    """
    boxes: dict[str, Box] = {}
    for line in body.splitlines():
        if line.strip().startswith(("subgraph", "style", "classDef", "click", "%%")):
            continue
        for node, _, label, cls in _DEF.findall(line):
            boxes[node] = Box(node, label, cls or "")
        plain = _DEF.sub(lambda m: m.group(1), line)
        edge = _EDGE_MID.match(plain) or _EDGE_PIPE.match(plain)
        if edge and "~~~" not in plain:
            src, label, dst = edge.group(1), edge.group(2) or "", edge.group(3)
            for node in (src, dst):
                boxes.setdefault(node, Box(node, node, ""))
            boxes[src].out.append((dst, label))
            boxes[dst].into.append((src, label))
    return boxes


def plain_text(html: str) -> str:
    """A label without its tags, line breaks as spaces. html: The label. Returns: The text."""
    text = re.sub(r"<br\s*/?>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text).replace("&lt;", "<").replace("&gt;", ">")).strip()


def short(html: str) -> str:
    """A label's first line up to its first " · ", plain and cut to _SHORT characters. html: The label.
    Returns: The text."""
    first = plain_text(re.split(r"<br\s*/?>", html)[0]).split(" · ")[0].rstrip(" :")
    return first if len(first) <= _SHORT else first[: _SHORT - 1].rstrip() + "…"


def docstrings(repo: str, subdir: str = "billie") -> dict[str, list[tuple[str, str]]]:
    """Every function, method and class in a checkout with its docstring's first line.

    repo: The checkout. subdir: The folder to read.
    Returns: Name (plain, "Class.method" and "module.name") -> [(file, first docstring line)], files relative to repo.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for path in glob.glob(os.path.join(repo, subdir, "**", "*.py"), recursive=True):
        if any(part in path for part in _SKIP):
            continue
        try:
            tree = ast.parse(open(path).read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel, stem = os.path.relpath(path, repo), os.path.splitext(os.path.basename(path))[0]

        def visit(node: ast.AST, prefix: str) -> None:
            """Adds the definitions under node. node: An AST node. prefix: "Class." inside a class."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = first_sentence(ast.get_docstring(child) or "")
                    if doc:
                        for name in {child.name, prefix + child.name, f"{stem}.{child.name}"}:
                            out.setdefault(name, []).append((rel, doc))
                    if isinstance(child, ast.ClassDef):
                        visit(child, child.name + ".")

        visit(tree, "")
    return out


def first_sentence(doc: str) -> str:
    """A docstring's first sentence, as HTML: its first paragraph up to the first ". ", `names` as code.

    doc: The docstring. Returns: The sentence ("" for none).
    """
    paragraph = " ".join(doc.strip().split("\n\n")[0].split())
    end = paragraph.find(". ")
    sentence = paragraph if end < 0 else paragraph[: end + 1]
    if len(sentence) > _SENTENCE:
        cut = max(sentence.rfind(mark, 0, _SENTENCE) for mark in (", ", ": ", " (", "; "))
        sentence = sentence[: cut if cut > 40 else _SENTENCE].rstrip(" ,:;(") + "…"
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(sentence, quote=False))


def _doc_for(name: str, label: str, docs: dict[str, list[tuple[str, str]]]) -> str | None:
    """The docstring line for a name drawn in a box: the definition in a file the box names, else the only one.

    name: The drawn name. label: The box's label (it may name the file). docs: docstrings().
    Returns: The line, or None when the name is unknown or ambiguous.
    """
    found = docs.get(name, [])
    if len(found) == 1:
        return found[0][1]
    stems = set(re.findall(r"(\w+)\.py", plain_text(label)))
    named = [doc for path, doc in found if os.path.splitext(os.path.basename(path))[0] in stems]
    return named[0] if named else None


def box_note(box: Box, boxes: dict[str, Box], when: str, docs: dict, own: str | None = None,
             link: tuple[str, str] | None = None) -> str:  # fmt: skip
    """The note for one box, as HTML.

    box: The box. boxes: Its chart's boxes. when: The chart's "When" line (for triggers).
    docs: docstrings(). own: A hand-written note, put first, or None.
    link: (href, text) of the chart or area the box points to, or None.
    Returns: The note ("" when there is nothing to say).
    """
    lines: list[str] = []
    calls = list(dict.fromkeys(_BOLD_CALL.findall(box.label)))[:_MAX_CALLS]
    for name in calls:
        doc = _doc_for(name, box.label, docs)
        if doc:
            lines.append(f"<code>{name}()</code>: {doc}")
    if box.cls == "trig":
        lines.insert(0, f"Starts the flow: {plain_text(when)}")
    elif box.cls.endswith("Dec"):
        cases = [f"<i>{plain_text(label) or 'next'}</i> → {short(boxes[dst].label)}" for dst, label in box.out]
        if cases:
            lines.insert(0, "; ".join(cases) + ".")
    elif box.cls in ("ok", "stop"):
        reached = [f"{short(boxes[src].label)}" + (f" ({plain_text(lbl)})" if lbl else "") for src, lbl in box.into]
        kind = "Goes ahead" if box.cls == "ok" else "Refused, skipped or failed"
        lines.insert(0, f"{kind}." + (" From: " + "; ".join(reached) + "." if reached else ""))
    elif box.cls == "ref":
        lines.insert(0, "Drawn in its own chart (each function has one home); click the link to open it.")
    if own:
        lines.insert(0, own)
    if link:
        lines.append(f'<a href="{link[0]}">{link[1]}</a>')
    return "<br>".join(lines)


def notes_script(notes: dict[str, dict[str, str]]) -> str:
    """The preview's script and style for the notes: a click on a box opens its note beside it.

    notes: Chart key ("4", or "mapper" for a chart without a number) -> box id -> note HTML.
    Returns: The <style> and <script> to append to the page.
    """
    return f"""<style>
  .box-note {{ position: absolute; z-index: 20; max-width: 380px; padding: 8px 28px 8px 10px; font-size: 13px;
    line-height: 1.45; background: var(--bg); color: var(--fg); border: 1px solid var(--rule); border-radius: 6px;
    box-shadow: 0 4px 14px rgba(0,0,0,.18); }}
  .box-note code {{ font-size: 90%; }}
  .box-note button {{ position: absolute; top: 2px; right: 4px; border: 0; background: none; color: var(--muted);
    font-size: 16px; cursor: pointer; }}
  .flow g.node {{ cursor: pointer; }}
</style>
<script>
  // A click on a chart box opens a short note about it (the preview only; GitHub runs no scripts).
  (() => {{
    const notes = {json.dumps(notes, ensure_ascii=False)};
    let open = null;
    const close = () => {{ if (open) {{ open.remove(); open = null; }} }};
    const chartKey = (flow) => {{
      let n = flow.closest("details") || flow;
      for (n = n.previousElementSibling; n; n = n.previousElementSibling) {{
        if (n.tagName === "H2") {{
          const m = n.textContent.match(/^Chart (\\d+) /);
          return m ? m[1] : "mapper";
        }}
      }}
      return null;
    }};
    document.addEventListener("click", (e) => {{
      if (e.target.closest && e.target.closest(".box-note")) {{
        if (e.target.closest("button")) close();
        return;
      }}
      const node = e.target.closest ? e.target.closest(".flow g.node") : null;
      if (!node) {{ close(); return; }}
      e.preventDefault();
      e.stopPropagation();
      const flow = node.closest(".flow");
      const id = (node.id.match(/flowchart-(.+)-\\d+$/) || [])[1];
      const text = (notes[chartKey(flow)] || {{}})[id];
      close();
      if (!text) return;
      const note = document.createElement("div");
      note.className = "box-note";
      note.innerHTML = `<button aria-label="Close">×</button>${{text}}`;
      document.body.appendChild(note);
      const r = node.getBoundingClientRect();
      const left = Math.min(window.scrollX + r.right + 8, window.scrollX + document.documentElement.clientWidth - 396);
      note.style.left = `${{Math.max(window.scrollX + 8, left)}}px`;
      note.style.top = `${{window.scrollY + r.top}}px`;
      open = note;
    }}, true);
    document.addEventListener("keydown", (e) => {{ if (e.key === "Escape") close(); }});
  }})();
</script>
"""
