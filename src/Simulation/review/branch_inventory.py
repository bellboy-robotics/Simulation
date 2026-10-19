"""Every function, method and class a branch adds or changes, with what it does and who calls it.

    python -m Simulation.review.branch_inventory --repo billie-onboard --out output/review/inventory.json

Compares each changed Python file at the merge-base with main against the working tree (uncommitted and
untracked files included), definition by definition. "What it does" is the first docstring line; "called
from" lists the functions that reference the name, filtered so a same-named function in another module
does not count.
"""

import argparse
import ast
import glob
import json
import os
import subprocess

# Folders under the repo whose Python files are never part of a review (vendored code, virtualenvs).
_SKIP_DIRS = ("/vendor/", "/.venv/")


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    """Runs git in repo. args: The git arguments. Returns: The finished process (text output)."""
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def definitions(source: str) -> dict[str, tuple[str, str]]:
    """Every def and class in a source file, methods named Class.method.

    source: Python source text.
    Returns: {qualified name: (kind, text compared between versions)}; kind is "function", "method",
        "class" or "data class" (a dataclass, SerializableMessage or pydantic model).
    """
    tree = ast.parse(source)
    out: dict[str, tuple[str, str]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        """Adds the definitions directly under node. node: An AST node. prefix: "Class." for methods."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + child.name] = ("method" if prefix else "function", ast.get_source_segment(source, child) or "")
            elif isinstance(child, ast.ClassDef):
                bases = [ast.unparse(b) for b in child.bases]
                is_data = any("SerializableMessage" in b or "BaseModel" in b for b in bases) or any(
                    "dataclass" in ast.unparse(d) for d in child.decorator_list
                )
                # A class counts as changed when anything but its methods changed (those are compared one by one).
                body = "\n".join(ast.get_source_segment(source, s) or "" for s in child.body
                                 if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)))
                out[prefix + child.name] = ("data class" if is_data else "class", body + str(bases))
                visit(child, prefix + child.name + ".")

    visit(tree, "")
    return out


def changed_definitions(repo: str, base: str) -> list[dict]:
    """The definitions added, changed or removed since base, in the repo's billie/ folder.

    repo: Path of the git checkout. base: Commit to compare with (the merge-base with main).
    Returns: [{"file", "name", "kind", "status"}], status "new", "changed" or "removed".
    """
    files = _git(repo, "diff", "--name-only", base, "--", "billie").stdout.split()
    files += _git(repo, "ls-files", "--others", "--exclude-standard", "billie").stdout.split()
    rows = []
    for path in sorted({f for f in files if f.endswith(".py")}):
        old_source = _git(repo, "show", f"{base}:{path}")
        old = definitions(old_source.stdout) if old_source.returncode == 0 else {}
        new = definitions(open(os.path.join(repo, path)).read()) if os.path.exists(os.path.join(repo, path)) else {}
        for name in sorted(set(old) | set(new)):
            if name in old and name in new and old[name][1] == new[name][1]:
                continue
            status = "new" if name not in old else "removed" if name not in new else "changed"
            rows.append({"file": path, "name": name, "kind": (new.get(name) or old[name])[0], "status": status})
    return rows


class _Source:
    """One parsed file of the repo: its references by enclosing function, its definitions and imports."""

    def __init__(self, rel: str, tree: ast.Module):
        """rel: Path relative to the repo. tree: The parsed module."""
        self.rel = rel
        self.tree = tree
        self.text = ast.unparse(tree)
        self.refs: dict[str, set[str]] = {}  # referenced name -> enclosing qualified function names
        self.defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        self.imported = {a.asname or a.name: n.module.split(".")[-1] for n in ast.walk(tree)
                         if isinstance(n, ast.ImportFrom) and n.module for a in n.names}  # name -> module stem
        self._walk(tree, "")

    def _walk(self, node: ast.AST, scope: str) -> None:
        """Records every Name and Attribute under node with its enclosing function. scope: Qualified scope."""
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
            name = child.id if isinstance(child, ast.Name) else child.attr if isinstance(child, ast.Attribute) else None
            if name is not None:
                self.refs.setdefault(name, set()).add(scope or "<module>")
            self._walk(child, inner)


def _refers_to(source: _Source, row: dict) -> bool:
    """Whether a file's references to the row's short name can mean the row's definition.

    source: The referencing file. row: An inventory row.
    Returns: False when the name clearly means something else: the file never names the row's module (or
        class, for a method), defines its own function of that name, or imports that name from another module.
    """
    if source.rel == row["file"]:
        return True
    stem = os.path.splitext(os.path.basename(row["file"]))[0]
    simple = row["name"].split(".")[-1]
    owner = row["name"].split(".")[0] if "." in row["name"] else None
    if stem not in source.text or (owner is not None and owner not in source.text):
        return False
    if owner is None and simple in source.defined:
        return False
    return source.imported.get(simple, stem) == stem


def add_docs_and_callers(rows: list[dict], repo: str) -> None:
    """Adds "doc" (first docstring line), "used_by" (["file::function"], tests left out) and "tests" (count).

    rows: Inventory rows, changed in place. repo: Path of the git checkout.
    """
    sources: dict[str, _Source] = {}
    for path in glob.glob(os.path.join(repo, "billie", "**", "*.py"), recursive=True):
        if any(skip in path for skip in _SKIP_DIRS):
            continue
        try:
            rel = os.path.relpath(path, repo)
            sources[rel] = _Source(rel, ast.parse(open(path).read()))
        except SyntaxError:
            continue
    for row in rows:
        simple = row["name"].split(".")[-1]
        row["doc"] = ""
        if row["file"] in sources:
            for node in ast.walk(sources[row["file"]].tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == simple:
                    row["doc"] = (ast.get_docstring(node) or "").strip().split("\n")[0]
                    break
        users, tests = [], 0
        for rel, source in sources.items():
            if simple not in source.refs or not _refers_to(source, row):
                continue
            for scope in source.refs[simple]:
                if rel == row["file"] and (scope == row["name"] or scope.startswith(row["name"] + ".")):
                    continue  # its own body
                if "/tests/" in rel:
                    tests += 1
                else:
                    users.append(f"{rel}::{scope}")
        row["used_by"], row["tests"] = sorted(users), tests


def main() -> None:
    """Writes the inventory of the checkout's branch against its merge-base with main."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True, help="git checkout to review (e.g. billie-onboard)")
    parser.add_argument("--base", help="commit to compare with; default: the merge-base with origin/main")
    parser.add_argument("--out", required=True, help="inventory JSON file to write")
    args = parser.parse_args()
    base = args.base or _git(args.repo, "merge-base", "HEAD", "origin/main").stdout.strip()
    rows = changed_definitions(args.repo, base)
    add_docs_and_callers(rows, args.repo)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"base": base, "rows": rows}, f, indent=1)
    print(f"{len(rows)} definitions changed since {base[:10]} -> {args.out}")


if __name__ == "__main__":
    main()
