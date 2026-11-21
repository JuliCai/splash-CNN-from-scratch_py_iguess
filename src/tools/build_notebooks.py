#!/usr/bin/env python3
"""
Build the lab notebooks from the reference code.

    python3 tools/build_notebooks.py            # writes lab*.ipynb, addon_*.ipynb and solutions/*.py
    python3 tools/build_notebooks.py lab.ipynb  # only that notebook (+ its _solutions twin)

* babytorch.py / addons.py hold the reference code.  Blocks between "# >>> exercise: NAME" and
  "# <<< exercise: NAME" become TODO cells; "# >>> provided: NAME" blocks become ready-made cells.
* Student notebooks get stubs (signature + docstring + hint + raise NotImplementedError).
* *_solutions notebooks get the reference code in the same cells.
* solutions/NAME.py is what `%load solutions/NAME.py` pastes into a stuck student's cell.
The lesson text lives in tools/lesson_content.py.
"""

import ast
import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / "tools"))
import lesson_content as L  # noqa: E402

BLOCK = re.compile(r"^# >>> (exercise|provided): (\w+)\n(.*?)^# <<< \1: \2\n", re.S | re.M)


def read_blocks(path):
    out = {}
    for kind, name, body in BLOCK.findall(Path(path).read_text()):
        body = body.rstrip() + "\n"
        body = body.replace("  # noqa: F811 -- replaces the simple version: same answer, much faster",
                            "  # replaces the simple version: same answer, much faster")
        out[name] = (kind, body)
    return out


BLOCKS = {**read_blocks(SRC / "babytorch.py"), **read_blocks(SRC / "addons.py")}


# ------------------------------------------------------------------------------------------- stubbing
def _docstring_end(fn):
    first = fn.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first.end_lineno
    return None


def auto_stub(name, source, spec):
    """Replace the bodies of the chosen functions/methods with a hint + raise NotImplementedError."""
    tree = ast.parse(source)
    lines = source.splitlines()
    targets = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (spec.get("funcs") is None or node.name in spec["funcs"]):
            targets.append(node)
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in spec.get("methods", ()):
                    targets.append(m)
    hints = spec.get("hints", {})
    for fn in sorted(targets, key=lambda f: f.lineno, reverse=True):
        indent = " " * (fn.col_offset + 4)
        doc_end = _docstring_end(fn)
        start = doc_end + 1 if doc_end else fn.body[0].lineno
        hint = hints.get(fn.name, hints.get("*", ""))
        new = [indent + "# " + h if h else indent + "#" for h in hint.strip("\n").split("\n")] if hint else []
        new += [indent + f'raise NotImplementedError("todo: {fn.name}")']
        lines[start - 1:fn.end_lineno] = new
    return "\n".join(lines) + "\n"


def stub_for(name):
    spec = L.STUBS.get(name, {})
    if "override" in spec:
        return spec["override"].strip("\n") + "\n"
    return auto_stub(name, BLOCKS[name][1], spec)


# ------------------------------------------------------------------------------------------ notebooks
class NB:
    def __init__(self):
        self.cells = []

    def md(self, text):
        self.cells.append(("markdown", text.strip("\n")))

    def code(self, text):
        self.cells.append(("code", text.strip("\n")))

    def provided(self, name, note=None):
        kind, body = BLOCKS[name]
        assert kind == "provided", name
        head = "# given code - read it, then run it" + (f"  ({note})" if note else "")
        notes = getattr(L, "NOTES", {}).get(name)
        if notes:
            head += "\n#\n" + "\n".join("# " + line if line else "#" for line in notes.strip("\n").split("\n")) + "\n#"
        self.cells.append(("code", head + "\n" + body.rstrip()))

    def exercise(self, name):
        kind, body = BLOCKS[name]
        assert kind == "exercise", name
        self.cells.append(("exercise", name))

    def render(self, solutions):
        cells = []
        for i, (kind, payload) in enumerate(self.cells):
            if kind == "exercise":
                name = payload
                if solutions:
                    src = f"# answer (same as solutions/{name}.py)\n" + BLOCKS[name][1].rstrip()
                else:
                    src = (f"# todo: {name}   (stuck? put  %load solutions/{name}.py  in this cell and run it twice)\n"
                           + stub_for(name).rstrip())
                kind = "code"
            else:
                src = payload
            cell = {"cell_type": kind, "id": f"c{i:03d}", "metadata": {}, "source": _lines(src)}
            if kind == "code":
                cell.update({"execution_count": None, "outputs": []})
            cells.append(cell)
        return {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
                "nbformat": 4, "nbformat_minor": 5}


def _lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def main():
    out_sol = SRC / "solutions"
    out_sol.mkdir(exist_ok=True)
    built = []
    only = sys.argv[1:]
    for filename, builder in L.NOTEBOOKS.items():
        if only and filename not in only:
            continue
        nb = NB()
        builder(nb)
        for solutions in (False, True):
            path = SRC / (filename.replace(".ipynb", "_solutions.ipynb") if solutions else filename)
            path.write_text(json.dumps(nb.render(solutions), indent=1, ensure_ascii=False) + "\n")
            built.append(path.name)
    for name, (kind, body) in BLOCKS.items():   # every answer, for %load solutions/NAME.py
        if kind == "exercise":
            (out_sol / f"{name}.py").write_text(f"# answer for the todo '{name}' - run this cell to define it\n" + body)
    # every stub and solution must at least be valid Python
    for name, (kind, body) in BLOCKS.items():
        compile(body, name, "exec")
        if kind == "exercise":
            compile(stub_for(name), name + "_stub", "exec")
    print("wrote", ", ".join(built), f"and {len(list(out_sol.glob('*.py')))} solutions/*.py files")


if __name__ == "__main__":
    main()
