#!/usr/bin/env python3
"""
Git "clean" filter for notebooks: drops every output that contains a picture.

The lab's solution notebook shows memes (and pictures made from them) in its outputs. Those memes
belong to the people who made them, so they must not end up on GitHub. With this filter turned on,
git stores each notebook without its picture outputs, while the copy on your disk keeps them.
Text outputs (checks, training logs, tables, charts drawn as SVG) are kept.

Turn it on once per clone (see .gitattributes):
    git config filter.strip-images.clean "python3 src/tools/strip_image_outputs.py"
    git config filter.strip-images.smudge cat
    git config filter.strip-images.required true
"""

import json
import sys


def has_picture(output):
    data = output.get("data", {})
    if any(kind.startswith("image/") for kind in data):
        return True
    return "data:image" in json.dumps(data)


def main():
    notebook = json.load(sys.stdin)
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            cell["outputs"] = [o for o in cell["outputs"] if not has_picture(o)]
    sys.stdout.write(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
