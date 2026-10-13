"""The demo notebook must be loadable and every code cell must be valid.

A notebook whose `source` lines are missing their trailing newlines looks
fine in JSON but collapses every cell into one unrunnable line. That shipped
once; this test stops it shipping again.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _strip_ipython(code: str) -> str:
    """Replace shell/magic lines with `pass`, preserving indentation."""
    out = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
class TestNotebook:
    def test_is_valid_json_with_cells(self, path):
        notebook = json.loads(path.read_text())
        assert notebook["cells"], f"{path.name} has no cells"
        assert notebook["nbformat"] >= 4

    def test_source_lines_end_with_newline(self, path):
        """nbformat requires it; without it every cell becomes one line."""
        notebook = json.loads(path.read_text())
        offenders = []
        for index, cell in enumerate(notebook["cells"]):
            source = cell["source"]
            if not isinstance(source, list):
                continue
            for line in source[:-1]:
                if not line.endswith("\n"):
                    offenders.append(f"cell {index}: {line[:50]!r}")
        assert not offenders, "source lines missing trailing newline:\n" + "\n".join(
            offenders[:5]
        )

    def test_code_cells_are_syntactically_valid(self, path):
        notebook = json.loads(path.read_text())
        errors = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = _strip_ipython("".join(cell["source"]))
            try:
                ast.parse(code)
            except SyntaxError as error:
                errors.append(f"cell {index} line {error.lineno}: {error.msg}")
        assert not errors, "code cells do not parse:\n" + "\n".join(errors)

    def test_outputs_are_cleared(self, path):
        """Committed notebooks must not carry outputs (CONTRIBUTING.md)."""
        notebook = json.loads(path.read_text())
        with_output = [
            index
            for index, cell in enumerate(notebook["cells"])
            if cell.get("outputs")
        ]
        assert not with_output, f"cells with outputs: {with_output}"
