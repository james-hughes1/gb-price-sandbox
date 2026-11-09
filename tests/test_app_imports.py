"""Guard the 'no logic in UI files' rule: app pages may import only streamlit and the package."""

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
ALLOWED_ROOTS = {"streamlit", "gb_price_sandbox", "plotly"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_app_pages_only_import_allowed_modules():
    pages = list(APP_DIR.rglob("*.py"))
    assert pages, "no app pages found"
    for page in pages:
        extra = _imported_roots(page) - ALLOWED_ROOTS
        assert not extra, f"{page.name} imports {extra}; put logic in gb_price_sandbox"
