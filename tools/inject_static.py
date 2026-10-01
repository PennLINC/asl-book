"""Add the click-to-enlarge lightbox to every page of a built book.

The book theme has no lightbox and does not execute scripts placed in its content slots,
so this runs after ``myst build --html``: it copies ``book/_static/lightbox.css`` and
``lightbox.js`` into the HTML output and inserts a stylesheet link and a script tag into the
head of every HTML file, with paths relative to each file's depth so the result works at
any base URL.

Usage: ``python tools/inject_static.py [_build/html]``
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "book" / "_static"
FILES = ("lightbox.css", "lightbox.js")
MARKER = "aslbook-static"


def tags(prefix: str) -> str:
    return (
        f'<link rel="stylesheet" href="{prefix}lightbox.css" data-{MARKER}="1"/>'
        f'<script src="{prefix}lightbox.js" defer data-{MARKER}="1"></script>'
    )


def inject(html_dir: Path) -> int:
    for name in FILES:
        shutil.copy2(STATIC / name, html_dir / name)
    n = 0
    for page in html_dir.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        if MARKER in text or "</head>" not in text:
            continue
        depth = len(page.relative_to(html_dir).parts) - 1
        text = text.replace("</head>", tags("../" * depth) + "</head>", 1)
        page.write_text(text, encoding="utf-8")
        n += 1
    return n


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_build" / "html")
    if not out.is_dir():
        sys.exit(f"no built site at {out}; run `myst build --html` first")
    print(f"lightbox injected into {inject(out)} pages under {out}")
