"""Dependency-free checks for the MOSAIC static evidence workbench."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.scripts: list[str] = []
        self.styles: list[str] = []
        self.viewport = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        identifier = values.get("id")
        if identifier:
            if identifier in self.ids:
                raise AssertionError(f"duplicate id: {identifier}")
            self.ids.add(identifier)
        if tag == "a" and "href" in values:
            self.hrefs.append(values["href"])
        if tag == "script" and "src" in values:
            self.scripts.append(values["src"])
        if tag == "link" and values.get("rel") == "stylesheet":
            self.styles.append(values.get("href", ""))
        if tag == "meta" and values.get("name") == "viewport":
            self.viewport = True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "styles.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")
    parser = Parser()
    parser.feed(html)
    require(parser.viewport, "viewport meta is required")
    required = {"results", "lab", "protocol", "claims"}
    require(required <= parser.ids, f"missing sections: {sorted(required - parser.ids)}")
    require("styles.css" in parser.styles, "local CSS is missing")
    require("app.js" in parser.scripts, "local JS is missing")
    require(not [value for value in parser.styles + parser.scripts if value.startswith(("http:", "https:", "//"))], "external runtime asset")
    for href in parser.hrefs:
        if href.startswith("#") and len(href) > 1:
            require(href[1:] in parser.ids, f"broken anchor: {href}")
    markers = (
        "+2.89", "+7.90", "44", "SYNTHETIC INTERFACE LAB",
        "hard-negative", "不含 COCO/MSR-VTT 媒体", "NOT DATASET OUTPUT",
    )
    combined = "\n".join((html, css, js))
    for marker in markers:
        require(marker in combined, f"missing evidence marker: {marker}")
    require("@media (max-width: 540px)" in css, "390px breakpoint missing")
    require("prefers-reduced-motion" in css, "reduced motion support missing")
    require("focus-visible" in css, "keyboard focus style missing")
    for label, pattern in {
        "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
        "local drive": r"[A-Z]:\\",
        "dataset media": r"\.(?:png|jpe?g|mp4)(?:[\"'#?]|$)",
    }.items():
        require(not re.search(pattern, combined, re.IGNORECASE), f"possible {label} in docs")
    print(f"GitHub Pages checks: PASS; sections={len(required)} ids={len(parser.ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
