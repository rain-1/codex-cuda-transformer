#!/usr/bin/env python3
"""Convert Simple English Wikipedia HTML dumps into plain text files."""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def extract_plain_text(source: str) -> str:
    """Return a cleaned text version of the article body."""
    soup = BeautifulSoup(source, "html.parser")

    # Drop non-content elements globally to avoid stray scripts/styles.
    for tag in soup.find_all(["script", "style", "noscript", "link", "meta"]):
        tag.decompose()

    body = soup.find("div", id="bodyContent") or soup.find("div", id="content") or soup.body or soup

    for br in body.find_all("br"):
        br.replace_with("\n")

    # Remove navigation/footer regions that are not part of the article text.
    for selector in [
        ("div", {"class": "printfooter"}),
        ("div", {"id": "catlinks"}),
        (None, {"id": "siteSub"}),
        ("table", {"id": "toc"}),
        ("span", {"class": "editsection"}),
    ]:
        name, attrs = selector
        for element in body.find_all(name, attrs):
            element.decompose()

    block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "pre", "blockquote"}

    blocks: list[str] = []
    first_heading = soup.find("h1", class_="firstHeading")
    if first_heading is not None:
        title_text = first_heading.get_text(strip=True)
        lowered_title = title_text.lower()
        months = (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        )
        for month in months:
            if lowered_title.startswith(month + " "):
                remainder = lowered_title[len(month) :].strip()
                if remainder.isdigit():
                    return ""
        if title_text.isdigit():
            return ""
            return ""

    for element in body.find_all(block_tags):
        if element.find_parent(block_tags):
            continue
        text = element.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        if not text:
            continue
        lower = text.lower()
        if "this short article needs someone to make it better" in lower:
            continue
        if element.name == "li":
            text = f"- {text}"
        blocks.append(html.unescape(text))

    if blocks:
        return "\n\n".join(blocks)

    text = body.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return html.unescape(text)


def convert_file(html_path: Path, input_root: Path, output_root: Path, overwrite: bool) -> None:
    relative = html_path.relative_to(input_root)
    output_path = (output_root / relative).with_suffix(".txt")
    if not overwrite and output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    lowered = raw_html.lower()
    if "#redirect" in lowered:
        return
    if "http-equiv=\"refresh\"" in lowered or "http-equiv='refresh'" in lowered:
        return
    if "redirecting to" in lowered:
        return
    if "this short article needs someone to make it better" in lowered:
        return
    if "this page has been deleted" in lowered:
        return

    text = extract_plain_text(raw_html)
    if not text:
        return
    output_path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the root directory containing HTML files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will receive the plain-text output.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Optional single HTML file to convert (absolute or relative to input_dir).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate text files even if they already exist.",
    )

    args = parser.parse_args(argv)

    if args.file is not None:
        target = args.file
        if not target.is_absolute():
            target = args.input_dir / target
        if not target.exists():
            print(f"[error] HTML file {target} not found", file=sys.stderr)
            return 1
        html_files = [target]
    else:
        html_files = sorted(args.input_dir.rglob("*.html"))
    if not html_files:
        print(f"[warn] No HTML files found under {args.input_dir}", file=sys.stderr)
        return 1

    skip_prefixes = (
        "Talk~",
        "User~",
        "User_talk~",
        "Image~",
        "File~",
        "Template~",
        "Category~",
    )

    def _should_skip(path: Path) -> bool:
        name = path.name
        return any(name.startswith(prefix) for prefix in skip_prefixes)

    converted = 0
    for idx, path in enumerate(html_files, start=1):
        relative = path.relative_to(args.input_dir)
        if _should_skip(relative):
            continue
        convert_file(path, args.input_dir, args.output_dir, args.overwrite)
        converted += 1
        if idx % 500 == 0:
            print(f"Processed {idx} files", file=sys.stderr)

    if converted == 0:
        print("[warn] No pages converted (all matched skip filters)", file=sys.stderr)
    print(f"Converted {converted} files into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
