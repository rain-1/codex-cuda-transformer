#!/usr/bin/env python3
"""Lightweight paper queue app server.

Serves static UI and provides a tiny ArXiv metadata proxy endpoint:
GET /api/arxiv?query=<arxiv-id-or-url>
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
STATIC_DIR = Path(__file__).parent / "static"

ARXIV_API = "https://export.arxiv.org/api/query?id_list={id_list}"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class BadRequest(Exception):
    """Expected validation error."""


def extract_arxiv_id(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise BadRequest("Missing arXiv query. Provide an arXiv ID or URL.")

    if text.startswith("http://") or text.startswith("https://"):
        parsed = urllib.parse.urlparse(text)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"abs", "pdf"}:
            text = parts[1]
        elif parts:
            text = parts[-1]

    text = text.removesuffix(".pdf")
    if "v" in text:
        text = re.sub(r"v\d+$", "", text)

    modern = re.fullmatch(r"\d{4}\.\d{4,5}", text)
    legacy = re.fullmatch(r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}", text, flags=re.IGNORECASE)
    if modern or legacy:
        return text
    raise BadRequest(f"Could not parse arXiv ID from '{raw}'.")


def text_or_empty(node: ET.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def fetch_arxiv_metadata(query: str) -> dict:
    arxiv_id = extract_arxiv_id(query)
    url = ARXIV_API.format(id_list=urllib.parse.quote(arxiv_id))
    req = urllib.request.Request(url, headers={"User-Agent": "paper-queue/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = resp.read()
    except urllib.error.URLError as exc:
        raise BadRequest(f"Failed contacting arXiv API: {exc}") from exc

    root = ET.fromstring(payload)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise BadRequest(f"No paper found for arXiv ID '{arxiv_id}'.")

    title = re.sub(r"\s+", " ", text_or_empty(entry.find("atom:title", ATOM_NS)))
    abstract = re.sub(r"\s+", " ", text_or_empty(entry.find("atom:summary", ATOM_NS)))

    authors = [
        text_or_empty(author.find("atom:name", ATOM_NS))
        for author in entry.findall("atom:author", ATOM_NS)
        if text_or_empty(author.find("atom:name", ATOM_NS))
    ]

    categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ATOM_NS)]
    categories = [c for c in categories if c]

    pdf_url = ""
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href", "")
        if link.attrib.get("title") == "pdf" or href.endswith(".pdf"):
            pdf_url = href
            break

    canonical_id = text_or_empty(entry.find("atom:id", ATOM_NS))
    if canonical_id:
        canonical_id = canonical_id.rstrip("/").split("/")[-1]

    return {
        "id": canonical_id or arxiv_id,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published": text_or_empty(entry.find("atom:published", ATOM_NS)),
        "updated": text_or_empty(entry.find("atom:updated", ATOM_NS)),
        "categories": categories,
        "pdf_url": pdf_url,
        "arxiv_url": f"https://arxiv.org/abs/{canonical_id or arxiv_id}",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/arxiv":
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("query", [""])[0]
            try:
                result = fetch_arxiv_metadata(query)
                self.send_json(HTTPStatus.OK, result)
            except BadRequest as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Unexpected server error: {exc}"})
            return

        if path == "/":
            path = "/index.html"

        file_path = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }.get(file_path.suffix, "application/octet-stream")

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, status: HTTPStatus, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    print(f"Paper Queue app on http://{HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()
