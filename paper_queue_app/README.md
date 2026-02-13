# Paper Queue App

A lightweight local-first web app for tracking research papers.

## Features

- Add papers from ArXiv ID or URL (server-side metadata fetch).
- Queue-focused default view with status filters.
- Per-paper progress, tags, notes, project assignment, stars, and ratings.
- Project grouping with a default `Main` project.
- Local storage persistence in the browser.
- JSON export/import for backups and migration.

## Run

```bash
python3 paper_queue_app/app.py
```

Then open <http://127.0.0.1:8000>.

## Notes

This app stores data in browser localStorage under `paper_queue_v1` and imports/exports as JSON.
