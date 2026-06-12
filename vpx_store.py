#!/usr/bin/env python3
"""
VPX v2 — scenario store on Turso (libSQL), via the HTTP pipeline API.

Dependency-free: talks to Turso over HTTPS with urllib (no libsql client).
Append-only by design — every save / rename / delete INSERTs a new row with a
higher version; the "current" view folds to the latest version per id and hides
tombstones. Nothing is ever UPDATEd or hard-DELETEd, so you get a full audit
trail for free (the v2 plan's versioning requirement).

Config (set these as Vercel env vars):
    TURSO_DATABASE_URL   libsql://<db>-<org>.turso.io   (or https://...)
    TURSO_AUTH_TOKEN     <token from `turso db tokens create`>

If the env vars are absent, available() is False and the app falls back to
browser localStorage — so local dev works with no database.
"""
from __future__ import annotations

import json
import os
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS scenarios ("
    "id TEXT, version INTEGER, name TEXT, color TEXT, "
    "levers TEXT, params TEXT, kpis TEXT, deleted INTEGER DEFAULT 0, created_at TEXT)"
)


def _cfg() -> Tuple[Optional[str], Optional[str]]:
    return os.environ.get("TURSO_DATABASE_URL"), os.environ.get("TURSO_AUTH_TOKEN")


def available() -> bool:
    u, t = _cfg()
    return bool(u and t)


def _http_url(url: str) -> str:
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://"):]
    return url.rstrip("/") + "/v2/pipeline"


def _enc(v: Any) -> Dict[str, Any]:
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    return {"type": "text", "value": str(v)}


def _exec(statements: List[Tuple[str, list]]) -> list:
    url, token = _cfg()
    if not (url and token):
        raise RuntimeError("Turso not configured")
    reqs = [{"type": "execute",
             "stmt": {"sql": sql, "args": [_enc(a) for a in args]}}
            for sql, args in statements]
    reqs.append({"type": "close"})
    body = json.dumps({"requests": reqs}).encode()
    req = urllib.request.Request(
        _http_url(url), data=body,
        headers={"Authorization": "Bearer " + token,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["results"]


def _rows(execute_result: dict) -> List[Dict[str, Any]]:
    res = execute_result["response"]["result"]
    cols = [c["name"] for c in res["cols"]]
    out = []
    for row in res["rows"]:
        d = {}
        for name, cell in zip(cols, row):
            d[name] = None if cell["type"] == "null" else cell["value"]
        out.append(d)
    return out


def ensure() -> None:
    _exec([(SCHEMA, [])])


def save(sid: Optional[str], name: str, color: str,
         levers: Any, params: Any, kpis: Any, deleted: int = 0) -> str:
    ensure()
    if not sid:
        sid = "s" + uuid.uuid4().hex[:12]
    _exec([(
        "INSERT INTO scenarios "
        "(id, version, name, color, levers, params, kpis, deleted, created_at) "
        "VALUES (?, (SELECT COALESCE(MAX(version),0)+1 FROM scenarios), ?, ?, ?, ?, ?, ?, ?)",
        [sid, name, color, json.dumps(levers), json.dumps(params),
         json.dumps(kpis), int(deleted), datetime.now(timezone.utc).isoformat()],
    )])
    return sid


def delete(sid: str) -> None:
    # append a tombstone (append-only) — never a hard DELETE
    save(sid, "", "", None, None, None, deleted=1)


def list_active(limit: int = 3) -> List[Dict[str, Any]]:
    ensure()
    results = _exec([(
        "SELECT id, version, name, color, levers, params, kpis, deleted, created_at "
        "FROM scenarios ORDER BY version ASC", [])])
    rows = _rows(results[0])
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest[r["id"]] = r                       # ascending version -> last wins
    active = [r for r in latest.values() if int(r.get("deleted") or 0) == 0]
    active.sort(key=lambda r: int(r["version"]), reverse=True)
    out = []
    for r in active[:limit]:
        out.append({
            "id": r["id"], "name": r["name"], "color": r["color"],
            "levers": json.loads(r["levers"]) if r["levers"] else {},
            "params": json.loads(r["params"]) if r["params"] else {},
            "kpis": json.loads(r["kpis"]) if r["kpis"] else {},
        })
    return out
