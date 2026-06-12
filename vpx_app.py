#!/usr/bin/env python3
"""
VPX v2 — local dev server (zero-dependency).

Serves the static SPA (public/index.html) and the same JSON endpoints the
Vercel functions expose, all backed by the shared vpx_web layer. Use this for
local development; on Vercel the api/*.py functions take over.

    python3 vpx_app.py            # http://127.0.0.1:8765
    python3 vpx_app.py --port N --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import vpx_sim as E
import vpx_web


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, vpx_web.page_html(), "text/html; charset=utf-8")
        elif path == "/api/meta":
            self._json(vpx_web.meta())
        elif path == "/api/scenarios":
            self._json(vpx_web.scenarios_get())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        path = urlparse(self.path).path
        if path == "/api/simulate":
            self._json(vpx_web.simulate(body))
        elif path == "/api/optimize":
            self._json(vpx_web.do_optimize(body))
        elif path == "/api/scenarios":
            self._json(vpx_web.scenarios_post(body))
        else:
            self._send(404, b"not found", "text/plain")

    def do_DELETE(self):
        u = urlparse(self.path)
        if u.path == "/api/scenarios":
            sid = parse_qs(u.query).get("id", [""])[0]
            self._json(vpx_web.scenarios_delete(sid))
        else:
            self._send(404, b"not found", "text/plain")


def main() -> int:
    ap = argparse.ArgumentParser(description="VPX local dev server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://{}:{}".format(args.host, args.port)
    store = "Turso" if vpx_web.scenarios_available() else "localStorage (no Turso env set)"
    print("VPX UI running at  " + url + "   scenarios -> " + store + "   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
