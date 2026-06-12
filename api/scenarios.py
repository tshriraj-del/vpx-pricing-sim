import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import _bootstrap  # noqa: F401
import vpx_web


class handler(BaseHTTPRequestHandler):
    def _json(self, obj):
        out = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        self._json(vpx_web.scenarios_get())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        self._json(vpx_web.scenarios_post(body))

    def do_DELETE(self):
        sid = parse_qs(urlparse(self.path).query).get("id", [""])[0]
        self._json(vpx_web.scenarios_delete(sid))
