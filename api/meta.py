import json
from http.server import BaseHTTPRequestHandler

import _bootstrap  # noqa: F401  (puts repo root on sys.path)
import vpx_web


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(vpx_web.meta()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
