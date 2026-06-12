"""Shared bootstrap for the Vercel serverless functions.

Puts the repo root on sys.path so the api/*.py functions can import the engine
(vpx_sim), the web glue (vpx_web), and the store (vpx_store), which live one
directory up. vercel.json's includeFiles ensures those modules ship in the
function bundle.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
