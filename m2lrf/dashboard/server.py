# -*- coding: utf-8 -*-
"""
M-2LRF Interactive Web Visualizer & Engineering Dashboard Server.
Pure-Python zero-dependency local dashboard for model memory calculation,
kurtosis inspection, and 2-bit dual-basis lattice visualization.
"""

import http.server
import os
import socketserver
import sys
import threading
import webbrowser
from typing import Optional


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the dashboard static assets from m2lrf/dashboard directory."""

    def __init__(self, *args, **kwargs):
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=dashboard_dir, **kwargs)

    def log_message(self, format, *args):
        # Quiet logger
        pass


def launch_dashboard(
    port: int = 7860,
    open_browser: bool = True,
    block: bool = False,
) -> socketserver.TCPServer:
    """
    Launches the local M-2LRF web dashboard server.

    Args:
        port: Local port to bind (default 7860)
        open_browser: Whether to automatically open the web browser
        block: Whether to block the current thread
    """
    handler = DashboardHandler
    server = socketserver.TCPServer(("", port), handler)
    url = f"http://localhost:{port}"

    print(f"======================================================================")
    print(f"  M-2LRF Enterprise Interactive Dashboard running at:")
    print(f"  --> {url}")
    print(f"======================================================================")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down M-2LRF Dashboard server...")
            server.server_close()
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    launch_dashboard(port=port, open_browser=True, block=True)
