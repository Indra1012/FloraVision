"""FloraVision Standalone Glassmorphic Server (Phase 7).
Zero external web dependencies — uses Python's built-in http.server module.
"""

import os
import sys
import json
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Safe stdout handling for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from inference import get_pipeline

STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
REPORTS_DIR = PROJECT_ROOT / "reports"


class FloraVisionHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving static files, HTML templates, and the Computer Vision API."""

    def end_headers(self):
        # Enable CORS and disable caching during dev
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            index_path = TEMPLATES_DIR / "index.html"
            content = index_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = STATIC_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                content = file_path.read_bytes()
                mime = "text/css" if file_path.suffix == ".css" else ("application/javascript" if file_path.suffix == ".js" else "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", f"{mime}; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Static file not found")
                return

        elif path.startswith("/reports/"):
            rel_path = path[len("/reports/"):]
            file_path = REPORTS_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                content = file_path.read_bytes()
                mime = "image/png" if file_path.suffix == ".png" else "text/plain"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Report file not found")
                return

        elif path == "/api/samples":
            valid_exts = {".jpg", ".jpeg", ".png"}
            samples = [f.name for f in SAMPLES_DIR.iterdir() if f.suffix.lower() in valid_exts]
            resp_bytes = json.dumps({"samples": sorted(samples)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        elif path == "/api/analyze-sample":
            query = urllib.parse.parse_qs(parsed.query)
            filename = query.get("filename", [""])[0]
            sample_path = SAMPLES_DIR / filename
            if not sample_path.exists():
                self.send_error(404, f"Sample leaf {filename} not found")
                return

            try:
                pipeline = get_pipeline()
                results = pipeline.analyze(sample_path)
                resp_bytes = json.dumps(results).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
                return
            except Exception as e:
                self.send_error(500, f"Analysis failed: {str(e)}")
                return

        elif path == "/api/health":
            resp_bytes = json.dumps({"status": "online", "model": "MobileNetV2 (96.95% Test Acc)"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Extract image bytes from multipart/form-data
            image_bytes = None
            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[1].encode()
                parts = body.split(boundary)
                for part in parts:
                    if b"filename=" in part:
                        # Header ends at \r\n\r\n
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            # Image data is after header until final \r\n
                            image_bytes = part[header_end + 4:].rstrip(b"\r\n--")
                            break
            else:
                image_bytes = body

            if not image_bytes:
                self.send_error(400, "No image payload found in request")
                return

            try:
                pipeline = get_pipeline()
                results = pipeline.analyze(image_bytes)
                resp_bytes = json.dumps(results).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
                return
            except Exception as e:
                self.send_error(500, f"Analysis failed: {str(e)}")
                return

        self.send_error(404, "Endpoint not found")


def run(host="127.0.0.1", port=8000):
    server_address = (host, port)
    httpd = HTTPServer(server_address, FloraVisionHandler)
    print(f"\n=======================================================")
    print(f"🌿 FloraVision Apple Glass Web UI Server Active!")
    print(f"🔗 Open in your browser: http://{host}:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    run()

