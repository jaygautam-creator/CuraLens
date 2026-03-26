"""
CuraLens — Full System Smoke Test
===================================
End-to-end QA against the live Flask web app.

Usage:
    # Target an already-running server:
    python system_smoke_test.py --port 5001

    # Auto-start the server, run tests, then stop it:
    python system_smoke_test.py --autostart

    # Custom host / port:
    python system_smoke_test.py --host 127.0.0.1 --port 5001 --autostart

Exit codes:
    0 — all tests passed
    1 — one or more tests failed
    2 — server could not be reached
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import struct
import subprocess
import sys
import time
from typing import Any

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT          = os.path.abspath(os.path.dirname(__file__))
SAMPLE_IMAGE  = os.path.join(ROOT, "test_assets", "sample.jpg")
WEB_APP       = os.path.join(ROOT, "web_app.py")
PYTHON        = os.path.join(ROOT, "venv_oralcancer", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable  # fallback

V2_RISK_LEVELS = {"Low", "Medium", "High"}   # title-cased from utils_v2/risk_scoring.py
V2_RISK_RANGES = {
    "Low"   : (0.0, 0.3),
    "Medium": (0.3, 0.7),
    "High"  : (0.7, 1.001),
}
SERVER_STARTUP_TIMEOUT = 90   # seconds (TF model load can take ~30 s)
SERVER_POLL_INTERVAL   = 2    # seconds


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
CYAN  = "\033[96m"
RESET = "\033[0m"
BOLD  = "\033[1m"

_use_colour = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _use_colour else text


# ─────────────────────────────────────────────────────────────────────────────
# Test result tracking
# ─────────────────────────────────────────────────────────────────────────────
_results: list[dict] = []

def _record(label: str, passed: bool, detail: str = "") -> bool:
    _results.append({"label": label, "passed": passed, "detail": detail})
    icon    = _c(GREEN, "[PASS]") if passed else _c(RED, "[FAIL]")
    extra   = f"  {_c(YELLOW, detail)}" if detail else ""
    print(f"  {icon} {label}{extra}")
    return passed


def _section(title: str) -> None:
    print(f"\n{_c(BOLD, _c(CYAN, '─' * 55))}")
    print(f"  {_c(BOLD, title)}")
    print(_c(CYAN, "─" * 55))


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_png(b64_str: str) -> tuple[bool, str]:
    """Return (ok, reason). Checks base64 decoding and PNG magic bytes."""
    try:
        raw = base64.b64decode(b64_str)
    except Exception as e:
        return False, f"base64 decode error: {e}"

    # PNG magic signature: 8 bytes  \x89PNG\r\n\x1a\n
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    if len(raw) < 8:
        return False, f"too short ({len(raw)} bytes)"
    if raw[:8] != PNG_MAGIC:
        return False, f"bad PNG magic: {raw[:8]!r}"

    # Optional: verify IHDR chunk exists and is parseable
    try:
        # Bytes 8-11: IHDR chunk length
        chunk_len = struct.unpack(">I", raw[8:12])[0]
        chunk_type = raw[12:16]
        if chunk_type != b"IHDR":
            return False, f"first chunk is not IHDR: {chunk_type!r}"
    except Exception as e:
        return False, f"IHDR parse error: {e}"

    return True, ""


def _read_sample_image() -> bytes:
    if not os.path.exists(SAMPLE_IMAGE):
        print(f"\n  {_c(RED, 'ERROR')}: sample image not found — "
              f"{os.path.relpath(SAMPLE_IMAGE, ROOT)}")
        sys.exit(2)
    with open(SAMPLE_IMAGE, "rb") as f:
        return f.read()


def _post(url: str, files: dict | None = None,
          data: dict | None = None,
          timeout: int = 60) -> requests.Response:
    return requests.post(url, files=files, data=data, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Server management
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_server(base_url: str) -> bool:
    """Poll the root endpoint until the server responds or timeout."""
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    printed  = False
    while time.time() < deadline:
        try:
            r = requests.get(base_url, timeout=3)
            if r.status_code < 500:
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        if not printed:
            print(f"  ⏳ Waiting for server at {base_url} "
                  f"(up to {SERVER_STARTUP_TIMEOUT}s) …", end="", flush=True)
            printed = True
        else:
            print(".", end="", flush=True)
        time.sleep(SERVER_POLL_INTERVAL)
    print()
    return False


def _start_server(host: str, port: int) -> subprocess.Popen | None:
    """Launch web_app.py as a background subprocess."""
    env = os.environ.copy()
    env.setdefault("FLASK_ENV", "development")
    proc = subprocess.Popen(
        [PYTHON, WEB_APP, str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=ROOT,
    )
    print(f"  🚀 Started server (PID {proc.pid}) on port {port}")
    return proc


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE — /predict  (v1)
# ─────────────────────────────────────────────────────────────────────────────

def test_v1(base_url: str, img_bytes: bytes) -> None:
    _section("/predict  (v1 — Image-only Model)")
    url = f"{base_url}/predict"

    # ── happy path ─────────────────────────────────────────────────────────
    try:
        r = _post(url,
                  files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
                  data={"mode": "diagnostic"})
    except requests.exceptions.RequestException as e:
        _record("v1 endpoint reachable", False, str(e))
        return

    _record("v1 endpoint reachable",    r.status_code != 0)
    _record("v1 status code == 200",    r.status_code == 200,
            f"got {r.status_code}" if r.status_code != 200 else "")

    try:
        body: dict[str, Any] = r.json()
    except Exception as e:
        _record("v1 response is valid JSON", False, str(e))
        return

    _record("v1 response is valid JSON",        True)
    _record("v1 field: cancer_probability",
            "cancer_probability" in body,
            "key missing" if "cancer_probability" not in body else "")
    _record("v1 field: prediction  (label)",
            "prediction" in body,
            "key missing" if "prediction" not in body else "")
    _record("v1 field: risk_level",
            "risk_level" in body,
            "key missing" if "risk_level" not in body else "")
    _record("v1 field: recommendation",
            "recommendation" in body,
            "key missing" if "recommendation" not in body else "")
    _record("v1 field: success == True",
            body.get("success") is True,
            f"got {body.get('success')!r}")

    prob = body.get("cancer_probability")
    if prob is not None:
        _record("v1 cancer_probability in [0, 1]",
                0.0 <= float(prob) <= 1.0,
                f"got {prob}")

    # ── malformed: no image ────────────────────────────────────────────────
    try:
        r_bad = _post(url, data={"mode": "diagnostic"})
        code  = r_bad.status_code
        _record("v1 missing image → 400",
                code == 400,
                f"got {code} (expected 400)")
        try:
            err_body = r_bad.json()
            _record("v1 missing-image error body has 'error' key",
                    "error" in err_body,
                    "key missing" if "error" not in err_body else "")
        except Exception:
            _record("v1 missing-image error body has 'error' key",
                    False, "response not JSON")
    except requests.exceptions.RequestException as e:
        _record("v1 missing image → 400", False, str(e))

    # ── malformed: empty filename ──────────────────────────────────────────
    try:
        r_empty = _post(url,
                        files={"image": ("", b"", "image/jpeg")},
                        data={"mode": "diagnostic"})
        code = r_empty.status_code
        _record("v1 empty filename → 400",
                code == 400,
                f"got {code}")
    except requests.exceptions.RequestException as e:
        _record("v1 empty filename → 400", False, str(e))

    # ── malformed: corrupt image bytes ────────────────────────────────────
    try:
        r_corrupt = _post(url,
                          files={"image": ("bad.jpg", b"NOT_AN_IMAGE_!@#$", "image/jpeg")})
        code = r_corrupt.status_code
        _record("v1 corrupt image → 400",
                code == 400,
                f"got {code}")
    except requests.exceptions.RequestException as e:
        _record("v1 corrupt image → 400", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE — /predict_v2  (v2)
# ─────────────────────────────────────────────────────────────────────────────

def test_v2(base_url: str, img_bytes: bytes) -> None:
    _section("/predict_v2  (v2 — Multimodal Model)")
    url = f"{base_url}/predict_v2"

    # ── happy path — individual metadata fields ───────────────────────────
    try:
        r = _post(url,
                  files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
                  data={"age": "45", "smoking": "1",
                        "alcohol": "0", "sun_exposure": "3"})
    except requests.exceptions.RequestException as e:
        _record("v2 endpoint reachable", False, str(e))
        return

    _record("v2 endpoint reachable",    r.status_code != 0)
    _record("v2 status code == 200",    r.status_code == 200,
            f"got {r.status_code}" if r.status_code != 200 else "")

    try:
        body: dict[str, Any] = r.json()
    except Exception as e:
        _record("v2 response is valid JSON", False, str(e))
        return

    _record("v2 response is valid JSON", True)

    # ── required v2 fields ────────────────────────────────────────────────
    required = ["probability", "risk_level", "recommendation",
                "gradcam_png_b64", "metadata_used", "risk_label",
                "confidence_band", "color_code", "version"]
    for field in required:
        _record(f"v2 field: {field}",
                field in body,
                "missing" if field not in body else "")

    # ── probability range ─────────────────────────────────────────────────
    prob = body.get("probability")
    if prob is not None:
        _record("v2 probability in [0, 1]",
                0.0 <= float(prob) <= 1.0,
                f"got {prob}")
    else:
        _record("v2 probability in [0, 1]", False, "field absent")

    # ── risk_level tier mapping ───────────────────────────────────────────
    risk = body.get("risk_level", "")
    _record("v2 risk_level is a valid tier",
            risk in V2_RISK_LEVELS,
            f"got {risk!r}, expected one of {V2_RISK_LEVELS}")

    if risk in V2_RISK_LEVELS and prob is not None:
        lo, hi = V2_RISK_RANGES[risk]
        _record("v2 risk_level matches probability range",
                lo <= float(prob) < hi,
                f"probability={prob:.4f} not in [{lo}, {hi}) for {risk}")

    # ── metadata_used echoed back correctly ───────────────────────────────
    meta = body.get("metadata_used")
    if isinstance(meta, dict):
        meta_ok = (
            abs(float(meta.get("age", -1))          - 45.0) < 0.01 and
            abs(float(meta.get("smoking", -1))       -  1.0) < 0.01 and
            abs(float(meta.get("alcohol", -1))       -  0.0) < 0.01 and
            abs(float(meta.get("sun_exposure", -1))  -  3.0) < 0.01
        )
        _record("v2 metadata_used echoes sent values",
                meta_ok,
                f"got {meta}" if not meta_ok else "")
    else:
        _record("v2 metadata_used echoes sent values",
                False, f"metadata_used is {type(meta).__name__}, not dict")

    # ── Grad-CAM PNG validation ────────────────────────────────────────────
    gcam = body.get("gradcam_png_b64")
    if gcam is not None:
        ok, reason = _is_valid_png(gcam)
        _record("v2 gradcam_png_b64 decodes to valid PNG",
                ok, reason if not ok else "")
        if ok:
            raw = base64.b64decode(gcam)
            _record("v2 gradcam PNG size > 50 bytes",
                    len(raw) > 50,
                    f"got {len(raw)} bytes")
    else:
        _record("v2 gradcam_png_b64 decodes to valid PNG",
                False, "field is None — Grad-CAM generation may have failed")

    # ── happy path — JSON-encoded metadata field ──────────────────────────
    meta_json = json.dumps({"age": 30, "smoking": 0,
                             "alcohol": 1, "sun_exposure": 5})
    try:
        r2 = _post(url,
                   files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
                   data={"metadata": meta_json})
        _record("v2 JSON-encoded metadata field accepted",
                r2.status_code == 200,
                f"got {r2.status_code}")
    except requests.exceptions.RequestException as e:
        _record("v2 JSON-encoded metadata field accepted", False, str(e))

    # ── malformed: no image ────────────────────────────────────────────────
    try:
        r_bad = _post(url, data={"age": "45", "smoking": "0",
                                  "alcohol": "0", "sun_exposure": "0"})
        code = r_bad.status_code
        _record("v2 missing image → 400",
                code == 400,
                f"got {code} (expected 400)")
        try:
            err_body = r_bad.json()
            _record("v2 missing-image error body has 'error' key",
                    "error" in err_body,
                    "key missing" if "error" not in err_body else "")
        except Exception:
            _record("v2 missing-image error body has 'error' key",
                    False, "not JSON")
    except requests.exceptions.RequestException as e:
        _record("v2 missing image → 400", False, str(e))

    # ── malformed: missing metadata (should default gracefully, not crash) ──
    try:
        r_nm = _post(url,
                     files={"image": ("sample.jpg", img_bytes, "image/jpeg")})
        _record("v2 missing metadata defaults gracefully (2xx)",
                200 <= r_nm.status_code < 300,
                f"got {r_nm.status_code}")
    except requests.exceptions.RequestException as e:
        _record("v2 missing metadata defaults gracefully (2xx)",
                False, str(e))

    # ── malformed: corrupt image bytes ────────────────────────────────────
    try:
        r_corrupt = _post(url,
                          files={"image": ("bad.jpg",
                                           b"GARBAGE_BYTES_XYZ!",
                                           "image/jpeg")},
                          data={"age": "30"})
        code = r_corrupt.status_code
        _record("v2 corrupt image → 400",
                code == 400,
                f"got {code}")
    except requests.exceptions.RequestException as e:
        _record("v2 corrupt image → 400", False, str(e))

    # ── malformed: invalid JSON in metadata field ─────────────────────────
    try:
        r_badjson = _post(url,
                          files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
                          data={"metadata": "{NOT_VALID_JSON!!"})
        code = r_badjson.status_code
        _record("v2 malformed JSON metadata → 400",
                code == 400,
                f"got {code}")
    except requests.exceptions.RequestException as e:
        _record("v2 malformed JSON metadata → 400", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary() -> int:
    passed = [r for r in _results if r["passed"]]
    failed = [r for r in _results if not r["passed"]]
    total  = len(_results)

    print(f"\n{'═' * 55}")
    print(f"  {_c(BOLD, 'SMOKE TEST SUMMARY')}")
    print(f"{'═' * 55}")
    print(f"  Total  : {total}")
    print(f"  {_c(GREEN, 'Passed')} : {len(passed)}")
    print(f"  {_c(RED,   'Failed')} : {len(failed)}")

    if failed:
        print(f"\n  {_c(RED, _c(BOLD, 'Failed checks:'))}")
        for r in failed:
            detail = f"  ({r['detail']})" if r["detail"] else ""
            print(f"    {_c(RED, '✗')} {r['label']}{detail}")
        print()
        return 1

    print(f"\n  {_c(GREEN, _c(BOLD, '✅ All checks passed!'))}")
    print()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CuraLens end-to-end smoke test")
    parser.add_argument("--host",      default="127.0.0.1",
                        help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port",      type=int, default=5001,
                        help="Server port (default: 5001)")
    parser.add_argument("--autostart", action="store_true",
                        help="Auto-start web_app.py if server is not running")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"\n{'═' * 55}")
    print(f"  {_c(BOLD, 'CuraLens — System Smoke Test')}")
    print(f"  Target : {base_url}")
    print(f"  Image  : {os.path.relpath(SAMPLE_IMAGE, ROOT)}")
    print(f"{'═' * 55}")

    # ── verify sample image exists ─────────────────────────────────────────
    img_bytes = _read_sample_image()
    print(f"\n  📷 Loaded sample image ({len(img_bytes):,} bytes)")

    # ── server availability ────────────────────────────────────────────────
    server_proc: subprocess.Popen | None = None
    try:
        requests.get(base_url, timeout=3)
        server_running = True
    except requests.exceptions.RequestException:
        server_running = False

    if not server_running:
        if args.autostart:
            print(f"\n  Server not reachable — auto-starting web_app.py …")
            server_proc = _start_server(args.host, args.port)
            server_running = _wait_for_server(base_url)
            if not server_running:
                print(
                    f"\n  {_c(RED, 'ERROR')}: Server did not come up within "
                    f"{SERVER_STARTUP_TIMEOUT}s. Aborting."
                )
                if server_proc:
                    server_proc.terminate()
                sys.exit(2)
            print(f"\n  {_c(GREEN, '✅ Server is ready')}")
        else:
            print(
                f"\n  {_c(RED, 'ERROR')}: Cannot reach server at {base_url}.\n"
                f"  Start web_app.py first, or use --autostart flag.\n"
            )
            sys.exit(2)
    else:
        print(f"  {_c(GREEN, '✅ Server already running at')} {base_url}")

    # ── run test suites ────────────────────────────────────────────────────
    try:
        test_v1(base_url, img_bytes)
        test_v2(base_url, img_bytes)
    finally:
        if server_proc is not None:
            print(f"\n  🛑 Stopping server (PID {server_proc.pid}) …")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    exit_code = _print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
