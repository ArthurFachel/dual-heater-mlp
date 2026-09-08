"""Local, read-only HTTP dashboard for BERT SlowHeat telemetry."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from experiments.artifacts import read_json_object
from experiments.live_telemetry import read_events

_HOST = "127.0.0.1"
_HTML_PATH = Path(__file__).with_name("live_dashboard.html")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def discover_runs(run_dir: str | Path) -> dict[str, Path]:
    """Return telemetry directories constrained to the selected run root."""

    root = Path(run_dir).resolve()
    if not root.is_dir():
        return {}
    candidates: list[Path] = []
    if root.name == "telemetry" and (root / "manifest.json").is_file():
        candidates.append(root)
    candidates.extend(
        manifest.parent
        for manifest in root.rglob("manifest.json")
        if manifest.parent.name == "telemetry"
    )
    discovered: dict[str, Path] = {}
    for candidate in sorted(set(candidates)):
        resolved = candidate.resolve()
        if not _is_within(resolved, root):
            continue
        if resolved == root:
            discovered["."] = resolved
            continue
        owner = resolved.parent
        relative = owner.relative_to(root).as_posix()
        run_id = "." if relative in {"", "."} else relative
        discovered[run_id] = resolved
    return discovered


def _run_summaries(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for identifier, telemetry_dir in discover_runs(root).items():
        try:
            manifest = read_json_object(telemetry_dir / "manifest.json")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        events = read_events(telemetry_dir / "events.jsonl")
        latest = events[-1] if events else None
        identity = manifest.get("identity", {})
        config = identity.get("config", {}) if isinstance(identity, dict) else {}
        summaries.append(
            {
                "id": identifier,
                "run_id": manifest.get("run_id"),
                "created_at": manifest.get("created_at"),
                "seed": config.get("seed") if isinstance(config, dict) else None,
                "model_name": (
                    config.get("model_name") if isinstance(config, dict) else None
                ),
                "last_sequence": latest.get("sequence") if latest else 0,
                "last_event": latest.get("event") if latest else None,
                "last_timestamp": latest.get("timestamp") if latest else None,
            }
        )
    summaries.sort(
        key=lambda item: item.get("last_timestamp") or "",
        reverse=True,
    )
    return summaries


def _selected_telemetry(root: Path, identifier: str) -> Path | None:
    return discover_runs(root).get(identifier)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        return read_json_object(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _handler_class(root: Path, html: bytes):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "SlowHeatDashboard/1"

        def _send_bytes(
            self,
            payload: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(
            self,
            payload: Any,
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self._send_bytes(
                encoded,
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def _query_run(self, query: dict[str, list[str]]) -> Path | None:
            identifier = query.get("run", [""])[0]
            return _selected_telemetry(root, identifier)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_bytes(html, content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/health":
                self._send_json({"status": "ok", "host": _HOST})
                return
            if parsed.path == "/api/runs":
                self._send_json({"runs": _run_summaries(root)})
                return
            if parsed.path not in {"/api/events", "/api/heat", "/api/stages"}:
                self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            telemetry_dir = self._query_run(query)
            if telemetry_dir is None:
                self._send_json({"error": "invalid_run"}, status=HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/events":
                try:
                    after = int(query.get("after", ["0"])[0])
                    limit = min(10_000, int(query.get("limit", ["2000"])[0]))
                    events = read_events(
                        telemetry_dir / "events.jsonl",
                        after=after,
                        limit=limit,
                    )
                except ValueError:
                    self._send_json(
                        {"error": "invalid_query"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._send_json({"events": events})
                return
            if parsed.path == "/api/heat":
                self._send_json(
                    {"heat": _read_optional_json(telemetry_dir / "heat-latest.json")}
                )
                return
            snapshots = []
            history = telemetry_dir / "heat"
            if history.is_dir():
                for path in sorted(history.glob("*.json")):
                    payload = _read_optional_json(path)
                    if payload is not None:
                        snapshots.append(
                            {
                                "name": path.name,
                                "sequence": payload.get("sequence"),
                                "context": payload.get("context"),
                            }
                        )
            self._send_json({"stages": snapshots})

        def do_POST(self) -> None:
            self._send_json(
                {"error": "read_only"}, status=HTTPStatus.METHOD_NOT_ALLOWED
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def create_server(
    run_dir: str | Path,
    *,
    port: int = 8765,
) -> ThreadingHTTPServer:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"diretório de resultados inexistente: {root}")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65_535:
        raise ValueError("port deve estar em [0, 65535]")
    html = _HTML_PATH.read_bytes()
    server = ThreadingHTTPServer((_HOST, port), _handler_class(root, html))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        server = create_server(args.run_dir, port=args.port)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))
    host, port = server.server_address
    print(f"SlowHeat dashboard: http://{host}:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
