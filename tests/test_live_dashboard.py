import json
import threading
import urllib.error
import urllib.request

import pytest

from experiments.live_dashboard import create_server, discover_runs
from experiments.live_telemetry import TelemetryWriter


def _request(url, *, method="GET"):
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=3) as response:
        return response.status, response.headers, response.read()


@pytest.fixture
def dashboard(tmp_path):
    writer = TelemetryWriter(tmp_path, identity={"config": {"seed": 7}})
    writer.emit(
        "batch",
        method="slowheat",
        method_step=10,
        total_steps=20,
        loss=0.75,
    )
    writer.emit(
        "evaluation_end",
        method="vanilla",
        accuracy_matrix=[[0.8]],
        task_aware_accuracy_matrix=[[0.9]],
        validation_accuracy_matrix=[[0.7]],
    )
    writer.emit(
        "evaluation_end",
        method="slowheat",
        accuracy_matrix=[[0.85]],
        task_aware_accuracy_matrix=[[0.95]],
        validation_accuracy_matrix=[[0.75]],
    )
    writer.publish_heat(
        None,
        context={"method": "slowheat", "stage": 0},
    )
    writer.close()
    history = tmp_path / "telemetry/heat"
    history.mkdir()
    (history / "dualheat-stage-01-epoch-01-seq-00000002.json").write_text(
        json.dumps(
            {
                "sequence": 2,
                "available": True,
                "context": {"method": "dualheat", "stage": 0, "epoch": 0},
                "attention": [],
                "ffn": [{"layer": 0, "units": 2, "fast_heat": [0.2, 0.8]}],
            }
        ),
        encoding="utf-8",
    )
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", tmp_path, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_dashboard_serves_local_read_only_api(dashboard):
    base, _, server = dashboard
    assert server.server_address[0] == "127.0.0.1"

    status, headers, html = _request(base + "/")
    assert status == 200
    assert headers["X-Frame-Options"] == "DENY"
    assert b"BERT \xc3\x97 SlowHeat Live" in html
    assert b'value="fast_heat"' in html
    assert b'id="epochSnapshot"' in html
    assert b'id="accuracyMethod"' in html

    _, _, raw_runs = _request(base + "/api/runs")
    runs = json.loads(raw_runs)["runs"]
    assert runs[0]["id"] == "."
    assert runs[0]["seed"] == 7

    _, _, raw_events = _request(base + "/api/events?run=.&after=1")
    events = json.loads(raw_events)["events"]
    assert events[0]["sequence"] == 2

    _, _, raw_heat = _request(base + "/api/heat?run=.")
    assert json.loads(raw_heat)["heat"]["available"] is False

    _, _, raw_accuracy = _request(base + "/api/accuracy?run=.")
    accuracy = json.loads(raw_accuracy)["methods"]
    assert [item["method"] for item in accuracy] == ["vanilla", "slowheat"]
    assert accuracy[0]["accuracy_matrix"] == [[0.8]]
    assert accuracy[1]["task_aware_accuracy_matrix"] == [[0.95]]

    _, _, raw_stages = _request(base + "/api/stages?run=.")
    stages = json.loads(raw_stages)["stages"]
    assert stages[0]["context"]["method"] == "dualheat"

    _, _, raw_snapshot = _request(
        base + f"/api/snapshot?run=.&name={stages[0]['name']}"
    )
    snapshot = json.loads(raw_snapshot)["heat"]
    assert snapshot["ffn"][0]["fast_heat"] == [0.2, 0.8]


def test_dashboard_rejects_writes_and_path_traversal(dashboard):
    base, _, _ = dashboard

    with pytest.raises(urllib.error.HTTPError) as write_error:
        _request(base + "/api/events?run=.", method="POST")
    assert write_error.value.code == 405

    with pytest.raises(urllib.error.HTTPError) as traversal_error:
        _request(base + "/api/events?run=../../outside")
    assert traversal_error.value.code == 400

    with pytest.raises(urllib.error.HTTPError) as snapshot_error:
        _request(base + "/api/snapshot?run=.&name=../../outside.json")
    assert snapshot_error.value.code == 400


def test_dashboard_discovers_seed_subdirectories(tmp_path):
    first = TelemetryWriter(tmp_path / "seed_11", identity={"config": {"seed": 11}})
    second = TelemetryWriter(tmp_path / "seed_22", identity={"config": {"seed": 22}})
    first.close()
    second.close()

    assert set(discover_runs(tmp_path)) == {"seed_11", "seed_22"}
