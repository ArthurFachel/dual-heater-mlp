import json

import pytest
import torch
from torch import nn

from dual_heater.transformer import (
    SlowHeatAttentionTracker,
    SlowHeatFFNTracker,
)
from experiments.live_telemetry import (
    TelemetryWriter,
    read_events,
)


class _TrackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.ModuleList([SlowHeatAttentionTracker(2, 3)])
        self.ffn = nn.ModuleList([SlowHeatFFNTracker(5)])

    def get_attention_trackers(self):
        return list(self.attention)

    def get_ffn_trackers(self):
        return list(self.ffn)


def test_jsonl_reader_ignores_a_truncated_final_line(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"sequence":1,"event":"first"}\n'
        '{"sequence":2,"event":"second"}\n'
        '{"sequence":3,"event":',
        encoding="utf-8",
    )

    assert [event["sequence"] for event in read_events(events)] == [1, 2]
    assert [event["sequence"] for event in read_events(events, after=1)] == [2]


def test_writer_starts_a_valid_session_after_a_truncated_line(tmp_path):
    first = TelemetryWriter(tmp_path, identity={"run": "partial"})
    first.emit("batch", method_step=10)
    first._handle.write('{"sequence":3,"event":')
    first._handle.flush()
    first._handle.close()
    first._closed = True

    resumed = TelemetryWriter(
        tmp_path,
        identity={"run": "partial"},
        resumed=True,
    )
    resumed.close()

    events = read_events(resumed.events_path)
    assert [event["event"] for event in events][-2:] == [
        "session_start",
        "session_end",
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3, 4]


def test_writer_resumes_sequence_and_rejects_an_incompatible_identity(tmp_path):
    identity = {"config": {"seed": 4}, "data_sha256": "abc"}
    first = TelemetryWriter(tmp_path, identity=identity, every=10)
    first.emit("batch", method_step=10)
    first.close()
    last_sequence = read_events(first.events_path)[-1]["sequence"]

    resumed = TelemetryWriter(tmp_path, identity=identity, every=5, resumed=True)
    assert resumed.sequence == last_sequence + 1
    assert resumed.should_publish_batch(5)
    resumed.close()

    with pytest.raises(RuntimeError, match="protocolo incompatível"):
        TelemetryWriter(tmp_path, identity={"config": {"seed": 5}})


def test_writer_records_failure_without_hiding_the_original_error(tmp_path):
    writer = TelemetryWriter(tmp_path, identity={"run": 1})

    with pytest.raises(RuntimeError, match="falha simulada"), writer:
        raise RuntimeError("falha simulada")

    events = read_events(writer.events_path)
    assert events[-1]["event"] == "session_error"
    assert events[-1]["error_type"] == "RuntimeError"


def test_complete_heat_snapshot_contains_each_head_and_ffn_neuron(tmp_path):
    model = _TrackedModel()
    model.attention[0].task_ema.copy_(torch.tensor([0.25, 1.75]))
    model.ffn[0].slow_heat.copy_(torch.tensor([0.0, 1.0, 0.5, 0.0, 0.0]))
    writer = TelemetryWriter(tmp_path, identity={"run": 2})

    snapshot = writer.publish_heat(
        model,
        context={"method": "slowheat", "stage": 0},
        stage_snapshot=True,
    )
    writer.close()

    assert snapshot["available"] is True
    assert len(snapshot["attention"][0]["task_importance"]) == 2
    assert len(snapshot["ffn"][0]["heat"]) == 5
    history = next((tmp_path / "telemetry/heat").glob("slowheat-stage-01-*.json"))
    assert json.loads(history.read_text(encoding="utf-8"))["sequence"] == 1


def test_batch_cadence_is_exact():
    writer = object.__new__(TelemetryWriter)
    writer.every = 10

    published = [step for step in range(1, 31) if writer.should_publish_batch(step)]

    assert published == [10, 20, 30]
