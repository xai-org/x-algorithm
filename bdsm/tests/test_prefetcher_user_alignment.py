import sys
import types
import queue

import pytest

pa = pytest.importorskip("pyarrow")
zstandard = pytest.importorskip("zstandard")

import batch_prefetcher
import manhattan_arrow_adapter
import manhattan_scorer
from pipeline_security import safe_pickle_loads


class _BatchPublished(Exception):
    pass


def _sequence(uid):
    table = pa.table(
        {
            "user_actions": pa.array(
                [[{"action_name": 1, "client_app_id": uid, "engagement_time_ms": 1000}]],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("action_name", pa.int32()),
                            pa.field("client_app_id", pa.int32()),
                            pa.field("engagement_time_ms", pa.int64()),
                        ]
                    )
                ),
            )
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return zstandard.ZstdCompressor().compress(sink.getvalue().to_pybytes())


@pytest.mark.parametrize("failed_uid", [11, 22, 33, None])
@pytest.mark.parametrize("freeze", [False, True])
def test_fallback_keeps_user_ids_aligned_after_extraction_failure(monkeypatch, failed_uid, freeze):
    # Exercise the actual loop, Arrow feature extractor, merger and serializer.
    # External Redis/store/Kafka boundaries are local stand-ins; one extractor
    # failure is injected into the existing per-user failure handling path.
    uids = [11, 22, 33]
    raw_by_uid = {uid: _sequence(uid) for uid in uids}
    published = []
    frozen = queue.Queue()

    class Redis:
        def __init__(self, **kwargs):
            pass

        def ping(self):
            return True

        def zpopmin(self, key, count):
            return [(str(uid), 1.0) for uid in uids]

        def pipeline(self, **kwargs):
            return self

        def get(self, key):
            return self

        def execute(self):
            return [None] * len(uids)

    class Reader:
        def __init__(self, **kwargs):
            pass

        def get_raw_bytes(self, uid):
            return raw_by_uid[uid]

    class Producer:
        def __init__(self, config):
            pass

        def produce(self, topic, value, callback):
            published.append(safe_pickle_loads(zstandard.ZstdDecompressor().decompress(value)))

        def poll(self, timeout):
            raise _BatchPublished

    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=Redis))
    monkeypatch.setitem(sys.modules, "confluent_kafka", types.SimpleNamespace(Producer=Producer))
    monkeypatch.setitem(sys.modules, "abuse_v3_features", None)
    monkeypatch.setattr(manhattan_scorer, "ManhattanArrowReader", Reader)
    monkeypatch.setenv("FREEZE_ENABLED", "1" if freeze else "0")
    if freeze:
        # Retain the real freeze enqueue path, inspecting its local queue rather
        # than launching an unbounded background Parquet writer in this test.
        monkeypatch.setattr(batch_prefetcher.queue, "Queue", lambda **kwargs: frozen)
        thread = batch_prefetcher.threading.Thread

        def local_thread(*args, **kwargs):
            if kwargs.get("name") == "parquet_writer":
                return types.SimpleNamespace(start=lambda: None)
            return thread(*args, **kwargs)

        monkeypatch.setattr(batch_prefetcher.threading, "Thread", local_thread)
    monkeypatch.setenv("SCORING_WORK_REDIS_ENABLED", "0")
    monkeypatch.setenv("CLASSIFIER_SHADOW_ENABLED", "0")
    extract = manhattan_arrow_adapter.arrow_bytes_to_batch

    def extract_with_failure(rows, **kwargs):
        if rows[0][0] == failed_uid:
            raise ValueError("injected extraction failure")
        return extract(rows, **kwargs)

    monkeypatch.setattr(manhattan_arrow_adapter, "arrow_bytes_to_batch", extract_with_failure)
    with pytest.raises(_BatchPublished):
        batch_prefetcher.prefetcher_loop(
            redis_host="localhost", redis_port=6379, redis_key="ready",
            mh_endpoint="local", kafka_bootstrap="local", kafka_username="local",
            kafka_password="local", kafka_topic="ready_batches", batch_size=3,
            seq_len=2, n_action_types=256, metrics={}, seq_cache_enabled=False,
            realtime_cache_enabled=False,
        )

    payload = published[0]
    expected = [uid for uid in uids if uid != failed_uid]
    assert payload["uids"] == expected
    assert payload["B"] == len(expected)
    assert payload["batch"]["action_seq"]["client_app_id"][:len(expected), -1].tolist() == expected
    assert len(payload["score_ids"]) == len(expected)
    assert len(payload["action_histograms"]) == len(expected)
    assert len(payload["behavioral_summaries"]) == len(expected)
    if freeze:
        entries = [frozen.get_nowait() for _ in expected]
        assert [entry[0] for entry in entries] == expected
        assert [entry[1] for entry in entries] == [raw_by_uid[uid] for uid in expected]
        assert [entry[3] for entry in entries] == payload["score_ids"]
