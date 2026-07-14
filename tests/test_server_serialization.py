import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from thin_tts.server import _request_for, _stream_audio


class RecordingPipeline:
    def __init__(self):
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(self, _request):
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            yield 32000, np.zeros(16, dtype=np.int16)
        finally:
            with self._state_lock:
                self.active -= 1


def test_streams_serialize_access_to_shared_pipeline():
    pipeline = RecordingPipeline()
    start = threading.Barrier(3)

    def consume():
        start.wait()
        list(_stream_audio(pipeline, {}, 32000))

    workers = [threading.Thread(target=consume) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers)
    assert pipeline.max_active == 1


def test_stream_releases_pipeline_lock_after_generator_error():
    class FailingPipeline:
        def run(self, _request):
            raise RuntimeError("inference failed")
            yield

    failed_stream = _stream_audio(FailingPipeline(), {}, 32000)
    next(failed_stream)  # WAV header is emitted while the lock is held.
    with pytest.raises(RuntimeError, match="inference failed"):
        next(failed_stream)

    # A subsequent stream can acquire the lock instead of deadlocking.
    chunks = list(_stream_audio(RecordingPipeline(), {}, 32000))
    assert len(chunks) == 2


def test_request_can_opt_into_rng_isolation():
    cfg = SimpleNamespace(ref_audio="reference.wav", ref_text="reference text")

    request = _request_for(
        "测试文本。",
        cfg,
        mode=4,
        seed=314159,
        rng_isolation=True,
    )

    assert request["rng_isolation"] is True
