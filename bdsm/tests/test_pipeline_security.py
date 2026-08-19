import pickle

import numpy as np
import pytest

from runtime.pipeline_security import kafka_ssl_config, safe_pickle_loads


def test_numpy_roundtrip():
    payload = {"uids": np.arange(5, dtype=np.int64), "x": np.ones((2, 3), dtype=np.float32)}
    out = safe_pickle_loads(pickle.dumps(payload, protocol=5))
    assert out["uids"].tolist() == [0, 1, 2, 3, 4]
    assert out["x"].shape == (2, 3)


def test_rejects_arbitrary_globals():
    class Evil:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned",))

    with pytest.raises(pickle.UnpicklingError):
        safe_pickle_loads(pickle.dumps(Evil()))


def test_tls_verification_always_on():
    assert kafka_ssl_config()["enable.ssl.certificate.verification"] is True
