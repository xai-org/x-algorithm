import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("pyarrow")
import jax.numpy as jnp

import heads
from task_heads import head_logits, init_head_params, load_head_params
from train_head import save_checkpoint


@pytest.mark.parametrize("hidden_dims", [(), (16,), (512, 256)])
def test_training_checkpoint_roundtrip_for_supported_head_layouts(tmp_path, hidden_dims):
    params = init_head_params(4, hidden_dims)
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    save_checkpoint(str(tmp_path), 1, params, {"head_hidden_dims": list(hidden_dims)})

    loaded = load_head_params(str(tmp_path / "step_000001"))
    assert set(loaded) == set(params)
    np.testing.assert_allclose(head_logits(loaded, x), head_logits(params, x))
    assert head_logits(loaded, x).shape == (1, heads.N_HEADS)


def test_load_head_params_rejects_incomplete_layer_pairs(tmp_path):
    np.savez(tmp_path / "head_params.npz", **{"abuse_v3/head_mlp/w0": np.zeros((4, 8))})
    with pytest.raises(RuntimeError, match="unexpected head param layout"):
        load_head_params(str(tmp_path))
