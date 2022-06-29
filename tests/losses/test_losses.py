import pytest
from losses import chamfer_distance, kl_div_loss
from tests import mock_sequence_of_points
from jax import numpy as jnp
import numpy as np


@pytest.mark.parametrize("seed", [0, 2, 4])
def test_chamfer_loss(seed):
    point_set_a = mock_sequence_of_points(seed)

    loss = chamfer_distance(point_set_a, point_set_a)
    assert jnp.allclose(loss, 0.0)

    loss = chamfer_distance(point_set_a, point_set_a + 1.0)
    assert not jnp.allclose(loss, 0.0)


@pytest.mark.parametrize("embed_dim", [128, 256])
def test_kl_div_loss(embed_dim):
    mu = jnp.zeros([8, embed_dim])
    logvar = jnp.zeros([8, embed_dim])

    loss = kl_div_loss(mu, logvar)
    assert jnp.allclose(loss, 0.0)

    mu = jnp.ones([8, embed_dim])
    logvar = 2 * jnp.ones([8, embed_dim])

    loss = kl_div_loss(mu, logvar)
    assert not jnp.allclose(loss, jnp.zeros(loss.shape))
