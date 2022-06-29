import pytest
from jax import numpy as jnp, random
from models.point_net import SimplifiedPointNet
from tests import mock_sequence_of_points, are_code_vectors


@pytest.mark.parametrize("seed,", [0, 2, 4])
@pytest.mark.parametrize("internal_dim,", [64, 128, 256])
@pytest.mark.parametrize("internal_depth,", [3, 2, 1])
@pytest.mark.parametrize("out_dim,", [64, 128, 256])
def test_simplified_point_net(seed, internal_dim, internal_depth, out_dim):
    key = random.PRNGKey(seed)
    key, param_key = random.split(key)
    model = SimplifiedPointNet(
        internal_dim=internal_dim,
        internal_depth=internal_depth,
        out_dim=out_dim,
    )

    points = mock_sequence_of_points(seed)
    params = model.init(param_key, points)
    outputs = model.apply(params, points)
    assert are_code_vectors(outputs, out_dim)
