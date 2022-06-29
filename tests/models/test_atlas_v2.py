import pytest
from jax import numpy as jnp, random
from models.atlas_v2 import LearnedAdjustment, LearnedElementaryStructure, AtlasV2
from tests import (
    are_sequences_of_points,
    mock_code_vectors,
    mock_sequence_of_points,
)


@pytest.mark.parametrize("seed,", [0, 2, 4])
@pytest.mark.parametrize("num_points", [2, 4, 8])
@pytest.mark.parametrize("embed_dim", [64, 128, 256])
@pytest.mark.parametrize("initial_dim", [2, 4, 8])
@pytest.mark.parametrize("final_dim", [2, 4, 8])
def test_learned_elementary_structure(
    seed, num_points, embed_dim, initial_dim, final_dim
):
    key = random.PRNGKey(seed)
    key, param_key = random.split(key)
    model = LearnedElementaryStructure(
        num_points=num_points,
        embed_dim=embed_dim,
        initial_dim=initial_dim,
        final_dim=final_dim,
    )

    params = model.init(param_key)
    outputs = model.apply(params)
    assert are_sequences_of_points(outputs, final_dim)
    assert outputs.shape[1] == num_points


@pytest.mark.parametrize("seed", [0, 2, 4])
@pytest.mark.parametrize("embed_dim", [64, 128, 256])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_learned_adjustment(seed, embed_dim, depth):
    key = random.PRNGKey(seed)
    key, param_key = random.split(key)
    model = LearnedAdjustment(embed_dim=embed_dim, depth=depth)

    contexts = mock_code_vectors(embed_dim, seed)
    points = mock_sequence_of_points(seed)
    params = model.init(param_key, points, contexts)
    outputs = model.apply(params, points, contexts)

    assert are_sequences_of_points(outputs)
    assert jnp.all(jnp.less_equal(outputs, 1.0))
    assert jnp.all(jnp.greater_equal(outputs, -1.0))


@pytest.mark.parametrize("seed", [0, 2, 4])
@pytest.mark.parametrize("num_points", [2, 4])
@pytest.mark.parametrize("final_dim", [2, 4])
@pytest.mark.parametrize("initial_dim", [4, 8])
@pytest.mark.parametrize("embed_dim_mlp", [128, 256])
@pytest.mark.parametrize("depth_mlp", [2, 3])
def test_atlas_v2(seed, num_points, final_dim, initial_dim, embed_dim_mlp, depth_mlp):
    key = random.PRNGKey(seed)
    key, param_key = random.split(key)
    model = AtlasV2(
        num_points=num_points,
        final_dim=final_dim,
        initial_dim=initial_dim,
        embed_dim_mlp=embed_dim_mlp,
        depth_mlp=depth_mlp,
    )

    contexts = mock_code_vectors(embed_dim_mlp, seed)
    params = model.init(param_key, contexts)

    outputs = model.apply(params, contexts)

    assert are_sequences_of_points(outputs, 2)
    assert outputs.shape[0] == contexts.shape[0]
    assert jnp.all(jnp.less_equal(outputs, 1.0))
    assert jnp.all(jnp.greater_equal(outputs, -1.0))
