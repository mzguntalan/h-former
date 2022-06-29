import pytest
from models.transformer import MultiHeadAttention, TransformerBlock, Transformer
from jax import numpy as jnp, random
from tests import mock_sequence_of_vectors, are_sequences_of_vectors


@pytest.mark.parametrize("seed", [0, 2, 4])
@pytest.mark.parametrize("num_heads,embed_dim", [(4, 128), (8, 256), (16, 256)])
def test_multi_head_attention(seed, num_heads, embed_dim):
    key = random.PRNGKey(seed)
    key, param_key, keys_key, queries_key, values_key = random.split(key, num=5)
    model = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)

    keys = mock_sequence_of_vectors(embed_dim, seed)
    queries = mock_sequence_of_vectors(embed_dim, seed)
    values = mock_sequence_of_vectors(embed_dim, seed)

    params = model.init(param_key, queries, keys, values)
    outputs = model.apply(params, queries, keys, values)

    assert are_sequences_of_vectors(outputs, embed_dim)


@pytest.mark.parametrize("seed", [0, 2, 4])
@pytest.mark.parametrize("num_heads,embed_dim", [(4, 128), (8, 256), (16, 256)])
@pytest.mark.parametrize("feedforward_dim", [128, 256])
@pytest.mark.parametrize("dropout_rate", [0.05, 0.5])
def test_transformer_block(seed, num_heads, embed_dim, feedforward_dim, dropout_rate):
    key = random.PRNGKey(seed)
    key, param_key, dropout_key = random.split(key, num=3)
    model = TransformerBlock(
        num_heads=num_heads,
        embed_dim=embed_dim,
        feedforward_dim=feedforward_dim,
        dropout_rate=dropout_rate,
    )

    vectors = mock_sequence_of_vectors(embed_dim, seed)
    params = model.init({"params": param_key, "dropout": dropout_key}, vectors)

    outputs = model.apply(params, vectors, rngs={"dropout": dropout_key})

    assert are_sequences_of_vectors(outputs, embed_dim)


@pytest.mark.parametrize("seed", [0, 2, 4])
@pytest.mark.parametrize("num_blocks", [2, 3])
@pytest.mark.parametrize("num_heads,embed_dim", [(4, 128), (8, 256), (16, 256)])
@pytest.mark.parametrize("feedforward_dim", [128, 256])
@pytest.mark.parametrize("dropout_rate", [0.05, 0.5])
def test_transformer(
    seed, num_blocks, num_heads, embed_dim, feedforward_dim, dropout_rate
):
    key = random.PRNGKey(seed)
    key, param_key, dropout_key = random.split(key, num=3)
    model = Transformer(
        num_blocks=num_blocks,
        num_heads=num_heads,
        embed_dim=embed_dim,
        feedforward_dim=feedforward_dim,
        dropout_rate=dropout_rate,
    )

    vectors = mock_sequence_of_vectors(embed_dim, seed)
    params = model.init({"params": param_key, "dropout": dropout_key}, vectors)

    outputs = model.apply(params, vectors, rngs={"dropout": dropout_key})
    assert are_sequences_of_vectors(outputs, embed_dim)
