from jax import numpy as jnp
from flax import linen as nn


class MultiHeadAttention(nn.Module):
    num_heads: int
    embed_dim: int

    @nn.compact
    def __call__(self, queries, keys, values):
        queries = nn.Dense(self.embed_dim, use_bias=False)(queries)
        keys = nn.Dense(self.embed_dim, use_bias=False)(queries)
        values = nn.Dense(self.embed_dim, use_bias=False)(queries)

        queries = jnp.reshape(
            queries, (queries.shape[0], queries.shape[1], self.num_heads, -1)
        )
        keys = jnp.reshape(keys, (keys.shape[0], keys.shape[1], self.num_heads, -1))
        values = jnp.reshape(
            values, (values.shape[0], values.shape[1], self.num_heads, -1)
        )

        similarity = jnp.einsum("nqhe, nkhe->nhqk", queries, keys)

        attention = nn.softmax(similarity / jnp.sqrt(keys.shape[-1]))
        weightedvalues = jnp.einsum("nhqk, nkhd->nqhd", attention, values)
        weightedvalues = jnp.reshape(
            weightedvalues, (values.shape[0], values.shape[1], -1)
        )

        return nn.Dense(self.embed_dim)(weightedvalues)


class TransformerBlock(nn.Module):
    num_heads: int
    embed_dim: int
    feedforward_dim: int
    dropout_rate: int

    @nn.compact
    def __call__(self, sequence_of_vectors):
        outputs = nn.LayerNorm()(sequence_of_vectors)
        outputs = MultiHeadAttention(
            num_heads=self.num_heads, embed_dim=self.embed_dim
        )(outputs, outputs, outputs)

        dropout_rng = self.make_rng("dropout")
        outputs = nn.Dropout(self.dropout_rate, deterministic=False)(outputs)
        outputs = nn.LayerNorm()(outputs)
        outputs = nn.Sequential(
            [nn.Dense(self.feedforward_dim), nn.relu, nn.Dense(self.embed_dim)]
        )(outputs)

        return outputs


class Transformer(nn.Module):
    num_blocks: int
    num_heads: int
    embed_dim: int
    feedforward_dim: int
    dropout_rate: int

    @nn.compact
    def __call__(self, sequence_of_vectors):
        outputs = sequence_of_vectors
        outputs = nn.Sequential(
            [
                TransformerBlock(
                    num_heads=self.num_heads,
                    embed_dim=self.embed_dim,
                    feedforward_dim=self.feedforward_dim,
                    dropout_rate=self.dropout_rate,
                )
                for i in range(self.num_blocks)
            ]
        )(outputs)
        return outputs
