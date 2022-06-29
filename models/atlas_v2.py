from jax import numpy as jnp
from flax import linen as nn
import numpy as np


class LearnedElementaryStructure(nn.Module):
    num_points: int
    embed_dim: int
    initial_dim: int
    final_dim: int

    @nn.compact
    def __call__(self):
        indices = np.reshape(np.arange(self.num_points), [1, self.num_points])
        point_contexts = nn.Embed(self.num_points, self.initial_dim)(
            indices
        )  # [1, n, e]
        points = nn.Sequential(
            [
                nn.Dense(self.embed_dim),
                nn.relu,
                nn.Dense(self.embed_dim),
                nn.relu,
                nn.Dense(self.final_dim),
                nn.tanh,
            ]
        )(point_contexts)

        return points


class LearnedAdjustment(nn.Module):
    embed_dim: int
    depth: int

    @nn.compact
    def __call__(self, patch, context):
        # patch [N,s,2]
        # context [N,e]
        context = jnp.expand_dims(context, axis=1)
        context = jnp.repeat(context, patch.shape[1], axis=1)

        combined = jnp.concatenate([patch, context], axis=-1)
        adjusted_points = nn.Sequential(
            [
                nn.Sequential([nn.Dense(self.embed_dim), nn.relu])
                for i in range(self.depth)
            ]
            + [nn.Dense(2), nn.tanh]
        )(combined)

        return adjusted_points


class AtlasV2(nn.Module):
    num_points: int
    final_dim: int
    initial_dim: int
    embed_dim_mlp: int
    depth_mlp: int

    @nn.compact
    def __call__(self, contexts):
        patch = LearnedElementaryStructure(
            num_points=self.num_points,
            embed_dim=self.embed_dim_mlp,
            initial_dim=self.initial_dim,
            final_dim=self.final_dim,
        )()
        patch = jnp.repeat(patch, contexts.shape[0], axis=0)  # [N,s,e]
        points = LearnedAdjustment(embed_dim=self.embed_dim_mlp, depth=self.depth_mlp)(
            patch, contexts
        )
        return points
