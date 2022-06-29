from flax import linen as nn
from jax import numpy as jnp


class SimplifiedPointNet(nn.Module):
    internal_dim: int
    internal_depth: int
    out_dim: int

    @nn.compact
    def __call__(self, inputs):
        outputs = nn.Sequential(
            [
                nn.Sequential(
                    [nn.Dense(self.internal_dim * (self.internal_depth - i)), nn.relu]
                )
                for i in range(self.internal_depth)
            ]
            + [nn.LayerNorm(), nn.Dense(self.out_dim), nn.relu]
        )(inputs)

        outputs = jnp.max(outputs, axis=-2)  # is this what have caused it?
        # outputs = jnp.max(outputs, axis=2)  # is this what have caused it?
        return outputs
