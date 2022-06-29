from jax import numpy as jnp


def chamfer_distance(point_set_a, point_set_b):
    difference = jnp.subtract(
        jnp.expand_dims(point_set_a, axis=-2),
        jnp.expand_dims(point_set_b, axis=-3),
    )

    squared_distances = jnp.einsum("...i,...i->...", difference, difference)
    minimum_squared_distance_from_a_to_b = jnp.min(squared_distances, axis=-1)
    minimum_squared_distance_from_b_to_a = jnp.min(squared_distances, axis=-2)

    return jnp.mean(
        jnp.add(
            jnp.mean(minimum_squared_distance_from_a_to_b, axis=-1),
            jnp.mean(minimum_squared_distance_from_b_to_a, axis=-1),
        )
    )


def kl_div_loss(mu, logvar):
    return jnp.clip(
        jnp.mean(
            jnp.sum(jnp.exp(logvar) - (0.5 * logvar) - 1 + jnp.square(mu), axis=1)
        ),
        0.0,
        1e9,
    )
