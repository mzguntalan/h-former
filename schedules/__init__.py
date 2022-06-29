from jax import numpy as jnp


def common_schedule(hyper, warmup_steps=4000):
    def rsqrt(x):
        return jnp.reciprocal(jnp.sqrt(x))

    def schedule(step):
        arg1 = rsqrt(step)
        arg2 = jnp.multiply(step, jnp.power(warmup_steps, -1.5))

        return rsqrt(hyper) * jnp.minimum(arg1, arg2)

    return schedule
