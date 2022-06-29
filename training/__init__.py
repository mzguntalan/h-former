from typing import Any
import optax
from flax import linen as nn
from jax import numpy as jnp, random, jit
import jax
from models.h_former import HFormer
from functools import partial
from losses import chamfer_distance, kl_div_loss

Optimizer = Any


class Training:
    def __init__(self, model: nn.Module, optimizer: Optimizer, max_epochs: int):
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self._epochs_past = 0

    @property
    def epochs_past(self):
        return self._epochs_past

    @partial(jit, static_argnums=(0))
    def forward(
        self,
        params,
        dropout_key,
        reparameterization_key,
        sequence_of_patches,
        glyph_ids,
    ):
        return self.model.apply(
            params,
            sequence_of_patches,
            glyph_ids,
            rngs={"dropout": dropout_key, "reparameterization": reparameterization_key},
        )

    @partial(jit, static_argnums=0)
    def individual_losses(
        self,
        params,
        dropout_key,
        reparameterization_key,
        sequence_of_patches,
        glyph_ids,
        expected,
    ):
        mu, logvar, reconstructions = self.forward(
            params, dropout_key, reparameterization_key, sequence_of_patches, glyph_ids
        )

        main = chamfer_distance(reconstructions, expected)
        kl = jnp.clip(kl_div_loss(mu, logvar), 1e-5, 1e9)
        adj = 0.0

        for w in jax.tree_leaves(params["params"]["decoder"]["AtlasV2_0"]):
            adj += jnp.mean(jnp.abs(w) + jnp.square(w))
        for w in jax.tree_leaves(params["params"]["decoder"]["AtlasV2_1"]):
            adj += jnp.mean(jnp.abs(w) + jnp.square(w))
        for w in jax.tree_leaves(params["params"]["decoder"]["AtlasV2_2"]):
            adj += jnp.mean(jnp.abs(w) + jnp.square(w))

        return main, kl, adj

    @partial(jit, static_argnums=0)
    def loss_fn(
        self,
        params,
        dropout_key,
        reparameterization_key,
        sequence_of_patches,
        glyph_ids,
        expected,
    ):
        main, kl, adj = self.individual_losses(
            params,
            dropout_key,
            reparameterization_key,
            sequence_of_patches,
            glyph_ids,
            expected,
        )
        loss = 1000 * main + kl + adj
        return loss

    @partial(jit, static_argnums=0)
    def grad_loss_fn(
        self,
        params,
        dropout_key,
        reparameterization_key,
        sequence_of_patches,
        glyph_ids,
        expected,
    ):
        return jax.grad(self.loss_fn)(
            params,
            dropout_key,
            reparameterization_key,
            sequence_of_patches,
            glyph_ids,
            expected,
        )

    @partial(jit, static_argnums=0)
    def optimizer_update(self, grads, opt_state):
        return self.optimizer.update(grads, opt_state)

    @partial(jit, static_argnums=0)
    def apply_updates(self, params, updates):
        return optax.apply_updates(params, updates)

    def train_step(
        self,
        params,
        dropout_key,
        reparameterization_key,
        sequence_of_patches,
        glyph_ids,
        expected_outputs,
        opt_state,
    ):
        self._epochs_past += 1
        main, kl, adj = self.individual_losses(
            params,
            dropout_key,
            reparameterization_key,
            sequence_of_patches,
            glyph_ids,
            expected_outputs,
        )
        grads = self.grad_loss_fn(
            params,
            dropout_key,
            reparameterization_key,
            sequence_of_patches,
            glyph_ids,
            expected_outputs,
        )

        updates, opt_state = self.optimizer_update(grads, opt_state)
        new_params = self.apply_updates(params, updates)

        return main, kl, adj, new_params, opt_state
