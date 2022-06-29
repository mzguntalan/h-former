from jax import numpy as jnp
from jax import random
from flax import linen as nn
import numpy as np
from models.point_net import SimplifiedPointNet
from models.transformer import Transformer
from models.atlas_v2 import AtlasV2


class Encoder(nn.Module):
    embed_dim: int
    num_holder_vars: int
    depth_transformer: int
    num_heads_transformer: int
    num_glyphs: int

    @nn.compact
    def __call__(self, sequence_of_patches, glyph_ids):
        # sequence_of_patches [N,p,s//p,2]
        # glyph_ids [N]

        patch_contexts = SimplifiedPointNet(
            internal_dim=self.embed_dim, internal_depth=3, out_dim=self.embed_dim
        )(
            sequence_of_patches
        )  # [N,p,e]
        patch_contexts = nn.Sequential(
            [
                nn.Dense(self.embed_dim),
                nn.relu,
                nn.Dense(self.embed_dim),
                nn.relu,
            ]
        )(patch_contexts)

        glyph_contexts = nn.Embed(self.num_glyphs, self.embed_dim)(glyph_ids)  # [N,e]
        glyph_contexts = nn.Sequential(
            [
                nn.Dense(self.embed_dim),
                nn.relu,
                nn.Dense(self.embed_dim),
                nn.relu,
            ]
        )(glyph_contexts)
        glyph_contexts = jnp.expand_dims(glyph_contexts, axis=1)

        holder_indices = np.reshape(np.arange(0, self.num_holder_vars), [-1, 1])
        holder_contexts = nn.Sequential(
            [
                nn.Dense(self.embed_dim),
                nn.relu,
                nn.Dense(self.embed_dim),
                nn.relu,
            ]
        )(
            holder_indices
        )  # [h,e]
        holder_contexts = jnp.expand_dims(holder_contexts, axis=0)
        holder_contexts = jnp.repeat(
            holder_contexts, patch_contexts.shape[0], axis=0
        )  # [N,h,e]

        encoding_contexts = jnp.concatenate(
            [holder_contexts, glyph_contexts, patch_contexts], axis=1
        )
        # encoding_contexts [N, h+1+s, e]

        encoding_contexts = Transformer(
            num_blocks=self.depth_transformer,
            num_heads=self.num_heads_transformer,
            embed_dim=self.embed_dim,
            feedforward_dim=self.embed_dim * 2,
            dropout_rate=0.05,
        )(encoding_contexts)

        code = encoding_contexts[:, 0, :]
        code = nn.Sequential(
            [
                nn.Dense(self.embed_dim),
                nn.relu,
                nn.LayerNorm(),
                nn.Dense(self.embed_dim),
                nn.relu,
            ]
        )(code)

        # VAE
        mu = nn.Dense(self.embed_dim)(code)
        logvar = nn.Dense(self.embed_dim)(code)

        return mu, logvar


def reparametrize(key, mu, logvar):
    eps = random.normal(key, mu.shape)
    return jnp.add(mu, jnp.multiply(eps, jnp.exp(jnp.divide(logvar, 2))))


class Decoder(nn.Module):
    embed_dim: int
    num_patches: int
    num_glyphs: int
    num_points: int

    @nn.compact
    def __call__(self, codes, glyph_ids):
        glyph_contexts = nn.Embed(self.num_glyphs, self.embed_dim)(glyph_ids)
        glyph_contexts = nn.Sequential(
            [
                nn.Dense(self.embed_dim),
                nn.relu,
                nn.Dense(self.embed_dim),
                nn.relu,
            ]
        )(glyph_contexts)

        contexts = jnp.concatenate([codes, glyph_contexts], axis=1)  # [N, e1+e2]
        contexts = nn.Sequential([nn.Dense(self.embed_dim), nn.relu, nn.LayerNorm()])(
            contexts
        )

        points = jnp.concatenate(
            [
                AtlasV2(
                    num_points=self.num_points // self.num_patches,
                    final_dim=2,
                    initial_dim=8,
                    embed_dim_mlp=self.embed_dim,
                    depth_mlp=3,
                )(contexts)
                for i in range(self.num_patches)
            ],
            axis=1,
        )

        return points


class HFormer(nn.Module):
    embed_dim: int
    num_holder_vars: int
    depth_transformer: int
    num_heads_transformer: int
    num_patches: int
    num_glyphs: int
    num_points: int
    reparameterization_trick: bool

    def setup(self):
        self.encoder = Encoder(
            embed_dim=self.embed_dim,
            num_holder_vars=self.num_holder_vars,
            depth_transformer=self.depth_transformer,
            num_heads_transformer=self.num_heads_transformer,
            num_glyphs=self.num_glyphs,
        )
        self.resample_mu_logvar = (
            (lambda key, mu, logvar: mu)
            if not self.reparameterization_trick
            else reparametrize
        )
        self.decoder = Decoder(
            embed_dim=self.embed_dim,
            num_patches=self.num_patches,
            num_glyphs=self.num_glyphs,
            num_points=self.num_points,
        )

    def __call__(self, sequence_of_patches, glyph_ids):
        mu, logvar = self.encoder(sequence_of_patches, glyph_ids)
        reparam_key = self.make_rng("reparameterization")
        codes = self.resample_mu_logvar(reparam_key, mu, logvar)

        points = self.decoder(codes, glyph_ids)

        return mu, logvar, points
