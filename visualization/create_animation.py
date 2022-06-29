import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from models.h_former import Encoder, Decoder, HFormer
from config import Config
from visualization.animation import AnimationForHFormer
from flax import serialization
from jax import random
from training.train import load_dataset, create_dummy_patches, create_dummy_glyph_ids
from jax import numpy as jnp
import jax.dlpack
import tensorflow as tf


def main():
    key = random.PRNGKey(Config.seed)
    key, param_key, dropout_key, reparametrization_key, dummy_init_key = random.split(
        key, num=5
    )
    encoder = Encoder(
        embed_dim=Config.embed_dim,
        num_holder_vars=Config.num_holder_vars,
        depth_transformer=Config.depth_transformer,
        num_heads_transformer=Config.num_heads_transformer,
        num_glyphs=Config.num_glyphs,
    )

    decoder = Decoder(
        embed_dim=Config.embed_dim,
        num_patches=Config.num_patches,
        num_glyphs=Config.num_glyphs,
        num_points=Config.num_points,
    )

    h_former = HFormer(
        embed_dim=Config.embed_dim,
        num_holder_vars=Config.num_holder_vars,
        depth_transformer=Config.depth_transformer,
        num_heads_transformer=Config.num_heads_transformer,
        num_patches=Config.num_patches,
        num_glyphs=Config.num_glyphs,
        num_points=Config.num_points,
        reparameterization_trick=False,
    )

    dummy_patches = create_dummy_patches(
        dummy_init_key,
        Config.num_fonts_per_batch,
        Config.num_glyphs,
        Config.num_patches,
        Config.num_points,
    )

    dummy_glyph_ids = create_dummy_glyph_ids(
        Config.num_fonts_per_batch, Config.num_glyphs
    )

    params = h_former.init(
        {
            "params": param_key,
            "dropout": dropout_key,
            "reparameterization": reparametrization_key,
        },
        dummy_patches,
        dummy_glyph_ids,
    )

    params = load_h_former_params_from_bytes(
        params, f"{Config.model_weights_directory}{Config.model_weights_filename}"
    )
    encoder_params = get_encoder_params(params)
    decoder_params = get_decoder_params(params)

    print("Loaded encoder and decoder parameters")

    animation = AnimationForHFormer(
        encoder,
        decoder,
        encoder_params,
        decoder_params,
        Config.num_patches,
        Config.seed,
    )

    fonts = get_fonts(Config.dataset_filename, Config.animation_num_fonts)
    fonts = jax.device_put(fonts, jax.devices()[0])

    print("Loaded fonts")
    print("Starting animation generation")
    gif_name = animation(
        fonts,
        Config.animation_num_betweens,
        Config.animation_directory,
        Config.animation_name,
        Config.animation_dpi,
    )
    print(f"Finished animation. Saved {gif_name}")


def get_fonts(dataset_tfrecords_filename, num_fonts):
    tf.random.set_seed(Config.seed)
    dataset = load_dataset(num_fonts, dataset_tfrecords_filename)

    for batch in dataset.get_batches():
        batch = batch["font"]
        break

    batch = tf.experimental.dlpack.to_dlpack(batch)
    batch = jax.dlpack.from_dlpack(batch)

    return batch


def load_h_former_params_from_bytes(
    params_tempplate: FrozenDict, filename: str
) -> FrozenDict:
    with open(filename, "rb") as f:
        params = serialization.from_bytes(params_tempplate, f.read())
    return params


def get_encoder_params(params: FrozenDict) -> FrozenDict:
    return FrozenDict({"params": params["params"]["encoder"]})


def get_decoder_params(params: FrozenDict) -> FrozenDict:
    return FrozenDict({"params": params["params"]["decoder"]})


if __name__ == "__main__":
    main()
