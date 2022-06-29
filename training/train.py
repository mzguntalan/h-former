from typing import Any, Tuple
import jax
import jax.dlpack
from jax import numpy as jnp, random
import numpy as np
import optax
import tensorflow.keras as K
from flax import serialization
from utils.training import Dataset
import tensorflow as tf
from models.h_former import HFormer
from config import Config
from utils.training import group_points_into_patches
from schedules import common_schedule
from training import Training

Array = Any
Parameters = Any
OptimizerState = Any


def main():
    tf.random.set_seed(Config.seed)
    key = random.PRNGKey(seed=Config.seed)

    dataset = load_dataset(Config.num_fonts_per_batch, Config.dataset_filename)
    num_batches = dataset.num_batches
    print(f"Loaded Dataset with {num_batches} batches per epoch")

    model = HFormer(
        embed_dim=Config.embed_dim,
        num_holder_vars=Config.num_holder_vars,
        depth_transformer=Config.depth_transformer,
        num_heads_transformer=Config.num_heads_transformer,
        num_patches=Config.num_patches,
        num_glyphs=Config.num_glyphs,
        num_points=Config.num_points,
        reparameterization_trick=True,
    )

    key, params_key, dropout_key, reparametrization_key, dummy_init_key = random.split(
        key, num=5
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

    params = model.init(
        {
            "params": params_key,
            "dropout": dropout_key,
            "reparameterization": reparametrization_key,
        },
        dummy_patches,
        dummy_glyph_ids,
    )
    print("Initialized H-Former")

    optimizer = optax.chain(
        optax.adam(
            common_schedule(
                Config.optimizer_common_schedule_hyper,
                Config.optimizer_common_schedule_warmup_steps,
            )
        ),
        optax.clip_by_global_norm(1.0),
    )
    opt_state = optimizer.init(params)
    print("Initialized Optimizer")

    training = Training(
        model=model, optimizer=optimizer, max_epochs=Config.train_num_epochs
    )
    print("Initialized Training Loop")

    for epoch in range(0, Config.train_num_epochs):
        progbar = K.utils.Progbar(
            num_batches, stateful_metrics=["epoch", "main", "kl", "adj"]
        )
        progbar.update(0, [("epoch", epoch), ("main", 0), ("kl", 0), ("adj", 0)])

        for batch in dataset.get_batches():
            glyphs = batch["font"]
            key, dropout_key, reparametrization_key = random.split(key, num=3)
            main_loss, kl, adj, params, opt_state = train_on_a_batch(
                training, glyphs, params, opt_state, dropout_key, reparametrization_key
            )

            progbar.add(1, [("main", main_loss), ("kl", kl), ("adj", adj)])

    print("Training complete")

    filename = save_model_params_as_bytes(
        params, f"{Config.model_weights_directory}{Config.model_weights_filename}"
    )
    print(f"Model weights saved at {filename}")


def train_on_a_batch(
    training: Training, glyphs, params, opt_state, dropout_key, reparametrization_key
) -> Tuple[float, float, float, Parameters, OptimizerState]:
    glyphs = tf.experimental.dlpack.to_dlpack(glyphs)
    glyphs = jax.dlpack.from_dlpack(glyphs)

    glyphs = jnp.reshape(glyphs, [-1, Config.num_points, 2])
    as_patches = group_points_into_patches(glyphs, Config.num_patches)

    font_glyph_ids = jnp.repeat(
        jnp.reshape(jnp.arange(0, 52, dtype="int32"), [1, -1]),
        Config.num_fonts_per_batch,
        axis=0,
    )
    glyph_ids = jnp.reshape(font_glyph_ids, [-1])

    # put to gpu
    glyphs = jax.device_put(glyphs, jax.devices()[0])
    as_patches = jax.device_put(as_patches, jax.devices()[0])
    glyph_ids = jax.device_put(glyph_ids, jax.devices()[0])

    main_loss, kl, adj, params, opt_state = training.train_step(
        params,
        dropout_key,
        reparametrization_key,
        as_patches,
        glyph_ids,
        glyphs,
        opt_state,
    )

    return main_loss, kl, adj, params, opt_state


def create_dummy_patches(
    key: random.PRNGKey,
    num_fonts_per_batch: int,
    num_glyphs: int,
    num_patches: int,
    num_points: int,
) -> Array:
    return random.uniform(
        key,
        [
            num_fonts_per_batch * num_glyphs,
            num_patches,
            num_points // num_patches,
            2,
        ],
    )


def create_dummy_points(key, num_fonts_per_batch, num_glyphs, num_points) -> Array:
    return random.uniform(
        key,
        [num_fonts_per_batch * num_glyphs, num_points, 2],
    )


def create_dummy_glyph_ids(num_fonts_per_batch, num_glyphs) -> Array:
    return np.ones([num_fonts_per_batch * num_glyphs], dtype="int32")


def load_dataset(num_fonts_per_batch, dataset_tfrecords_filename) -> Dataset:
    dataset = Dataset(dataset_tfrecords_filename, num_fonts_per_batch)
    return dataset


def save_model_params_as_bytes(params, filename) -> str:
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(params))
    return filename


if __name__ == "__main__":
    main()
