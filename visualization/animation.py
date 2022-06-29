from typing import Callable, Any
from jax import random
import jax
from jax import numpy as jnp, jit
from visualization.colors import get_colors
import matplotlib.pyplot as plt
from functools import partial
from itertools import chain
from utils.training import group_points_into_patches
import numpy as np
from flax.core.frozen_dict import FrozenDict
import imageio
import os
from models.h_former import Encoder, Decoder
from tqdm import tqdm


def create_plot_for_font(font, filename, num_patches=1, dpi=32):
    colors = get_colors(num_patches)

    fig, axes = plt.subplots(
        8, 7, sharex="col", sharey="row", figsize=(20, 20), dpi=dpi
    )
    gid = -1
    for i in range(8):
        for j in range(7):
            gid += 1
            ax = axes[i, j]
            if gid < font.shape[0]:
                ax.set_ylim([-1.1, 1.1])
                ax.set_xlim([-1.1, 1.1])

                points = font[gid, :, :]
                # expose the patches
                points = jnp.reshape(
                    points, [num_patches, points.shape[0] // num_patches, 2]
                )

                for k in range(num_patches):
                    color = colors[k % len(colors)]

                    x = points[k, :, 0]
                    y = points[k, :, 1]

                    ax.scatter(x, y, color=color)

    if filename != "":
        plt.savefig(filename, dpi=dpi)
        plt.close()
    else:
        plt.show()


Array = Any


class AnimationForHFormer:
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        encoder_params: FrozenDict,
        decoder_params: FrozenDict,
        num_patches,
        seed: int = 0,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.num_patches = num_patches
        self.seed = seed

    def __call__(
        self,
        fonts: Array,
        num_in_between: int,
        directory: str = "./",
        filename_of_gif: str = "h-former",
        dpi=32,
    ):
        animation_set = self.get_animation_set_of_fonts(fonts, num_in_between)
        print("Creating Frames")
        for i, font in tqdm(enumerate(animation_set), total=animation_set.shape[0]):
            create_plot_for_font(
                font, directory + f"{filename_of_gif}-{i}.png", self.num_patches, dpi
            )
        gif_name = f"{directory + filename_of_gif}.gif"

        print("Finished creating Frames")
        print("Compiling to GIF")
        with imageio.get_writer(gif_name, mode="I") as writer:
            for i in tqdm(
                range(0, animation_set.shape[0]), total=animation_set.shape[0]
            ):
                target_name = f"{directory}{filename_of_gif}-{i}.png"
                image = imageio.imread(target_name)
                writer.append_data(image)
                os.remove(target_name)

        return gif_name

    @partial(jit, static_argnums=0)
    def encoder_forward(self, sequence_of_patches, glyph_ids):
        dropout_key, reparameterization_key = random.split(random.PRNGKey(self.seed))
        return self.encoder.apply(
            self.encoder_params,
            sequence_of_patches,
            glyph_ids,
            rngs={"dropout": dropout_key, "reparameterization": reparameterization_key},
        )

    @partial(jit, static_argnums=0)
    def decoder_forward(self, contexts, glyph_ids):
        return self.decoder.apply(self.decoder_params, contexts, glyph_ids)

    @partial(jit, static_argnums=0)
    def compute_mu_of_fonts(self, fonts, glyph_ids):
        # glyphs [N, 52, s, e]
        glyphs = jnp.reshape(fonts, [-1, fonts.shape[-2], fonts.shape[-1]])
        glyph_ids = jnp.reshape(glyph_ids, [-1])
        as_patches = group_points_into_patches(glyphs, self.num_patches)

        font_mus, logvar = self.encoder_forward(as_patches, glyph_ids)

        font_mus = jnp.reshape(font_mus, [fonts.shape[0], fonts.shape[1], -1])

        return font_mus

    @partial(jit, static_argnums=(0, 3))
    def create_transition_between_font_mus(self, font_mu_1, font_mu_2, num):
        # mu_1 mu_2 [52 e]
        return jnp.linspace(font_mu_1, font_mu_2, num)  # [num 52 e]

    @partial(jax.jit, static_argnums=(0, 2))
    def stretch_font_mu_sequence_with_transitions(self, font_mus, num_in_between):
        return jnp.concatenate(
            [
                self.create_transition_between_font_mus(
                    font_mu_1, font_mu_2, num_in_between
                )
                for font_mu_1, font_mu_2 in zip(
                    font_mus, chain(font_mus[1:], font_mus[0:1])
                )
            ],
            axis=0,
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def get_animation_set_of_fonts(self, fonts, num_in_between):
        font_glyph_ids = jnp.repeat(
            jnp.reshape(jnp.arange(0, 52, dtype="int32"), [1, -1]),
            fonts.shape[0],
            axis=0,
        )

        font_mus = self.compute_mu_of_fonts(fonts, font_glyph_ids)
        font_mus = self.stretch_font_mu_sequence_with_transitions(
            font_mus, num_in_between
        )
        glyph_ids = np.repeat(
            np.reshape(np.arange(0, fonts.shape[1]), [1, -1]),
            repeats=font_mus.shape[0],
            axis=0,
        )

        font_mus_reshaped = jnp.reshape(font_mus, [-1, font_mus.shape[-1]])
        glyph_ids_reshaped = np.reshape(glyph_ids, [font_mus_reshaped.shape[0]])
        reconstructions_flattened = self.decoder_forward(
            font_mus_reshaped,
            glyph_ids_reshaped,
        )

        reconstruction_per_font = jnp.reshape(
            reconstructions_flattened, [font_mus.shape[0], font_mus.shape[1], -1, 2]
        )

        return reconstruction_per_font
