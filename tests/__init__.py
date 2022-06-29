from jax import numpy as jnp, random


def have_the_same_shape(tensor_1, tensor_2):
    return tensor_1.shape == tensor_2.shape


def have_shape(tensor, shape: tuple):
    return tensor.shape == shape


def is_a_sequence_of_points(tensor, last_dim: int = 2):
    return len(tensor.shape) == 2 and tensor.shape[-1] == last_dim


def are_sequences_of_points(tensor, last_dim: int = 2):
    return len(tensor.shape) == 3 and tensor.shape[-1] == last_dim


def are_sequences_of_vectors(tensor, last_dim: int):
    return len(tensor.shape) == 3 and tensor.shape[-1] == last_dim


def are_code_vectors(tensor, embed_dim: int):
    return len(tensor.shape) == 2 and tensor.shape[-1] == embed_dim


def are_vectors(tensor, embed_dim: int):
    return len(tensor.shape) == 2 and tensor.shape[-1] == embed_dim


def are_patches_of_points(tensor, num_patches):
    return (
        len(tensor.shape) == 4
        and tensor.shape[1] == num_patches
        and tensor.shape[-1] == 2
    )


def mock_sequence_of_points(seed: int = 0):
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    return random.normal(subkey, [8, 128, 2])


def mock_sequence_of_vectors(embed_dim: int, seed: int = 0):
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    return random.normal(subkey, [8, 128, embed_dim])


def mock_patches_of_points(seed: int = 0):
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    return random.normal(subkey, [8, 32, 128 // 32, 2])


def mock_code_vectors(embed_dim: int, seed: int = 0):
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    return random.normal(subkey, [8, embed_dim])
