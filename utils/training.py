from data.data_loading import get_dataset, get_batches
from jax import numpy as jnp


class Dataset:
    def __init__(self, data_file: str or list, batch_size):
        if isinstance(data_file, str):
            self._dataset = get_dataset(data_file)
        if isinstance(data_file, list):
            datasets = []
            for name in data_file:
                datasets.append(get_dataset(name))

            self._dataset = datasets[0]
            for dataset in datasets[1:]:
                self._dataset = self._dataset.concatenate(dataset)

        self._batch_size = batch_size
        self.num_batches = sum([1 for batch in self.get_batches()])

    def get_batches(self):
        return get_batches(self._dataset, self._batch_size)


def group_points_into_patches(point_sequences, num_patches):
    return jnp.reshape(
        point_sequences,
        [
            point_sequences.shape[0],
            num_patches,
            point_sequences.shape[1] // num_patches,
            point_sequences.shape[2],
        ],
    )
