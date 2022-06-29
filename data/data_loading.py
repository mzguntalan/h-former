import tensorflow as tf


def _parse_example(example_proto):
    feature_description = {
        "font": tf.io.FixedLenFeature([], tf.string, default_value="")
    }

    parsed = tf.io.parse_example(example_proto, feature_description)
    tensor = tf.io.parse_tensor(parsed["font"], tf.float32)
    tensor = tf.reshape(tensor, (52, -1, 2))

    return {"font": tensor}


def get_dataset(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


def get_batches(dataset, BATCH_SIZE=32, BUFFER_SIZE=10_000):
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    batches = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    return batches
