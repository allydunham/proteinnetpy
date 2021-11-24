"""
Tensorflow Datasets for working with ProteinNet data
"""
import tensorflow as tf

__all__ = ["proteinnet_tf_dataset"]

def _get_tf_tuple(input_tuple, func=tf.TensorShape):
    """
    Convert a generic nested tuple to a nested tuple of tensorflow objects.
    E.g. tf.TensorShapes or tf.Dtypes
    """
    output = []
    for i in input_tuple:
        if isinstance(i, tuple):
            output.append(_get_tf_tuple(i, func=func))
        else:
            output.append(func(i))
    return tuple(output)

def proteinnet_tf_dataset(pn_map, batch_size, prefetch=0,
                          shuffle_buffer=0, repeat=True):
    """
    Initiate a tf.data.Dataset from a ProteinNetMap and parameters. The function
    mapped accross pn_map must include an output_fields attribute describing which
    bits of data it outputs (options: wt, phi, psi, chi1)

    pn_map: ProteinNetMap object defining the dataset
    batch_size: Minibatch size
    prefetch: How many datapoints to prefect
    shuffle_buffer: Size of shuffle buffer
    repeat: Should dataset be endless
    """
    # fetch output shapes/types from pn_map ProteinNetMapFunction
    output_shapes = _get_tf_tuple(pn_map.func.output_shapes, tf.TensorShape)
    output_types = _get_tf_tuple(pn_map.func.output_types, tf.as_dtype)

    dataset = tf.data.Dataset.from_generator(pn_map.generate,
                                             output_types=output_types,
                                             output_shapes=output_shapes)

    if prefetch:
        dataset = dataset.prefetch(prefetch)

    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.padded_batch(batch_size, padded_shapes=output_shapes)

    return dataset
