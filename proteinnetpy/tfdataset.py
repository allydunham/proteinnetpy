"""
Tensorflow Datasets for working with ProteinNet data
"""
import tensorflow as tf

__all__ = ["proteinnet_tf_dataset"]

def _get_tf_tuple(input_tuple, func=tf.TensorShape):
    """
    Convert a generic nested tuple to a nested tuple of tensorflow objects. E.g. tf.TensorShapes or tf.Dtypes.

    Parameters
    ----------
    input_tuple : tuple
        Tuple of values to convert.
    func        : function
        Function converting to TensorFlow objects. Usually tf.TensorShape or tf.as_dtype.

    Returns
    -------
    tuple
        Parsed tupe of TensorShapes or DTypes.
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
    Initiate a TensorFlow Dataset from a ProteinNetMap.

    Create a TensorFlow Dataset from a ProteinNetMap, taking the output of the map and adding  batching, shuffling and repeating in addition to optimised loading for neural network training.
    The function mapped accross pn_map must include output_types and output_shapes attribute describing the data it returns (see data.LabeledFunction).

    Parameters
    ----------
    pn_map         : ProteinNetMap
        data.ProteinNetMap object to draw data from.
    batch_size     : int
        Minibatch size.
    prefetch       : int
        Number of elements to prefech into memory.
    shuffle_buffer : int
        Size of the Dataset shuffle buffer, which randomises the order of chunks of elements to reduce training bias.
    repeat         : bool
        Repeat the dataset once exhausted.

    Returns
    -------
    tensorflow.data.Dataset
        TensorFlow dataset containing the output of the specified ProteinNetMap.
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
