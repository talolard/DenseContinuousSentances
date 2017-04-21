import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer

from models.densenet import ops


def DenseNetEncoder(_input, num_blocks, layers_per_batch, growth_rate, bottle=False):
    """Bottle will significantly reduce the amount of computation needed."""
    tf.logging.info(
        'Building DenseNet Encoder with {} blocks and {} layers per block'.format(num_blocks, layers_per_batch))
    output = _input
    for block in range(num_blocks):
        with tf.variable_scope("block_{}".format(block),initializer=xavier_initializer()):
            output = ops.makeBlock(output, growth_rate=growth_rate, num_layers=layers_per_batch, bottle=bottle)
            print('output after block', block, output)
            if block < num_blocks - 1:
                output = ops.transition_layer(output)
            else:
                output = ops.transition_to_vector(output)

    #output = tf.squeeze(output)
    return output
