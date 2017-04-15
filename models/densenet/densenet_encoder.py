import tensorflow as tf
from models.densenet import ops
def DenseNetEncoder(_input, num_blocks, layers_per_batch, growth_rate):
    output = _input
    for block in range(num_blocks):
        with tf.variable_scope("block_{}".format(block)):
            output = ops.makeBlock(output,growth_rate=growth_rate,num_layers=layers_per_batch)
            if block < num_blocks -1:
                output = ops.transition_layer(output)
            else:
                output = ops.transition_to_vector(output)
    return output


