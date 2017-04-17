import tensorflow as tf
from models.densenet import ops
from models.ops import conv1d_transpose
from arg_getter import FLAGS
def DenseNetDecoder(_input, layers_per_batch, growth_rate,expansion_rate,final_width=FLAGS.max_len):
    output = _input
    block =0
    while output.get_shape().as_list()[1] <final_width:
        with tf.variable_scope("decode_block_{}".format(block)):
            output = conv1d_transpose(output, FLAGS.hidden1, 3, expansion_rate)
            output = ops.makeBlock(output,growth_rate=growth_rate,num_layers=layers_per_batch)

            block+=1
    return output


