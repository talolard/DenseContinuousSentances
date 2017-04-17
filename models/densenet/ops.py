'''
DenseNet for text with 1dconvolutions
'''
import tensorflow as tf
from tensorflow.contrib.layers import linear, fully_connected,layer_norm as ln
from arg_getter import FLAGS


def avg_pool1d(_input, k):
    output = tf.expand_dims(_input,axis=1)
    ksize = [1, 1, k, 1]
    strides = [1, 1, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(output, ksize, strides, padding)
    output = tf.squeeze(output,axis=1)
    return output


def composite_function(_input,outSize,width):
    with tf.variable_scope("composite_function"):
        output = conv1d(_input,outSize,width=width)
        output = ln(output)
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output,keep_prob=FLAGS.dropout_keep_prob)
    return output

def conv1d(x, outSize, width):
    inputSize = x.get_shape()[-1]
    filter_ = tf.get_variable("conv_filter", shape=[width, inputSize, outSize])
    convolved = tf.nn.conv1d(x,filters=filter_,stride=1,padding="SAME")
    return convolved
def bottleneck(_input, growthRate):
    '''
    Per the paper, each bottlneck outputs 4k feature size where k is the growth rate of the network.
    :return:
    '''
    outSize = growthRate * 4
    output = conv1d(_input,outSize,width=1)
    output = ln(output)
    output =tf.nn.relu(output)

    return output

def addInternalLayer(_input,growth_rate):

    compOut = composite_function(_input,growth_rate,width=3)
    output = tf.concat([_input,compOut],axis=2)
    output = bottleneck(output,growth_rate)
    return output

def makeBlock(_input,growth_rate,num_layers,bottle=True):
    output = _input
    for layer in range(num_layers):
        with tf.variable_scope("layer_{}".format(layer)):
            output = addInternalLayer(output,growth_rate)
    if bottle:
        output= bottleneck(output, growth_rate)
    return output

def transition_layer(_input,reduction=1):
    outSize = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(_input,outSize,width=1)
    output = avg_pool1d(output,2)
    return output

def transition_to_vector(_input):
    '''
    Transforms the last block into a single vector by avg_pooling
    '''
    last_pool_kernel = int(_input.get_shape()[-2])
    output = avg_pool1d(_input,last_pool_kernel)
    output =ln(output)
    output = tf.nn.relu(output)

    return output



