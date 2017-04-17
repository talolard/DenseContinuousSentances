'''
DenseNet for text with 1dconvolutions
'''
import tensorflow as tf
from tensorflow.contrib.layers import linear, fully_connected,layer_norm as ln
# from arg_getter import FLAGS

"""as a note, layer normalization usually occurs right after the weight matrix is applied and BEFORE ANY ACTIVATIONS
"""

def avg_pool1d(_input, k):
    output = tf.expand_dims(_input,axis=1)
    ksize = [1, 1, k, 1]
    strides = [1, 1, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(output, ksize, strides, padding)
    output = tf.squeeze(output,axis=1)
    return output


def _conv1d(x, outSize, width):
    inputSize = x.get_shape()[-1]
    filter_ = tf.get_variable("conv_filter", shape=[width, inputSize, outSize])
    convolved = tf.nn.conv1d(x,filters=filter_,stride=1,padding="SAME")
    return convolved

def composite_function(_input,outSize,width):
    with tf.variable_scope("composite_function"):
        output = _conv1d(_input, outSize, width=width)
        output = ln(output)
        output = tf.nn.relu(output)

    return output

def bottleneck(_input, growthRate):
    '''
    Per the paper, each bottlneck outputs 4k feature size where k is the growth rate of the network.
    :return:
    '''

    outSize = growthRate * 4
    output = _conv1d(_input,outSize,width=1)
    output = ln(output)
    output = tf.nn.relu(output)

    return output


def addInternalLayer(_input, growth_rate, bottle=False):
    # notice that there is a 3x3 convolution at the beginning
    compOut = composite_function(_input, growth_rate, width=3)
    
    # Concatenation -- think of this as a skip connection
    output = tf.concat([_input,compOut],axis=2) #axis two is correct
    
    # This is a 1x1 convolution to follow up -- we can optionally turn this off
    if bottle:
        output = bottleneck(output,growth_rate) 
    return output

def makeBlock(_input,growth_rate,num_layers,bottle=False):
    """Bottle will significantly reduce the amount of computation needed."""
    output = _input
    for layer in range(num_layers):
        with tf.variable_scope("layer_{}".format(layer)):
            output = addInternalLayer(output,growth_rate,bottle=bottle)
            print('output for internal layer', layer, output)
    if bottle:
        output= bottleneck(output, growth_rate)
    return output

def transition_layer(_input,reduction=1):
    out_size = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(_input,out_size,width=1)
    output = avg_pool1d(output,2)
    return output

def transition_to_vector(_input):
    '''
    Transforms the last block into a single vector by avg_pooling
    '''
    output =ln(_input)
    output = tf.nn.relu(output)
    last_pool_kernel = int(output.get_shape()[-2])
    output = avg_pool1d(output,last_pool_kernel)
    output = tf.squeeze(output,axis=1)
    return output



