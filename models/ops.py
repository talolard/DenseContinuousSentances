import tensorflow as tf
from arg_getter import FLAGS
from tensorflow.contrib.layers import linear, fully_connected,layer_norm as ln
def conv1d(x,outputsize,width):
    inputSize = x.get_shape()[-1]
    filter_ = tf.get_variable("conv_filter",shape=[width,inputSize,outputsize])
    bias = tf.get_variable("conv_bias",shape=[outputsize])
    convolved = tf.nn.conv1d(x,filters=filter_,stride=2,padding="SAME")
    convolved += bias
    convolved =ln(convolved)
    return convolved


def conv1d_transpose(x,targetHidden,width,growth_rate):
    length,inputHidden = x.get_shape()[-2:]
    outputShape = [FLAGS.batch_size,1,length.value*growth_rate,targetHidden]
    while len(x.get_shape() ) <4:
        x = tf.expand_dims(x, axis=1)
    bias = tf.get_variable("deconv_filter", shape=[targetHidden])
    filter_ = tf.get_variable("deconv_bias",shape=[1,width,targetHidden,inputHidden])
    conv_trans = tf.nn.conv2d_transpose(x,filter=filter_,output_shape=outputShape,strides=[1,1,growth_rate,1])
    conv_trans +=bias
    conv_trans =ln(conv_trans)
    conv_trans =tf.nn.relu(conv_trans)
    conv_trans = tf.squeeze(conv_trans,1)
    return conv_trans


def nonlin_ln(x):
    return ln(tf.nn.dropout(tf.nn.softsign(x),keep_prob=FLAGS.dropout_keep_prob))


