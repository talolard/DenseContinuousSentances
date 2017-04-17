import tensorflow as tf
from tensorflow.contrib.layers import linear
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer, xavier_initializer_conv2d
from tensorflow.contrib.layers.python.layers.regularizers import l1_regularizer
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import GRUCell

from data.preprocess import DataLoader
from models.densenet.densenet_decoder import DenseNetDecoder
from models.ops import conv1d ,conv1d_transpose,nonlin_ln
from arg_getter import FLAGS
import numpy as np
xavier_initializer_conv2d
from utils import print_paramater_count
from models.densenet.densenet_encoder import DenseNetEncoder
from models.densenet  import ops as densenet_ops
class BaselineConvEncDec():
    def __init__(self,):

        self.input = tf.placeholder_with_default(tf.ones(shape=[FLAGS.batch_size,FLAGS.max_len],dtype=tf.int32),shape=[FLAGS.batch_size,FLAGS.max_len],)
        embedded = self.embed_sentances(self.input)
        encoded = DenseNetEncoder(_input=embedded, growth_rate=4, num_blocks=3, layers_per_batch=5)
        decoded = DenseNetDecoder(encoded,layers_per_batch=3,growth_rate=16,expansion_rate=2)
        logits = self.to_logits(decoded)
        self.preds_op = self.preds(logits)
        self.loss_op = self.loss(self.input,logits)
        reg_loss =tf.reduce_sum(tf.losses.get_regularization_losses())
        tf.summary.scalar("loss",self.loss_op)
        tf.summary.scalar("reg_loss", reg_loss)
        self.gs =tf.contrib.framework.get_or_create_global_step()
        self.train_op = self.train(self.loss_op+reg_loss,self.gs)
        self.make_gradient_summaries(self.loss_op+reg_loss)
        self.summaries = tf.summary.merge_all()


    @staticmethod
    def embed_sentances(s1):
        with tf.device("/cpu:0"):
            embedding_size = 4
            embedding_matrix = tf.get_variable("embedding_matrix", shape=[FLAGS.vocab_size, embedding_size],
                                               dtype=tf.float32)
            s1_embeded = tf.nn.embedding_lookup(embedding_matrix, s1)
        return s1_embeded
    def preds(self,logits):
        probs = tf.nn.softmax(logits)
        return tf.arg_max(probs,dimension=2)
    def encoder(self,sentance):
        next_input = sentance
        layer_num =0
        while next_input.get_shape()[1] > 1:
            with tf.variable_scope("conv_{}".format(layer_num),initializer=xavier_initializer()):
                next_input = nonlin_ln(conv1d(next_input,FLAGS.hidden1,width=3))
                tf.summary.histogram("conv_act_{}".format(layer_num),next_input)
            layer_num+=1
        encoded_sentance =next_input
        return encoded_sentance

    def rnn_decoder(self,encoded,embedded_input):
        encoded = tf.squeeze(encoded,axis=1)
        encoded = tf.contrib.layers.fully_connected(encoded,num_outputs=FLAGS.hidden1)
        cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.hidden1)
        state = encoded
        next_input = cell.zero_state(FLAGS.batch_size,dtype=tf.float32)
        outputs =[]

        with tf.variable_scope("rnn_decoder") as scope:
            for step in range(FLAGS.max_len):
                if step >0:
                    scope.reuse_variables()
                next_input,state = cell(next_input,state)
                outputs.append(next_input)
        outputs = tf.stack(outputs,axis=1)
        return outputs
    def decoder(self,encoded_sentance):

        decoded = DenseNetDecoder(encoded_sentance,3,4)
        return decoded
    def encoded_to_latent(self,encoded):
        hidden = tf.contrib.layers.fully_connected(encoded,num_outputs=FLAGS.hidden2*2,activation_fn=tf.nn.softplus)
        mean =  tf.contrib.layers.fully_connected(hidden,num_outputs=FLAGS.hidden2,activation_fn=None)
        std =   tf.contrib.layers.fully_connected(hidden,num_outputs=FLAGS.hidden2,activation_fn=None)
        eps = tf.random_normal((FLAGS.batch_size, FLAGS.hidden2), 0, 1, dtype=tf.float32)  # Adding a random number
        z = tf.add(mean, tf.mul(tf.sqrt(tf.exp(std)), eps))  # The sampled z
        return z

    def to_logits(self,decoded):
        with tf.variable_scope("logits",):
            return linear(decoded,num_outputs=FLAGS.vocab_size,)
    def loss(self,targets,logits):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)
        return tf.reduce_mean(loss)
    def train(self,loss,gs):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gvs = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gvs,gs)

        return train_op

    @staticmethod
    def make_gradient_summaries(loss):
        with tf.name_scope("gradients"):
            grads = tf.gradients(loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            grad_summaries = []
            for grad, var in grads:
                if not "LayerNorm" in var.name and not "layer_weight" in var.name:
                    grad_summaries.append(tf.summary.histogram(var.name + '/gradient', grad))
            return grad_summaries

