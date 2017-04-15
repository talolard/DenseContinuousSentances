import tensorflow as tf
from tensorflow.contrib.layers import linear
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer, xavier_initializer_conv2d

from data.preprocess import DataLoader
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
        #encoded = self.encoder(embedded)
        decoded = self.decoder(encoded)
        logits = self.to_logits(decoded)
        self.preds_op = self.preds(logits)
        self.loss_op = self.loss(self.input,logits)
        tf.summary.scalar("loss",self.loss_op)
        self.gs =tf.contrib.framework.get_or_create_global_step()
        self.train_op = self.train(self.loss_op,self.gs)
        self.make_gradient_summaries(self.loss_op)
        self.summaries = tf.summary.merge_all()


    @staticmethod
    def embed_sentances(s1):
        with tf.device("/cpu:0"):
            embedding_size = int(np.sqrt(FLAGS.vocab_size) + 1)
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
                next_input = nonlin_ln(conv1d(next_input,FLAGS.hidden1,width=8))
                tf.summary.histogram("conv_act_{}".format(layer_num),next_input)
            layer_num+=1
        encoded_sentance =next_input
        return encoded_sentance

    def decoder(self,encoded_sentance):

        layer_num =0
        with tf.variable_scope("upsaple{}".format(layer_num),initializer=xavier_initializer()):
            next_input = conv1d_transpose(encoded_sentance, 16, 4)  # [batch_size,1,hidden] =>[bs,h,1,2]
            tf.summary.histogram("upsaple_act_{}".format(layer_num), next_input)
            layer_num +=1
        decoded = next_input
        decoded = tf.squeeze(decoded,axis=1)
        with tf.variable_scope("decoder_dense"):
            dense_decoded = densenet_ops.makeBlock(decoded,4,5)
        return dense_decoded
    def to_logits(self,decoded):
        with tf.variable_scope("logits",):
            return linear(decoded,num_outputs=FLAGS.vocab_size,)
    def loss(self,targets,logits):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)
        return tf.reduce_mean(loss)
    def train(self,loss,gs):
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gvs = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(gvs,global_step=gs)

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

