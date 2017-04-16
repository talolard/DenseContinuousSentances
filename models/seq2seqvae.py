import tensorflow.contrib.seq2seq as seq2seq
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import GRUCell
from arg_getter import FLAGS
from models.decoder_fn import nodone_econd_func
from models.densenet.densenet_encoder import DenseNetEncoder

EOS=0
class Seq2SeqVAE():
    def __init__(self,):

        self.input = tf.placeholder_with_default(tf.ones(shape=[FLAGS.batch_size,FLAGS.max_len],dtype=tf.int32),shape=[FLAGS.batch_size,FLAGS.max_len],)
        embedded,matrix = self.embed_sentances(self.input)
        #encoded = DenseNetEncoder(_input=embedded, growth_rate=4, num_blocks=3, layers_per_batch=5)

        encoded =self.encoder(embedded,None)
        deocder_logits= self.simple_decoder(encoded,)
        self.preds_op = self.preds(deocder_logits)
        self.loss_op = self.loss(self.input,deocder_logits)
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
        return s1_embeded,embedding_matrix
    def encoder(self,sentances,lengths):
        with tf.variable_scope("Encoder",initializer=xavier_initializer()) as scope:
            cell =GRUCell(num_units=FLAGS.hidden1)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
              cell=cell,
              inputs=sentances,
              sequence_length=lengths,
              time_major=False,
              dtype=tf.float32
            )
        return encoder_state

    def simple_decoder(self,encoded):
        with tf.variable_scope("rnn_decoder") as scope:
            outputs =[]
            encoded = tf.contrib.layers.fully_connected(encoded, num_outputs=FLAGS.hidden2)
            cell = GRUCell(num_units=FLAGS.hidden2)
            state = encoded
            next_input = encoded
            for step in range(FLAGS.max_len):
                if step >0:
                    scope.reuse_variables()
                next_input,state = cell(next_input,state)
                outputs.append(next_input)
            outputs =tf.stack(outputs,1)
            return tf.contrib.layers.linear(outputs, FLAGS.vocab_size, scope=scope)

    def decoder(self,enocder_state,embedding_matrix,lengths=None):
        with tf.variable_scope("Decoder",initializer=xavier_initializer()) as scope:
            cell = GRUCell(num_units=FLAGS.hidden1)
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, FLAGS.vocab_size, scope=scope)

            decoder_fn = nodone_econd_func(
                output_fn=output_fn,
                encoder_state=enocder_state,
                embeddings=embedding_matrix,
                start_of_sequence_id=EOS,
                end_of_sequence_id=EOS,
                maximum_length=FLAGS.max_len-1,
                num_decoder_symbols=FLAGS.vocab_size,
            )

            decoder_logits,decoder_state,decoder_context_state= seq2seq.dynamic_rnn_decoder(
                    cell=cell,
                    decoder_fn=decoder_fn,
                    inputs=None,
                    sequence_length=tf.constant(FLAGS.max_len,shape=[FLAGS.batch_size],dtype=tf.float32),
                    time_major=False,
                    scope=scope,
                )
            return decoder_logits

    def loss(self,targets,logits):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)
        return tf.reduce_mean(loss)
    def train(self,loss,gs):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gvs = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gvs,gs)

        return train_op
    def preds(self,logits):
        probs = tf.nn.softmax(logits)
        return tf.arg_max(probs,dimension=2)

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

