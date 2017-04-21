import tensorflow as tf
from arg_getter import FLAGS
from models.baseline_conv_enc import BaselineConvEncDec
from models.densenet.densenet_decoder import DenseNetDecoder
from models.densenet.densenet_encoder import DenseNetEncoder


class VAE(BaselineConvEncDec):
    def __init__(self,):
        self.gs = tf.contrib.framework.get_or_create_global_step()
        self.input = tf.placeholder_with_default(tf.ones(shape=[FLAGS.batch_size,FLAGS.max_len],dtype=tf.int32),shape=[FLAGS.batch_size,FLAGS.max_len],)
        embedded = self.embed_sentances(self.input)
        encoded = DenseNetEncoder(_input=embedded, growth_rate=16, num_blocks=3, layers_per_batch=5)
        normed_encoded,kl_loss = self.to_normed_vec(encoded)
        decoded = DenseNetDecoder(encoded,layers_per_batch=5,growth_rate=4,expansion_rate=2)
        logits = self.to_logits(decoded)
        kl_weight = 1.0- tf.train.exponential_decay(1.0,global_step=self.gs,decay_steps=100000,decay_rate=0.999)
        kl_weight =0
        self.preds_op = self.preds(logits)
        self.loss_op = self.loss(self.input,logits)
        kl_loss =tf.reduce_sum(kl_loss)
        total_loss  = self.loss_op+kl_weight*kl_loss
        tf.summary.scalar("loss",self.loss_op)
        tf.summary.scalar("kl_loss", kl_loss)
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("kl_weight", kl_weight)

        self.train_op = self.train(total_loss,self.gs)
        self.make_gradient_summaries(self.loss_op+kl_loss)
        self.summaries = tf.summary.merge_all()

    def to_normed_vec(self,latent):
        mu = tf.contrib.layers.linear(latent,num_outputs=FLAGS.hidden2)
        sig = tf.contrib.layers.linear(latent,num_outputs=FLAGS.hidden2)
        epsilon = tf.random_normal(tf.shape(sig), name='epsilon')
        std =  tf.exp(0.5 * sig)
        z = mu+tf.multiply(std,epsilon)
        kl_loss =  -0.5 * tf.reduce_sum(1 + sig - tf.pow(mu, 2) - tf.exp(sig), reduction_indices=1)
        return z,kl_loss



