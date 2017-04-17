import tensorflow as tf
from arg_getter import FLAGS
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
import numpy as np
import os
import pandas as pd
import time
from datetime import datetime
from data.preprocess import  DataLoader
from models.baseline_conv_enc import BaselineConvEncDec
from models.seq2seqvae import Seq2SeqVAE
from utils import print_paramater_count



def main(__):
    #    label, inputs, lengths = prepareInputsBatch(FLAGS.batch_size)
    train_dir = os.path.join(FLAGS.save_dir,"train","results")
    val_dir = os.path.join(FLAGS.save_dir, "val","results")
    test_dir = os.path.join(FLAGS.save_dir, "test","results")



    DP = DataLoader()
    model = BaselineConvEncDec()
    print_paramater_count()
    init = tf.global_variables_initializer()
    with MonitoredTrainingSession(
            checkpoint_dir=FLAGS.save_dir,
            save_summaries_steps=20,
            hooks=[]

    ) as sess:


        sess.run(init,)

        for epoch in range(FLAGS.num_epochs):
            for batch_num,batch in enumerate(DP.get_batch()):
                _, loss,summary = sess.run([model.train_op, model.loss_op,model.summaries], feed_dict={model.input: batch})
                if batch_num % 100 == 0:
                    preds = sess.run(model.preds_op, feed_dict={model.input: batch})
                    inp = batch[0]
                    pred = preds[0]
                    print(loss)
                    print(DP.num_to_str(inp))
                    print(DP.num_to_str(pred))


if __name__ == '__main__':
    tf.app.run()
