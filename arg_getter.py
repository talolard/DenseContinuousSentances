import tensorflow as tf
import pickle

flags = tf.app.flags
flags.DEFINE_string('input_file','/home/tal/dev/vae_text/data/on_the_origin_of_species.txt','data_file')
flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
flags.DEFINE_float('dropout_keep_prob', 1, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('num_topics', 8, 'Number of LDA topics.')
flags.DEFINE_integer('max_len', 256, 'Maximum sequence length.')

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('vocab_size', None, 'Vocab size.')
flags.DEFINE_string('data_dir', '/home/tal/dev/vae_text/data/species','Where to get data')
flags.DEFINE_string('save_dir', './chkpoint/3/','Where to save checkpoints')
flags.DEFINE_string('mode', 'normal','set to test to check flow')
FLAGS = tf.app.flags.FLAGS
