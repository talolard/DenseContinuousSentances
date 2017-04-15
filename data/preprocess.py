from __future__ import print_function
import argparse
import os
import numpy as np
import codecs
import collections
from six.moves import cPickle as pickle
from arg_getter import FLAGS
PAD_TOKEN = chr(1)
class DataLoader():
    def __init__(self):
        self.padding = [0 for _ in range(FLAGS.max_len)]
        self.vocab_path = os.path.join(FLAGS.data_dir,"voab.pkl")
        self.train_path = os.path.join(FLAGS.data_dir, "train")
        if os.path.exists(self.vocab_path):

            with open(self.vocab_path,"rb") as f:
                self.vocab =pickle.load(f)
                self.data = np.load(self.train_path+'.npy')
        else:
            os.mkdir(FLAGS.data_dir)
            print("Loading data")
            self.load_data()
        FLAGS.vocab_size = len(self.vocab)
        self.reverse_vocab = {val:key for key,val in self.vocab.items()}
    def num_to_str(self,nums):
        return ''.join(map(self.reverse_vocab.get,nums))
    def to_padded_array(self,s):
        enumerated = list(map(self.vocab.get,s))
        padded = (enumerated +self.padding)[:FLAGS.max_len]
        return np.array(padded)
    def get_batch(self):
        start =0
        end =start + FLAGS.batch_size
        while end < len(self.data):
            batch = self.data[start:end]
            yield batch
            start =end
            end +=FLAGS.batch_size
    def load_data(self):
        with open(FLAGS.input_file) as f:
            txt = f.read()
            vocab = set(txt)
            vocab = {char:num+1 for num,char in enumerate(vocab)}
            vocab[PAD_TOKEN] =0
            self.vocab  =vocab
            with open(self.vocab_path,"wb") as f:
                pickle.dump(self.vocab,f)
            start =0
            end = FLAGS.max_len
            sentances =[]
            while end <len(txt) + FLAGS.max_len:
                sentances.append(txt[start:end])
                start+=FLAGS.max_len
                end += FLAGS.max_len
            data = list(map(self.to_padded_array,sentances))
            self.data =np.stack(data)
            np.save(self.train_path,self.data)

