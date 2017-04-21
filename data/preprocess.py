from __future__ import print_function
import argparse
import os
import numpy as np
import codecs
import collections
from six.moves import cPickle as pickle
from arg_getter import FLAGS
import random
import string
PAD_TOKEN = chr(128)
class DataLoader():
    def __init__(self):
        self.padding = [0 for _ in range(256)]
        self.vocab_path = os.path.join(FLAGS.data_dir,"voab.pkl")
        self.train_path = os.path.join(FLAGS.data_dir, "train")
        self.printable_set = set(string.printable)
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
    def is_pure_ascii(self,s):
        chars = set(s)
        return len(chars.difference(self.printable_set)) ==0
    def num_to_str(self,nums):
        return ''.join(map(self.reverse_vocab.get,nums))
    def to_encoded_array(self, s):
        enumerated = list(map(self.vocab.get,s))
        padded = (enumerated +self.padding)[:256]
        return np.array(padded)
    def get_batch(self):
        start =0
        end =start + FLAGS.batch_size
        np.random.shuffle(self.data)
        while end < len(self.data):
            batch = self.data[start:end]
            yield batch[:,:FLAGS.max_len]
            start =end
            end +=FLAGS.batch_size
    def load_data(self):
        with open(FLAGS.input_file) as f:
            sentances = f.readlines()
            sentances =list(filter(self.is_pure_ascii,sentances))
            txt =''.join(sentances)
            vocab = set(txt)
            vocab = {char:num+1 for num,char in enumerate(vocab)}
            vocab[PAD_TOKEN] =0
            self.vocab  =vocab
            with open(self.vocab_path,"wb") as f:
                pickle.dump(self.vocab,f)
            data = list(map(self.to_encoded_array, sentances))
            self.data =np.stack(data)
            np.save(self.train_path,self.data)

