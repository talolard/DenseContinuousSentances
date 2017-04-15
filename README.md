# Working towards implementing [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) but with DenseNet
Someone on Reddit asked for an implementation of DenseNet for text (eg 1d convs) so I'm posting this here half baked.
If thats what you are after look in the models directory. If you want to help with the implementation of the paper pull requests are welcome.

## Using this
models.baseline_conv_env_dec is an attempt at an autoencoder comprised of a DenseNet encoder and a "deconvolution" decoder.
 The decoder part still needs work but it does something on short sentances (16 chars)
 Run
 ````
 python train.py
 ````
 To train on the default data. Look in arg_getter.py for the args you can pass.

##Acknoledgemnts
 I borrowed code liberally from other Repos. I promise I'll mention you soon, I just wanted to get this up quick for the guy on Reddit.