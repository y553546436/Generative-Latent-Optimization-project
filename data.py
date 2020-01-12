import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
## enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)

class FLAGS(object):
    def __init__(self):
        self.n_epoch = 25 # "Epoch to train [25]"
        self.z_dim = 32 # "Num of noise value]"
        #self.lr = 0.01 # "Learning rate for SGD"
        #self.beta1 = 0.5 # "Momentum term of adam [0.5]")
        self.batch_size = 512 # "The number of batch images [64]")
        self.output_size = 32 # "The size of the output images to produce [64]")
        self.sample_size = 64 # "The number of sample images [64]")
        self.c_dim = 1 # "Number of image channels. [3]")
        self.save_every_epoch = 1 # "The interval of saveing checkpoints.")
        # self.dataset = "celebA" # "The name of dataset [celebA, mnist, lsun]")
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]")
        assert np.sqrt(self.sample_size) % 1 == 0., 'Flag `sample_size` needs to be a perfect square'
flags = FLAGS()

tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image
