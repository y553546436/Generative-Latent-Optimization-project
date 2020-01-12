import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.files import exists_or_mkdir
from glob import glob
from data import flags
from model import get_generator
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import trange
import torch

num_tiles = int(np.sqrt(flags.sample_size))
def train():
    writer_path = './exp1/'
    exists_or_mkdir(writer_path)
    writer = SummaryWriter(writer_path)

    # load mnist
    X_train, y_train, X_val, y_val, X_test, y_test=tl.files.load_mnist_dataset(shape=(-1,28,28,1))
    n = len(X_train)

    # convert mnist images to 32*32
    xnew=np.zeros((n,32,32,1),dtype='float32')
    for i in range(n):
        xnew[i] = tl.prepro.imresize(X_train[i],size=(32,32),interp='bicubic',mode=None)
        xnew[i] /= 128
        xnew[i] -= 1
    X_train=xnew

    G = get_generator([None, flags.z_dim])

    G.train()

    # 2 different optimizers to train G and z separately
    g_optimizer = tf.optimizers.SGD(lr=1e-3)
    z_optimizer = tf.optimizers.SGD(lr=0.1)

    # initialize z by sampling from a Gaussian distribution
    z = tf.Variable(tf.random.normal([n, flags.z_dim], stddev=np.sqrt(1.0/flags.z_dim)), name="z", trainable=True)

    step = 0
    for epoch in trange(flags.n_epoch, desc='epoch loop'):  ## iterate the dataset n_epoch times
        start_time = time.time()
        # iterate over the entire training set once
        for i in range(n//flags.batch_size):
            
            # get X_batch by indexing, without shuffling (important!)
            X_batch=X_train[i*flags.batch_size:(i+1)*flags.batch_size]

            step_time = time.time()
            step += 1
            G.train()  # enable dropout
            with tf.GradientTape(persistent=True) as tape:
                # compute outputs
                z_batch = z[i*flags.batch_size:(i+1)*flags.batch_size]
                # tape.watch(z_batch)
                fake_X = G(z_batch)
                # compute loss and update model
                loss = tl.cost.mean_squared_error(fake_X,X_batch,name='train_loss')

            # compute gradient to G and z
            grad = tape.gradient(loss, G.trainable_weights+[z])

            # Back_propagation
            grad_g = grad[:len(G.trainable_weights)]
            grad_z = grad[len(G.trainable_weights):]
            g_optimizer.apply_gradients(zip(grad_g, G.trainable_weights))
            z_optimizer.apply_gradients(zip(grad_z, [z]))
            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, loss: {:.5f}".format(epoch, flags.n_epoch, i, n//flags.batch_size, time.time() - step_time, loss))

        # normalize z (remain unchanged for those vectors whose length is shorter than 1)
        z.assign(z / tf.math.maximum(tf.math.sqrt(tf.math.reduce_sum(z**2, axis=1))[:, tf.newaxis], 1))

        # testing
        G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
        G.eval()

        # sampling from the z distribution
        z_mean=np.mean(z.numpy(),axis=0)
        z_cov=np.cov(z.numpy(),rowvar=False)
        sample=np.random.multivariate_normal(z_mean,z_cov,size=(25))
        result=G(sample.astype(np.float32))

        G.train()
        tl.visualize.save_images(result.numpy(), [5, 5],
                              '{}/train_{:02d}.png'.format(flags.sample_dir, epoch))
        save_res = np.tile(tf.transpose(result, [0, 3, 1, 2]).numpy(), [1, 3, 1, 1])
        save_res = torch.tensor(save_res)
        save_grid = make_grid(save_res.cpu(), nrow=5, range=(-1, 1), normalize=True)
        writer.add_image('eval/recon_imgs', save_grid, epoch + 1)

if __name__ == '__main__':
    train()
