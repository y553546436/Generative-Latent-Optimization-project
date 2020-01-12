# Generative-Latent-Optimization-project


This is the tensorlayer implementation of the paper Optimizing the Latent Space of Generative Networks.

## Prerequisites

- python 3.6

- tensorflow 2.0.0

- tensorlayer 2.2.0

## Description

Generative Latent Optimization(GLO) is a new framework to train deep convolutional generators using simple reconstruction losses. The success of GANs come from two complementary sources: (A)Leveraging the powerful inductive bias of deep convnets (B)The adversarial training protocol. This paper attempts to disentangle the two factors. So instead of using another deep convolutional network called discriminaters to train the generators, GLO jointly trains the generators and the distribution of the noise space so that the model relies on (A) and avoids (B). GLO gets competitive results compared to GAN, which proves the importance of inductive bias of convnets.

As this paper provides new interesting insights into the generative models, we want to reimplement the basic version of GLO and test it on the MNIST dataset. To adjust to the network architecture, we resize the images to 32 pixels large and scale the pixel value to [-1, 1]. Following are the implementation details.

The generator of GLO follows the same architecture as the generator of DCGAN. We use the API Variable in tensorflow to create as many 32-dimension trainable vectors as the pictures MNIST dataset has, so that there is a one-to-one mapping between the noise vectors and the pictures. We initialize the random vectors of GLO using a Guassian distribution. After each update, the noise vectors are projected to the unit sphere. (i.e. We project each noise vector after each update by dividing its value by the maximum between its length and 1. We use an MSE loss and Stochastic Gradient Descent(SGD) to optimize both the parameters of GLO generator and noise vectors. The authors claim that they set learning rate for GLO generator as 1 and the learning rate of noise vectors at 10, but we find using this set of parameters the loss can't converge well in 25 epoches. After plenty of experiments, we find when setting the two learning rate at 0.001 and 0.1, we can achieve the best converge of the loss and also genetate the most sharp samples.

After training, we generate sample images in the following way. First, we use a gaussian distribution to fit the trained distribution of noise vectors. We randomly select one vector from this gaussian distribution then and put this vector in the generator convnet, the output image is our sample.

However, even though our implementation strictly follows the paper, the generated samples are far less sharp than the pictures shown in the paper. And the MES loss can only converge to about 0.2. That means on average, the generated picture and the target picture have an above 0.4 difference in every pixel, which is considerably large given that the value of each pixel has been scaled to [-1,1]. We tried several adjustments such as changing the batchsize, changing the optimizer from SGD to Adam, changing the learning rate of SGD, but none of these worked. We debugged our code and found the gradient for noise vectors is very small. We think this may have something to do with the poor results, but we can't find why because our training procudure is almost the same as the paper described. If you have any idea about what is wrong with our implementation, please contact us or pull requests.
