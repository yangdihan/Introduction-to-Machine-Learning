"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=14):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)


        # Add optimizers for appropriate variables
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        d_vars = tf.trainable_variables("d_")
        g_vars = tf.trainable_variables("g_")

        self.update_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.d_loss, var_list=d_vars)
        self.update_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.g_loss, var_list=g_vars)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1).
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("d_", reuse=reuse) as scope:

            # d_w_conv1 = tf.get_variable('d_w_conv1', [7, 7, 1, 14], initializer=tf.contrib.layers.xavier_initializer())
            # d_b_conv1 = tf.get_variable('d_b_conv1', [14], initializer=tf.constant_initializer(0))
            # d_w_conv2 = tf.get_variable('d_w_conv2', [7, 7, 14, 196], initializer=tf.contrib.layers.xavier_initializer())
            # d_b_conv2 = tf.get_variable('d_b_conv2', [196], initializer=tf.constant_initializer(0))
            # d_w_fc1 = tf.get_variable('d_w_fc1', [self._ndims, 28], initializer=tf.contrib.layers.xavier_initializer())
            # d_b_fc1 = tf.get_variable('d_b_fc1', [28], initializer=tf.constant_initializer(0))
            # d_w_fc2 = tf.get_variable('d_w_fc2', [28, 1], initializer=tf.contrib.layers.xavier_initializer())
            # d_b_fc2 = tf.get_variable('d_b_fc2', [1], initializer=tf.constant_initializer(0))
            #
            # #First Conv and Pool Layers
            # origin = tf.reshape(x,[tf.shape(x)[0],28,28,1])
            # conv1 = tf.nn.conv2d(input=origin, filter=d_w_conv1, strides=[1, 1, 1, 1], padding='SAME')
            # h_conv1 = tf.nn.relu(conv1 + d_b_conv1)
            # h_pool1 = tf.nn.avg_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #
            # #Second Conv and Pool Layers
            # conv2 = tf.nn.conv2d(input=h_pool1, filter=d_w_conv2, strides=[1, 1, 1, 1], padding='SAME')
            # h_conv2 = tf.nn.relu(conv2 + d_b_conv2)
            # h_pool2 = tf.nn.avg_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #
            # #First Fully Connected Layer
            # h_pool2_flat = tf.reshape(h_pool2, [-1, self._ndims])
            # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, d_w_fc1) + d_b_fc1)
            #
            # #Second Fully Connected Layer
            #
            #
            # #Final Layer
            # y=tf.matmul(h_fc1, d_w_fc2) + d_b_fc2

            d_w1 = tf.get_variable('d_w1', [self._ndims,196], initializer=tf.contrib.layers.xavier_initializer())
            d_w2 = tf.get_variable('d_w2', [196,14], initializer=tf.contrib.layers.xavier_initializer())
            d_w3 = tf.get_variable('d_w3', [14,1], initializer=tf.contrib.layers.xavier_initializer())
            d_b1 = tf.get_variable('d_b1', [196], initializer=tf.constant_initializer(0))
            d_b2 = tf.get_variable('d_b2', [14], initializer=tf.constant_initializer(0))
            d_b3 = tf.get_variable('d_b3', [1], initializer=tf.constant_initializer(0))


            h1 = tf.nn.relu(tf.add(tf.matmul(x,d_w1),d_b1))
            h2 = tf.nn.relu(tf.add(tf.matmul(h1,d_w2),d_b2))
            y = tf.add(tf.matmul(h2,d_w3),d_b3)

            return y


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y, labels = tf.ones_like(y)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = tf.zeros_like(y_hat)))
        l = d_loss_real + d_loss_fake
        return l


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("g_", reuse=reuse) as scope:

            g_w1 = tf.get_variable('g_w1', [self._nlatent,196], initializer=tf.contrib.layers.xavier_initializer())
            g_w3 = tf.get_variable('g_w3', [196,self._ndims], initializer=tf.contrib.layers.xavier_initializer())
            g_b1 = tf.get_variable('g_b1', [196], initializer=tf.constant_initializer(0))
            g_b3 = tf.get_variable('g_b3', [self._ndims], initializer=tf.constant_initializer(0))
            
            # Layers
            h1 = tf.nn.relu(tf.add(tf.matmul(z,g_w1),g_b1))
            x_hat = tf.nn.sigmoid(tf.add(tf.matmul(h1,g_w3),g_b3))
            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = tf.ones_like(y_hat)))
        return l
