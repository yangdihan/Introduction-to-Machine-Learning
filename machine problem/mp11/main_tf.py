"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan


def train(model, mnist_dataset, learning_rate=0.0005, batch_size=16,
          num_steps=8000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1,1,[batch_size,14])
        # Train generator and discriminator
        _,_, gl, dl = model.session.run([model.update_g, model.update_d, model.g_loss, model.d_loss], feed_dict={model.x_placeholder:batch_x, model.z_placeholder:batch_z, model.learning_rate_placeholder:learning_rate})

        # model.session.run(, feed_dict={model.x_placeholder:batch_x, model.z_placeholder:batch_z, model.learning_rate_placeholder:learning_rate})
        if (step % 1000==0):
            print("Step "+str(step)+"; G-loss: "+str(gl)+"; D-loss: "+str(dl))

def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan()

    # Start training
    train(model, mnist_dataset)


    out = np.empty((28*20, 28*20))
    for x_idx in range(20):
        for y_idx in range(20):
            z = np.random.uniform(-1., 1., size=[1, 14])
            img = model.session.run(model.x_hat, feed_dict={model.z_placeholder: z})
            out[x_idx*28:(x_idx+1)*28,
                y_idx*28:(y_idx+1)*28] = img[0].reshape(28, 28)
    plt.imsave('latent_space_vae.png', out, cmap="gray")

if __name__ == "__main__":
    tf.app.run()
