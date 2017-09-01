import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from network import *
from optim import optimizer
from loss import loss

from tensorflow.examples.tutorials.mnist import input_data
from data import preprocess, deprocess, noise, image_show


def train_mnist(layers_d, layers_g, gan_type, optim_d, optim_g, batch, latent_dim, train_config, save_file_dir,
                loss_lambda=0, device="/gpu:0"):
    
    plt.rcParams['image.cmap'] = 'gray'

    # draw graph
    z = tf.placeholder(shape=(batch, latent_dim), dtype=tf.float32)
    x = tf.placeholder(shape=(batch, 28, 28, 1), dtype=tf.float32)
    lr_d = tf.placeholder(tf.float32)
    lr_g = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    w_gp_e = tf.placeholder(shape=(batch, 1, 1, 1), dtype=tf.float32)

    with tf.device(device):
        # network and loss node
        g_image, g_layers, g_vars, g_moving_avgs = network("generator", z, layers_g, is_training=is_training)
        with tf.variable_scope("") as scope:
            logit_real, d_layers, d_vars, d_moving_avgs = network("discriminator", x, layers_d, is_training=is_training)
            scope.reuse_variables()
            logit_fake, _, _, _ = network("discriminator", g_image, layers_d, is_training=is_training)

        d_loss, g_loss = loss(gan_type, logit_real, logit_fake)
        if gan_type == 'wGAN_GP':
            with tf.variable_scope("") as scope:
                scope.reuse_variables()
                mid_image = w_gp_e * x + (1 - w_gp_e) * g_image
                logit, _, _, _ = network("discriminator", mid_image, layers_d, is_training=is_training)
                add_loss = tf.reduce_mean(
                    (tf.sqrt(tf.reduce_sum(tf.square(tf.gradients(logit, mid_image)), axis=(1, 2, 3))) - 1.0) ** 2)
            d_loss = d_loss + loss_lambda * add_loss

        # optimizer and train step
        d_solver = optimizer(optim_d['type'], lr=optim_d['lr'], beta1=optim_d['beta1'],beta2=optim_d['beta2'])
        g_solver = optimizer(optim_g['type'], lr=optim_g['lr'], beta1=optim_g['beta1'],beta2=optim_g['beta2'])
        
        if gan_type =='wGAN':
            d_clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]
            d_train_prev = d_moving_avgs + d_clip
        else:
            d_train_prev = d_moving_avgs
            
        with tf.control_dependencies(d_train_prev):
            d_train = d_solver.minimize(d_loss, var_list=d_vars)
        with tf.control_dependencies(g_moving_avgs):
            g_train = g_solver.minimize(g_loss, var_list=g_vars)


    # load mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    data = mnist.train.next_batch(10)[0]
    print('mnist shape : ', data[0].shape)
    print('mnist range : ', np.min(data), "to", np.max(data))
    image_show(preprocess(data))


    # train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        g_loss_hist = []
        d_loss_hist = []
        if gan_type == "GAN":
            show_node = [tf.nn.sigmoid(logit_real), tf.nn.sigmoid(logit_fake)]
        else:
            show_node = [logit_real, logit_fake]

        max_iter = int(mnist.train.num_examples * train_config['num_epoch'] / batch)
        print("\nstart training")
        print("iteration number : ", max_iter)

        for it in range(max_iter):
            z_noise = noise(batch, latent_dim)
            minibatch = preprocess(mnist.train.next_batch(batch)[0]).reshape(-1, 28, 28, 1)

            # one step
            for _ in range(train_config['d_per_g']):
                if gan_type == 'wGAN_GP':
                    w_e = np.random.uniform(0, 1, (batch, 1, 1, 1))
                    d_loss_curr, _ = sess.run([d_loss, d_train],
                                              feed_dict={x: minibatch, z: z_noise, lr_d: optim_d['lr'], w_gp_e: w_e, is_training: True})
                else:
                    d_loss_curr, _ = sess.run([d_loss, d_train], feed_dict={x: minibatch, z: z_noise, lr_d: optim_d['lr'], is_training: True})
            g_loss_curr, _ = sess.run([g_loss, g_train], feed_dict={z: z_noise, lr_g: optim_g['lr'], is_training: True})

            g_loss_hist.append(g_loss_curr)
            d_loss_hist.append(d_loss_curr)

            # print loss
            if it % train_config['print_every'] == 0:
                print('Iter: %d, d_loss : %0.3f, g_loss : %0.3f' % (it, d_loss_curr, g_loss_curr))

            # print image and some inforamative node    
            if it % train_config['show_every'] == 0:
                real, fake = sess.run(show_node, feed_dict={x: minibatch, z: z_noise, is_training: False})
                print("discriminator output real : %0.2f, fake : %0.2f" % (np.mean(real), np.mean(fake)))

                g_images = sess.run(g_image, feed_dict={z: z_noise, is_training: False})
                image_show(g_images[:10])

        # plot loss
        plt.plot(g_loss_hist, label="G_loss_hist")
        plt.plot(d_loss_hist, label="D_loss_hist")
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        # save variables and configuration
        saver.save(sess, (save_file_dir + '.ckpt'))
        with open(save_file_dir + '.pickle', 'wb') as f:
            pickle.dump({'layers_d': layers_d, 'layers_g': layers_g, 'gan_type': gan_type,
                         'optim_d': optim_d, 'optim_g': optim_g, 'batch': batch, 'latent_dim': latent_dim,
                         'd_per_g': train_config['d_per_g'], 'num_epoch': train_config['num_epoch']}
                        , f)
        print("model, configuration saved in file %s" % save_file_dir)
