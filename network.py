import tensorflow as tf
import numpy as np


def relu(x):
    return tf.nn.relu(x)


def lrelu(x, a=0.01):
    return a * (x-tf.abs(x))/2 + (tf.abs(x)+x)/2


def elu(x, a=1.0):
    return (tf.abs(x) + x)/2.0 + a*(tf.exp((x-tf.abs(x))/2) - 1)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def network(name, x, layers, is_training = True):
    tensors = {}
    with tf.variable_scope(name):
        for (layer_name, prop) in layers:
            with tf.variable_scope(layer_name):
                shape = prop.get('reshape', None)
                pad = prop.get('pad', 'SAME')
                batch = prop.get('batch', None)
                if shape is not None:
                    x = tf.reshape(x, x.get_shape().as_list()[:1] + shape)
                if layer_name[:2] == 'fc':
                    if batch is not None:
                        x = tf.contrib.layers.fully_connected(x, num_outputs=prop['n_out'], activation_fn=prop['act'],
                                                              normalizer_fn=tf.contrib.layers.batch_norm,
                                                              normalizer_params={'decay': 0.9,   # this works better than 0.999
                                                              'scale': True, 'is_training': is_training}
                                                              )
                        tensors[layer_name] = x
                    else:
                        x = tf.contrib.layers.fully_connected(x, num_outputs=prop['n_out'], activation_fn=prop['act'])
                        tensors[layer_name] = x
                elif layer_name[:4] == 'conv':
                    if batch is not None:
                        x = tf.contrib.layers.conv2d(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=prop['act'],
                                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                                     normalizer_params={'decay': 0.9,
                                                                        'scale': True, 'is_training': is_training}
                                                     )
                        tensors[layer_name] = x
                    else:
                        x = tf.contrib.layers.conv2d(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=prop['act']
                                                     )
                        tensors[layer_name] = x
                elif layer_name[:4] == 'cnvT':
                    if batch is not None:
                        x = tf.contrib.layers.conv2d_transpose(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=prop['act'],
                                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                                     normalizer_params={'decay': 0.9,
                                                                        'scale': True, 'is_training': is_training}
                                                     )
                        tensors[layer_name] = x
                    else:
                        x = tf.contrib.layers.conv2d_transpose(inputs=x, num_outputs=prop['n_out'], kernel_size=prop['k'],
                                                     stride=prop['s'], padding=pad, activation_fn=prop['act']
                                                     )
                        tensors[layer_name] = x
                        
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
    moving_avgs = tf.get_collection(tf.GraphKeys.UPDATE_OPS,name)  # these are assigin nodes of moving_average -non trainable variables
    
    return x, tensors, vars, moving_avgs


if __name__ == "__main__":
    print("start checking networks")

    layers_d = [('conv1', {'n_out': 64, 'k': 4, 's': 2, 'act': lrelu}),
                ('conv2', {'n_out': 128, 'k': 4, 's': 2, 'act': lrelu, 'batch': True}),
                ('fc3', {'n_out': 1024, 'act': lrelu, 'batch': True, 'reshape': [-1]}),
                ('fc4', {'n_out': 1, 'act': sigmoid})
                ]

    layers_g = [('fc1', {'n_out': 1024, 'act': relu, 'batch': True}),
                ('fc2', {'n_out': 7 * 7 * 128, 'act': relu, 'batch': True}),
                ('cnvT3', {'n_out': 64, 'k': 4, 's': 2, 'act': relu, 'batch': True, 'reshape': [7, 7, 128]}),
                ('cnvT4', {'n_out': 1, 'k': 4, 's': 2, 'act': tanh})
                ]

    print("variables")

    a = tf.placeholder(shape = [64,28,28,1], dtype = tf.float32)
    z = tf.placeholder(shape = [64,64], dtype = tf.float32)

    x, layers_d, d_vars, _ = network('discriminator', a, layers_d)
    x_g, layers_g, g_vars, _ = network('generator', z, layers_g)

    n = 0
    print("discriminator output:", x.shape)
    print("layers")
    for i ,j in layers_d.items():
        print(i ,j)
        n += np.prod(j.get_shape().as_list())
    print()
    print("generator output:", x_g)
    print("layers")
    for i ,j in layers_g.items():
        print(i ,j)
        n += np.prod(j.get_shape().as_list())
    print("\n total data size: ", n)

    n = 0
    for v in d_vars:
        n += np.prod(v.get_shape().as_list())
    for v in g_vars:
        n += np.prod(v.get_shape().as_list())
    print("\n total variable size: ", n)

