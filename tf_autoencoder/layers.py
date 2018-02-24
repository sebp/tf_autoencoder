import tensorflow as tf
import tensorflow.contrib.slim as slim


def add_hidden_layer_summary(value):
    tf.summary.scalar('fraction_of_zero_values', tf.nn.zero_fraction(value))
    tf.summary.histogram('activation', value)


def fc_encoder(inputs, hidden_units, dropout, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'encoder', [inputs]):
        tf.assert_rank(inputs, 2)
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope(
                    'layer_{}'.format(layer_id),
                    values=(net,)) as layer_scope:
                net = tf.contrib.layers.fully_connected(
                    net,
                    num_outputs=num_hidden_units,
                    scope=layer_scope)
                if dropout is not None:
                    net = slim.dropout(net)
                add_hidden_layer_summary(net)

    return net


def fc_decoder(inputs, hidden_units, dropout, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'decoder', [inputs]):
        for layer_id, num_hidden_units in enumerate(hidden_units[:-1]):
            with tf.variable_scope(
                    'layer_{}'.format(layer_id),
                    values=(net,)) as layer_scope:
                net = tf.contrib.layers.fully_connected(
                    net,
                    num_outputs=num_hidden_units,
                    scope=layer_scope)
                if dropout is not None:
                    net = slim.dropout(net, scope=layer_scope)
                add_hidden_layer_summary(net)

        with tf.variable_scope(
                'layer_{}'.format(len(hidden_units) - 1),
                values=(net,)) as layer_scope:
            net = tf.contrib.layers.fully_connected(net, hidden_units[-1],
                                                    activation_fn=None,
                                                    scope=layer_scope)
            tf.summary.histogram('activation', net)
    return net


def autoencoder_arg_scope(activation_fn, dropout, weight_decay, data_format, mode):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if weight_decay is None or weight_decay <= 0:
        weights_regularizer = None
    else:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with slim.arg_scope([tf.contrib.layers.fully_connected, conv2d_fixed_padding, tf.contrib.layers.conv2d_transpose],
                        weights_initializer=slim.initializers.variance_scaling_initializer(),
                        weights_regularizer=weights_regularizer,
                        activation_fn=activation_fn),\
         slim.arg_scope([slim.dropout],
                        keep_prob=dropout,
                        is_training=is_training),\
         slim.arg_scope([conv2d_fixed_padding, tf.contrib.layers.conv2d_transpose],
                        kernel_size=3, padding='SAME', data_format=data_format), \
         slim.arg_scope([tf.contrib.layers.max_pool2d], kernel_size=2, data_format=data_format) as arg_sc:
        return arg_sc


def fully_connected_autoencoder(inputs, hidden_units, activation_fn, dropout, weight_decay, mode, scope=None):
    """Create autoencoder with fully connected layers.

    Parameters
    ----------
    inputs : tf.Tensor
        Tensor holding the input data.
    hidden_units : list of int
        Number of units in each hidden layer.
    activation_fn : callable|None
        Activation function to use.
    dropout : float|None
         Percentage of nodes to remain activate in each layer,
         or `None` to disable dropout.
    weight_decay : float|None
        Amount of regularization to use on the weights
        (excludes biases).
    mode : tf.estimator.ModeKeys
        The mode of the model.
    scope : str
        Name to use in Tensor board.

    Returns
    -------
    net : tf.Tensor
        Output of the decoder's reconstruction layer.
    """
    with tf.variable_scope(scope, 'FCAutoEnc', [inputs]):
        with slim.arg_scope(
                autoencoder_arg_scope(activation_fn, dropout, weight_decay, None, mode)):
            net = fc_encoder(inputs, hidden_units, dropout)
            n_features = inputs.shape[1].value
            decoder_units = hidden_units[:-1][::-1] + [n_features]
            net = fc_decoder(net, decoder_units, dropout)

    return net


def fixed_padding(inputs, data_format):
    """Pad image with zeros to have even width and height"""
    h = inputs.shape[2].value
    pad_total = h % 2
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'NHWC':
        pad = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
    elif data_format == 'NCHW':
        pad = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
    else:
        raise ValueError('data_format {} not supported'.format(data_format))

    return tf.pad(inputs, pad)


@slim.add_arg_scope
def conv2d_fixed_padding(inputs,
                         num_outputs,
                         kernel_size,
                         stride=1,
                         padding='SAME',
                         data_format=None,
                         activation_fn=tf.nn.relu,
                         dropout=None,
                         is_training=True,
                         weights_initializer=slim.initializers.variance_scaling_initializer(),
                         weights_regularizer=None,
                         scope=None):
    with tf.variable_scope(scope, 'conv2d', [inputs]):
        h = inputs.shape[2].value
        if h % 2 != 0:
            inputs = fixed_padding(inputs, data_format)

        net = tf.contrib.layers.conv2d(
            inputs=inputs,
            num_outputs=num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            data_format=data_format,
            activation_fn=activation_fn,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer)

        if dropout is not None:
            net = slim.dropout(net, is_training=is_training)
        add_hidden_layer_summary(net)

    return net


def conv_encoder(inputs, num_filters, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'encoder', [inputs]):
        tf.assert_rank(inputs, 4)
        for layer_id, num_outputs in enumerate(num_filters):
            with tf.variable_scope('block{}'.format(layer_id)):
                net = slim.repeat(net, 2, conv2d_fixed_padding, num_outputs=num_outputs)
                net = tf.contrib.layers.max_pool2d(net)

    return net


def conv_decoder(inputs, num_filters, output_shape, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'decoder', [inputs]):
        tf.assert_rank(inputs, 4)
        for layer_id, units in enumerate(num_filters):
            with tf.variable_scope('block_{}'.format(layer_id),
                                   values=(net,)):
                net = tf.contrib.layers.conv2d_transpose(net, units, stride=2)
                add_hidden_layer_summary(net)

        with tf.variable_scope('linear', values=(net,)):
            net = tf.contrib.layers.conv2d_transpose(
                net, 1, activation_fn=None)
            tf.summary.histogram('activation', net)

        with tf.name_scope('crop', values=[net]):
            shape = net.get_shape().as_list()
            assert len(shape) == len(output_shape), 'shape mismatch'
            slice_beg = [0]
            slice_size = [-1]
            for sin, sout in zip(shape[1:], output_shape[1:]):
                if sin == sout:
                    slice_beg.append(0)
                    slice_size.append(-1)
                else:
                    assert sin > sout, "{} <= {}".format(sin, sout)
                    beg = (sin - sout) // 2
                    slice_beg.append(beg)
                    slice_size.append(sout)

            net = tf.slice(net, slice_beg, slice_size, name='output')

    return net


def convolutional_autoencoder(inputs, num_filters, activation_fn, dropout, weight_decay, mode, scope=None):
    """Create autoencoder with 2D convolution layers.

    Parameters
    ----------
    inputs : tf.Tensor
        Tensor holding the input data.
    num_filters : list of int
        Number of filters in each convolutional block.
    activation_fn : callable|None
        Activation function to use.
    dropout : float|None
         Percentage of nodes to remain activate in each layer,
         or `None` to disable dropout.
    weight_decay : float|None
        Amount of regularization to use on the weights
        (excludes biases).
    mode : tf.estimator.ModeKeys
        The mode of the model.
    scope : str
        Name to use in Tensor board.

    Returns
    -------
    net : tf.Tensor
        Output of the decoder's reconstruction layer.
    """
    data_format = "NHWC"
    if data_format == "NHWC":
        shape = [-1, 28, 28, 1]
    elif data_format == "NCHW":
        shape = [-1, 1, 28, 28]
    else:
        raise ValueError("unknown data_format {}".format(data_format))

    with tf.variable_scope(scope, 'ConvAutoEnc', [inputs]):
        with slim.arg_scope(
                autoencoder_arg_scope(activation_fn, dropout, weight_decay, data_format, mode)):

            net = tf.reshape(inputs, shape)
            net = conv_encoder(net, num_filters)
            net = conv_decoder(net, num_filters[::-1], shape)

            net = tf.reshape(net, [-1, 28 * 28])

    return net
