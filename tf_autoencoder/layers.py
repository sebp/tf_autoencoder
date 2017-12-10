import tensorflow as tf
import tensorflow.contrib.slim as slim


def add_hidden_layer_summary(value):
    tf.summary.scalar('fraction_of_zero_values', tf.nn.zero_fraction(value))
    tf.summary.histogram('activation', value)


def encoder(inputs, hidden_units, dropout, is_training, scope=None):
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
                    net = slim.dropout(net, is_training=is_training,
                                       scope=layer_scope)
                add_hidden_layer_summary(net)

    return net


def decoder(inputs, hidden_units, dropout, is_training, scope=None):
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
                    net = slim.dropout(net, is_training=is_training,
                                       scope=layer_scope)
                add_hidden_layer_summary(net)

        with tf.variable_scope(
                'layer_{}'.format(len(hidden_units) - 1),
                values=(net,)) as layer_scope:
            net = tf.contrib.layers.fully_connected(net, hidden_units[-1],
                                                    activation_fn=None,
                                                    scope=layer_scope)
            tf.summary.histogram('activation', net)
    return net


def autoencoder(inputs, hidden_units, activation_fn, dropout, weight_decay, mode, scope=None):
    """Create autoencoder layers

    Parameters
    ----------
    inputs : tf.Tensor
        Tensor holding the input data.
    hidden_units : list of int
        Number of unites in each hidden layer.
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
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if weight_decay is None:
        weights_regularizer = None
    else:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with tf.variable_scope(scope, 'AutoEnc', [inputs]):
        with slim.arg_scope([tf.contrib.layers.fully_connected],
                            weights_initializer=slim.initializers.variance_scaling_initializer(),
                            weights_regularizer=weights_regularizer,
                            activation_fn=activation_fn):
            net = encoder(inputs, hidden_units, dropout, is_training)
            n_features = inputs.shape[1].value
            decoder_units = hidden_units[:-1][::-1] + [n_features]
            net = decoder(net, decoder_units, dropout, is_training)

    return net
