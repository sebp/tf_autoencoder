import tensorflow as tf

from .layers import autoencoder


def _autoencoder_model_fn(features, labels, hidden_units,
                          activation_fn, dropout, weight_decay, learning_rate, mode):
    """Construct autoencoder model

    Parameters
    ----------
    features : tf.Tensor
        Tensor holding the input data.
    labels : tf.Tensor
        Tenor holding the data to reconstruct.
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
    learning_rate : float
        Learning rate.
    mode : tf.estimator.ModeKeys
        The mode of the model.

    Returns
    -------
    spec : tf.estimator.EstimatorSpec
        Specification of the model.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Define model's architecture
    logits = autoencoder(inputs=features,
                         hidden_units=hidden_units,
                         activation_fn=activation_fn,
                         dropout=dropout,
                         weight_decay=weight_decay,
                         mode=mode)

    probs = tf.nn.sigmoid(logits)

    predictions = {"prediction": probs}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    tf.losses.sigmoid_cross_entropy(labels, logits)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=is_training)

    train_op = None
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            optimizer="Adam",
            learning_rate=learning_rate,
            learning_rate_decay_fn=lambda lr, gs: tf.train.exponential_decay(lr, gs, 1000, 0.96, staircase=True),
            global_step=tf.train.get_global_step(),
            summaries=["learning_rate", "global_gradient_norm"])

        # Add histograms for trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float64), tf.cast(probs, tf.float64))
        }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


class AutoEncoder(tf.estimator.Estimator):
    """A Autoencoder estimator.

    Parameters
    ----------
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
    learning_rate : float
        Learning rate.
    model_dir : str
        Directory where outputs (checkpoints, event files, etc.)
        are written to.
    config : RunConfig
        Information about the execution environment.
    """

    def __init__(self, hidden_units, activation_fn=tf.nn.relu,
                 dropout=None, weight_decay=1e-5, learning_rate=0.001, model_dir=None, config=None):
        def _model_fn(features, labels, mode):
            return _autoencoder_model_fn(
                features=features,
                labels=labels,
                hidden_units=hidden_units,
                activation_fn=activation_fn,
                dropout=dropout,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                mode=mode)

        super(AutoEncoder, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config)
