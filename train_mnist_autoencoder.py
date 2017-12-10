import tensorflow as tf

from tf_autoencoder.inputs import MNISTReconstructionDataset
from tf_autoencoder.estimator import AutoEncoder

# Show debugging output
tf.logging.set_verbosity(tf.logging.INFO)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'model_dir', default_value='./mnist_training',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    'data_dir', default_value='./mnist_data',
    docstring='Directory to download the data to.')

tf.app.flags.DEFINE_float(
    'noise_factor', default_value=0.0,
    docstring='Amount of noise to add to input')
tf.app.flags.DEFINE_float(
    'learning_rate', default_value=0.001,
    docstring='Learning rate')
tf.app.flags.DEFINE_integer(
    'batch_size', default_value=256,
    docstring='Batch size')
tf.app.flags.DEFINE_integer(
    'steps', default_value=5000,
    docstring='Number of steps to perform for training')
tf.app.flags.DEFINE_float(
    'weight_decay', default_value=1e-5,
    docstring='Amount of weight decay to apply')
tf.app.flags.DEFINE_float(
    'dropout', default_value=None,
    docstring='The probability that each element is kept in dropout layers')


def create_experiment(run_config, hparams):
    data = MNISTReconstructionDataset(FLAGS.data_dir,
                                      noise_factor=FLAGS.noise_factor)
    train_input_fn = data.get_train_input_fn(hparams.batch_size)
    eval_input_fn = data.get_eval_input_fn(hparams.batch_size)

    estimator = AutoEncoder(hidden_units=[128, 64, 32],
                            dropout=hparams.dropout,
                            weight_decay=hparams.weight_decay,
                            learning_rate=hparams.learning_rate,
                            config=run_config)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=hparams.train_steps,
        train_monitors=[train_input_fn.init_hook],  # Hooks for training
        eval_hooks=[eval_input_fn.init_hook],  # Hooks for evaluation
        eval_steps=None,  # Use evaluation feeder until its empty
        checkpoint_and_export=True
    )

    return experiment


def run_train(args=None):
    # Define model parameters
    params = tf.contrib.training.HParams(
        dropout=FLAGS.dropout,
        weight_decay=FLAGS.weight_decay,
        learning_rate=FLAGS.learning_rate,
        train_steps=FLAGS.steps,
        batch_size=FLAGS.batch_size)

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=500)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=create_experiment,
        run_config=run_config,
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )


if __name__ == '__main__':
    # avoid printing duplicate log messages
    import logging
    logging.getLogger('tensorflow').propagate = False

    tf.app.run(main=run_train)
