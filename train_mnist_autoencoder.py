import tensorflow as tf

from tf_autoencoder.cli import create_train_parser
from tf_autoencoder.hooks import PrintParameterSummary, SaveReconstructionListener
from tf_autoencoder.inputs import MNISTReconstructionDataset
from tf_autoencoder.estimator import AutoEncoder, ConvolutionalAutoencoder

# Show debugging output
tf.logging.set_verbosity(tf.logging.INFO)


def create_conv_model(run_config, hparams):
    return ConvolutionalAutoencoder(num_filters=[16, 8, 8],
                                    dropout=hparams.dropout,
                                    weight_decay=hparams.weight_decay,
                                    learning_rate=hparams.learning_rate,
                                    config=run_config)


def create_fc_model(run_config, hparams):
    return AutoEncoder(hidden_units=[128, 64, 32],
                       dropout=hparams.dropout,
                       weight_decay=hparams.weight_decay,
                       learning_rate=hparams.learning_rate,
                       config=run_config)


def create_experiment(run_config, hparams):
    data = MNISTReconstructionDataset(hparams.data_dir,
                                      noise_factor=hparams.noise_factor)
    train_input_fn = data.get_train_input_fn(hparams.batch_size, hparams.num_epochs)
    eval_input_fn = data.get_eval_input_fn(hparams.batch_size)

    if hparams.model == 'fully_connected':
        estimator = create_fc_model(run_config, hparams)
    elif hparams.model == 'convolutional':
        estimator = create_conv_model(run_config, hparams)
    else:
        raise ValueError('unknown model {}'.format(hparams.model))

    if hparams.save_images is None:
        listeners = None
    else:
        recon_input_fn = data.get_eval_input_fn(10)
        listeners = [
            SaveReconstructionListener(
                estimator, recon_input_fn, hparams.save_images)
        ]

    saver_hook = tf.train.CheckpointSaverHook(
        estimator._model_dir,
        save_steps=data.mnist.train.num_examples // hparams.batch_size,
        listeners=listeners)

    train_monitors = [
        train_input_fn.init_hook,
        saver_hook,
        PrintParameterSummary(),
    ]

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=None,  # Use training feeder until its empty
        train_monitors=train_monitors,  # Hooks for training
        eval_hooks=[eval_input_fn.init_hook],  # Hooks for evaluation
        eval_steps=None,  # Use evaluation feeder until its empty
        checkpoint_and_export=True
    )

    return experiment


def run_train(args=None):
    parser = create_train_parser()
    args = parser.parse_args(args=args)

    # Define model parameters
    params = tf.contrib.training.HParams(
        model=args.model,
        data_dir=args.data_dir,
        save_images=args.save_images,
        noise_factor=args.noise_factor,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size)

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig(
        model_dir=args.model_dir,
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

    run_train()
