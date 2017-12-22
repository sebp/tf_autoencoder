import math
import tensorflow as tf
import matplotlib.pyplot as plt

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
    docstring='Amount of noise to add to input (default: 0)')
tf.app.flags.DEFINE_float(
    'dropout', default_value=None,
    docstring='The probability that each element is kept in dropout layers (default: 1)')
tf.app.flags.DEFINE_integer(
    'images', default_value=10,
    docstring='Number of test images to reconstruct (default: 10)')


def run_test(args=None):
    data = MNISTReconstructionDataset(FLAGS.data_dir, FLAGS.noise_factor)

    pred_estimator = AutoEncoder(hidden_units=[128, 64, 32],
                                 dropout=FLAGS.dropout,
                                 learning_rate=0.001,
                                 model_dir=FLAGS.model_dir)

    test_input_fn = data.get_test_input_fn(256, )
    with tf.Session() as sess:
        x, y = test_input_fn()
        test_input_fn.init_hook.after_create_session(sess, None)
        corrupt_images = x.eval()

    images = data.mnist.test.images
    n_rows = int(math.ceil(FLAGS.images / 10.))
    n_images = FLAGS.images
    is_denoising = FLAGS.noise_factor > 0
    factor = 3 if is_denoising else 2

    pred_iter = pred_estimator.predict(test_input_fn, hooks=[test_input_fn.init_hook])
    fig, axs = plt.subplots(factor * n_rows, 10, figsize=(15, 5))
    for i, pred in enumerate(pred_iter):
        if i == n_images:
            break
        row = factor * (i // 10)
        col = i % 10
        axs[row, col].imshow(images[i].reshape(28, 28), cmap=plt.cm.Greys_r)
        if is_denoising:
            axs[row + 1, col].imshow(corrupt_images[i].reshape(28, 28), cmap=plt.cm.Greys_r)
        axs[row + factor - 1, col].imshow(pred['prediction'].reshape(28, 28), cmap=plt.cm.Greys_r)

    for ax in axs.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    plt.show()


if __name__ == '__main__':
    # avoid printing duplicate log messages
    import logging
    logging.getLogger('tensorflow').propagate = False

    tf.app.run(main=run_test)
