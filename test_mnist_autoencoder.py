from os.path import join
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_autoencoder.cli import create_test_parser
from tf_autoencoder.embeddings import get_tensor_output, save_as_embedding
from tf_autoencoder.img_utils import write_sprite_image
from tf_autoencoder.inputs import MNISTReconstructionDataset
from tf_autoencoder.estimator import AutoEncoder, ConvolutionalAutoencoder

# Show debugging output
tf.logging.set_verbosity(tf.logging.INFO)


def create_conv_model(args):
    return ConvolutionalAutoencoder(num_filters=[16, 8, 8],
                                    dropout=args.dropout,
                                    model_dir=args.model_dir)


def create_fc_model(args):
    return AutoEncoder(hidden_units=[128, 64, 32],
                       dropout=args.dropout,
                       model_dir=args.model_dir)


def run_test(args=None):
    parser = create_test_parser()
    args = parser.parse_args(args=args)

    if args.model == 'fully_connected':
        pred_estimator = create_fc_model(args)
        encoder_output = 'FCAutoEnc/encoder/output:0'
    elif args.model == 'convolutional':
        pred_estimator = create_conv_model(args)
        encoder_output = 'ConvAutoEnc/encoder/output:0'
    else:
        raise ValueError('unknown model {}'.format(args.model))

    n_images = args.images
    data = MNISTReconstructionDataset(args.data_dir, args.noise_factor)
    test_input_fn = data.get_test_input_fn(n_images)

    if args.what == 'reconstruction':
        is_denoising = args.noise_factor > 0
        show_reconstructed_images(pred_estimator, test_input_fn, n_images, is_denoising)
    elif args.what == 'embedding':
        embedding_ckpt = join(args.model_dir, 'embedding', 'encoder')
        embedding_labels = join(args.model_dir, 'embedding', 'labels.csv')

        embedding_data = get_tensor_output(
            pred_estimator, test_input_fn, encoder_output)[:n_images]
        tf.logging.info('Writing embedding of shape %s to %s',
                        embedding_data.shape, embedding_ckpt)

        save_as_embedding(embedding_data,
                          embedding_ckpt,
                          metadata_path='./labels.csv',
                          sprite_image_path='./images.png')

        embedding_sprites = join(args.model_dir, 'embedding', 'images.png')
        write_sprite_image(data.mnist.test.images, [28, 28], embedding_sprites)

        np.savetxt(embedding_labels,
                   data.mnist.test.labels[:n_images],
                   fmt='%d')


def get_images(input_fn):
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            x, y = input_fn()

        with tf.Session() as sess:
            input_fn.init_hook.after_create_session(sess, None)
            corrupt_images, images = sess.run([x, y])
    return corrupt_images, images


def show_reconstructed_images(pred_estimator, input_fn, n_images, is_denoising):
    corrupt_images, images = get_images(input_fn)

    n_rows = int(math.ceil(n_images / 10.))
    factor = 3 if is_denoising else 2

    pred_iter = pred_estimator.predict(input_fn, hooks=[input_fn.init_hook])
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

    run_test()
