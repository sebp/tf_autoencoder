from os.path import dirname
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def get_tensor_output(estimator, input_fn, tensor_name):
    """Retrieve output from specified tensor.

    Parameters
    ----------
    estimator : :class:`tf.estimator.Estimator`
        Trained estimator. 
    input_fn : :class:`tf_autoencoder.inputs.BaseInputFunction`
        Function producing input to estimator. 
    tensor_name : str
        Name of tensor to retrieve output from.

    Returns
    -------
    array : ndarray
        Output of the specified tensor.
    """
    data_out = []
    with tf.Graph().as_default() as g:
        tf.train.create_global_step(g)

        with tf.device("/cpu:0"):
            x, y = input_fn()

        # create graph
        estimator._model_fn(
            x, y, tf.estimator.ModeKeys.EVAL)

        latest_path = tf.train.latest_checkpoint(estimator._model_dir)
        t = g.get_tensor_by_name(tensor_name)
        # flatten tensor
        t = tf.reshape(t, (tf.shape(t)[0], -1))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            input_fn.init_hook.after_create_session(sess, None)
            saver.restore(sess, latest_path)
            while True:
                try:
                    emb = sess.run(t)
                    data_out.append(emb)
                except tf.errors.OutOfRangeError:
                    break

    return np.concatenate(data_out, axis=0)


def save_as_embedding(data, save_path, metadata_path=None, sprite_image_path=None):
    """Save data as embedding in checkpoint.

    Parameters
    ----------
    data : ndarray
        Data to store as embedding.
    save_path : str
        Path to the checkpoint filename.
    metadata_path : str|None
        Path to meta-data file.
    sprite_image_path : str|None
        Path to sprite images.
    """
    checkpoint_dir = dirname(save_path)

    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            embedding_var = tf.Variable(data, name='embedding')

            writer = tf.summary.FileWriter(checkpoint_dir, g)
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            if metadata_path is not None:
                embedding.metadata_path = metadata_path
            if sprite_image_path is not None:
                embedding.sprite.image_path = sprite_image_path
                # Specify the width and height of a single thumbnail.
                embedding.sprite.single_image_dim.extend([28, 28])

            projector.visualize_embeddings(writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, save_path, 1)
            writer.close()
