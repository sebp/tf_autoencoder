from pathlib import Path
from tabulate import tabulate
import tensorflow as tf


class SaveReconstructionListener(tf.train.CheckpointSaverListener):

    def __init__(self, estimator, input_fn, output_dir, n_images=10):
        self._estimator = estimator
        self._input_fn = input_fn
        self._output_dir = Path(output_dir)
        self.n_images = n_images

    def begin(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def after_save(self, session, global_step_value):
        from .img_utils import write_image_grid

        with tf.Session(graph=tf.Graph()) as sess:
            x, y = self._input_fn()
            self._input_fn.init_hook.after_create_session(sess, None)
            img_in, img_actual = sess.run([x, y])

        p = self._estimator.predict(self._input_fn, hooks=[self._input_fn.init_hook])

        out_file = self._output_dir / 'step_{:05d}.png'.format(global_step_value)
        tf.logging.info('Saving reconstructions to %s', out_file)
        write_image_grid(str(out_file),
                         pred_iter=p,
                         images=img_actual,
                         corrupt_images=img_in,
                         n_images=self.n_images)


class PrintParameterSummary(tf.train.SessionRunHook):
    """
    Print a description of the current model parameters.

    Copied from tensorpack/tfutils/model_utils.py
    """

    def after_create_session(self, session, cord):
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if len(train_vars) == 0:
            tf.logging.warn("No trainable variables in the graph!")
            return

        total = 0
        total_bytes = 0
        data = []
        for v in train_vars:
            shape = v.get_shape()
            ele = shape.num_elements()
            total += ele
            total_bytes += ele * v.dtype.size
            data.append([v.name, shape.as_list(), ele, v.device])

        for d in data:
            d.pop()
        table = tabulate(data, headers=['name', 'shape', 'dim'])

        size_mb = total_bytes / 1024.0 ** 2
        summary_msg = "\nTotal #vars={}, #params={}, size={:.02f}MB".format(
            len(data), total, size_mb)
        tf.logging.info("Model Parameters: \n" + table + summary_msg)

