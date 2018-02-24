from pathlib import Path
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

