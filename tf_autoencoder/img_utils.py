import numpy as np
import math
import matplotlib.pyplot as plt


def num_rows(n_images, n_cols):
    return int(math.ceil(n_images / float(n_cols)))


def image_grid(pred_iter, images, corrupt_images=None, n_images=10, cmap=plt.cm.Greys_r):
    n_rows = num_rows(n_images, 10)
    factor = 3 if corrupt_images is not None else 2

    fig, axs = plt.subplots(factor * n_rows, 10, figsize=(12.5, 3.75))
    for i, pred in enumerate(pred_iter):
        if i == n_images:
            break
        row = factor * (i // 10)
        col = i % 10
        axs[row, col].imshow(images[i].reshape(28, 28), cmap=cmap)
        if corrupt_images is not None:
            axs[row + 1, col].imshow(corrupt_images[i].reshape(28, 28), cmap=cmap)
        axs[row + factor - 1, col].imshow(pred['prediction'].reshape(28, 28), cmap=cmap)

    for ax in axs.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    return fig


def write_image_grid(filename, **kwargs):
    fig = image_grid(**kwargs)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def write_sprite_image(images, image_shape, filename, cmap=plt.cm.Greys, n_cols=100):
    assert images.ndim == 2, 'expected 2D array'
    assert len(image_shape) == 2, 'expected shape of length 2'

    n_rows = num_rows(images.shape[0], n_cols)
    sprite_height = n_rows * image_shape[0]
    sprite_width = n_cols * image_shape[1]
    sprite_img = np.empty((sprite_height, sprite_width),
                          dtype=images.dtype)

    for k, img in enumerate(images):
        i_beg = k // n_cols * image_shape[0]
        j_beg = k % n_cols * image_shape[1]
        i_end = i_beg + image_shape[0]
        j_end = j_beg + image_shape[1]

        sprite_img[i_beg:i_end, j_beg:j_end] = np.reshape(img, image_shape)

    plt.imsave(filename, sprite_img, cmap=cmap, format='png')
