import math
import matplotlib.pyplot as plt


def image_grid(pred_iter, images, corrupt_images=None, n_images=10):
    n_rows = int(math.ceil(n_images / 10.))
    factor = 3 if corrupt_images is not None else 2

    fig, axs = plt.subplots(factor * n_rows, 10, figsize=(12.5, 3.75))
    for i, pred in enumerate(pred_iter):
        if i == n_images:
            break
        row = factor * (i // 10)
        col = i % 10
        axs[row, col].imshow(images[i].reshape(28, 28), cmap=plt.cm.Greys_r)
        if corrupt_images is not None:
            axs[row + 1, col].imshow(corrupt_images[i].reshape(28, 28), cmap=plt.cm.Greys_r)
        axs[row + factor - 1, col].imshow(pred['prediction'].reshape(28, 28), cmap=plt.cm.Greys_r)

    for ax in axs.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    return fig


def write_image_grid(filename, **kwargs):
    fig = image_grid(**kwargs)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
