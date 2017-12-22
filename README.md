# Denoising Autoencoder as TensorFlow estimator

This repository contains an implementation of
a (Denoising) Autoencoder using TensorFlow's
[Estimator](https://www.tensorflow.org/get_started/estimator)
and [Dataset](https://www.tensorflow.org/programmers_guide/datasets) API.
You can find a more detailed description in [my blog post](https://k-d-w.org/node/103).

## Training
You can train an autoencoder on the MNIST dataset using
``train_mnist_autoencoder.py``, which has the following arguments:

| Argument | Description |
| -------- | ----------- |
| --model_dir | Output directory for model and training stats. |
| --data_dir  | Directory to download the data to. |
| --noise_factor | Amount of noise to add to input (default: 0)
| --learning_rate | Learning rate (default: 0.001) |
| --batch_size | Batch size (default: 256) |
| --epochs | Number of epochs to perform for training (default: 50) |
| --weight_decay | Amount of weight decay to apply (default: 1e-5) |
| --dropout | The probability that each element is *kept* in dropout layers (default: 1) |

If ``--noise_factor`` is non-zero, a denoising autoencoder will be trained.

## Prediction
After training an autoencoder, you can visualize its predictions on
the MNIST test dataset using ``test_mnist_autoencoder.py``, which has
the following arguments:

| Argument | Description |
| -------- | ----------- |
| --model_dir | Output directory for model and training stats. |
| --data_dir  | Directory to download the data to. |
| --noise_factor | Amount of noise to add to input (default: 0)
| --dropout | The probability that each element is *kept* in dropout layers (default: 1) |
| --images | Number of test images to reconstruct (default: 10) |

Make sure you pass the same values you used during training.
