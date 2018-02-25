# Denoising Autoencoder as TensorFlow estimator

This repository contains an implementation of
a (Denoising) Autoencoder using TensorFlow's
[Estimator](https://www.tensorflow.org/get_started/estimator)
and [Dataset](https://www.tensorflow.org/programmers_guide/datasets) API.
You can find a more detailed description in [my blog post](https://k-d-w.org/node/103).

Two flavors of autoencoders are currently implemented

1. `fully_connected`: Uses fully-connected layers with 128, 64, and 32 units, respectively.
2. `convolutional`: Uses 2D convolutional layers with 3x3 kernels and 16, 8, and 8 filters, respectively.

## Training
You can train an autoencoder on the MNIST dataset using
``train_mnist_autoencoder.py``.
The first argument specifies which autoencoder to train
(`fully_connected` or `convolutional`), the remaining arguments
are listed below.

| Argument | Description |
| -------- | ----------- |
| --model_dir | Output directory for model and training stats. |
| --data_dir  | Directory to download the data to. |
| --noise_factor | Amount of noise to add to input (default: 0) |
| --learning_rate | Learning rate (default: 0.001) |
| --batch_size | Batch size (default: 256) |
| --epochs | Number of epochs to perform for training (default: 50) |
| --weight_decay | Amount of weight decay to apply (default: 1e-5) |
| --dropout | The probability that each element is *kept* in dropout layers (default: 1) |
| --save-images | Path to directory to store pairs of input and reconstructed images after each epoch (default: disabled) |

If ``--noise_factor`` is non-zero, a denoising autoencoder will be trained.

## Prediction
After training an autoencoder, you can inspect the model using
``test_mnist_autoencoder.py``.
You can visualize a model's predictions on the MNIST test dataset or
save the output of the encoder to display it in TensorBoard.
The first argument specifies which autoencoder to evaluate
(`fully_connected` or `convolutional`), the remaining arguments
are listed below.

| Argument | Description |
| -------- | ----------- |
| --model_dir | Output directory for model and training stats. |
| --data_dir  | Directory to download the data to. |
| --noise_factor | Amount of noise to add to input (default: 0) |
| --dropout | The probability that each element is *kept* in dropout layers (default: 1) |
| --images | Number of test images to reconstruct or save embedding for (default: 10) |
| --what | How to visualize a trained model. `reconstruction` displays input and reconstructed images next to each other. `embedding` creates a checkpoint that can be used with TensorBoard to visualize the low-dimensional embedding of the encoder (default: reconstruction) |

Make sure you pass the same values you used during training.
