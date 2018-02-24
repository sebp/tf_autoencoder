import argparse


def create_train_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='model')
    fc_parser = subparsers.add_parser('fully_connected')
    ParserCreator(fc_parser) \
        .add_data_arguments() \
        .add_common_arguments() \
        .add_train_arguments()

    conv_parser = subparsers.add_parser('convolutional')
    ParserCreator(conv_parser) \
        .add_data_arguments() \
        .add_common_arguments() \
        .add_train_arguments()

    return parser


def create_test_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='model')

    fc_parser = subparsers.add_parser('fully_connected')
    ParserCreator(fc_parser) \
        .add_data_arguments() \
        .add_common_arguments() \
        .add_test_arguments()

    conv_parser = subparsers.add_parser('convolutional')
    ParserCreator(conv_parser) \
        .add_data_arguments() \
        .add_common_arguments() \
        .add_test_arguments()

    return parser


class ParserCreator:

    def __init__(self, parser=None):
        if parser is None:
            self._parser = argparse.ArgumentParser()
        else:
            self._parser = parser

    def add_data_arguments(self):
        parser = self._parser.add_argument_group('Data')

        parser.add_argument(
            '--model_dir', default='./mnist_training',
            help='Output directory for model and training stats.')
        parser.add_argument(
            '--data_dir', default='./mnist_data',
            help='Directory to download the data to.')
        return self

    def add_common_arguments(self):
        parser = self._parser.add_argument_group('Hyper-parameters')

        parser.add_argument(
            '--batch_size', type=int, default=256,
            help='Batch size (default: 256)')
        parser.add_argument(
            '--noise_factor', type=float, default=0.0,
            help='Amount of noise to add to input (default: 0)')
        parser.add_argument(
            '--dropout', type=float, default=None,
            help='The probability that each element is kept in dropout layers (default: 1)')
        return self

    def add_train_arguments(self):
        parser = self._parser.add_argument_group('Training')

        parser.add_argument(
            '--learning_rate', type=float, default=0.001,
            help='Learning rate (default: 0.001)')
        parser.add_argument(
            '--epochs', type=int, default=50,
            help='Number of epochs to perform for training (default: 50)')
        parser.add_argument(
            '--weight_decay', type=float, default=1e-5,
            help='Amount of weight decay to apply (default: 1e-5)')
        parser.add_argument(
            '--save_images',
            help='Path to directory to store intermediate reconstructed images (default: disabled)')
        return self

    def add_test_arguments(self):
        parser = self._parser.add_argument_group('Testing')

        parser.add_argument(
            '--images', type=int, default=10,
            help='Number of test images to reconstruct (default: 10)')
        return self
