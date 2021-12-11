"""train.py -- Main training code.
"""

import argparse
import logging
import os
import pickle

from datetime import datetime

import h5py
import numpy as np
import scipy.io
import tensorflow as tf
import yaml

import data_exploration.complex_pca as complex_pca
import models
import util


# noinspection DuplicatedCode
def load_data(data_path: str, tone_start: int, tone_stop: int, constant_features_path: str = None,
              use_gain: bool = True, use_pca: bool = False, use_combined: bool = True,
              components: int = None,
              pca: complex_pca.ComplexPCA = None) -> (list[np.ndarray], np.ndarray, complex_pca.ComplexPCA):
    """Load a flattened dataset from the specified HDF5 file.
    :param data_path: path to the flattened dataset.
    :param tone_start: start index of tones.
    :param tone_stop: stop index of tones.
    :param constant_features_path: path to constant features file.
    :param use_gain: transform (rx, tx) -> rx / tx.
    :param use_pca: apply PCA.
    :param use_combined: combine all inputs (other than the RX HE-PPDU data) into a single tensor.
    :param components: number of principal components to use.
    :param pca: pre-fitted ComplexPCA instance (if one exists).
    :return: x and y values.
    """
    if constant_features_path is None:
        constant_features_path = 'data_preprocessing/constant_features.mat'

    # Load dataset and constant features.
    data = h5py.File(data_path, 'r')
    constant_features = scipy.io.loadmat(constant_features_path, squeeze_me=True)
    constant_features = constant_features['constant']

    # L-LTF extraction.
    rx_l_ltf_1 = np.array(data['rx_l_ltf_1'])
    rx_l_ltf_2 = np.array(data['rx_l_ltf_2'])

    tx_l_ltf = constant_features['txLltfFftOut'][()]

    rx_l_ltf_1_trimmed = rx_l_ltf_1[:, tx_l_ltf != 0]
    rx_l_ltf_2_trimmed = rx_l_ltf_2[:, tx_l_ltf != 0]
    tx_l_ltf_trimmed = tx_l_ltf[tx_l_ltf != 0]

    l_ltf_1_trimmed_gain = rx_l_ltf_1_trimmed / tx_l_ltf_trimmed
    l_ltf_2_trimmed_gain = rx_l_ltf_2_trimmed / tx_l_ltf_trimmed

    # HE-LTF extraction.
    he_ltf_data_indices = constant_features['iMDataTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_pilot_indices = constant_features['iMPilotTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_size = 256

    rx_he_ltf_data = np.array(data['rx_he_ltf_data'])
    rx_he_ltf_pilot = np.array(data['rx_he_ltf_pilot'])
    rx_he_ltf = np.zeros((rx_he_ltf_data.shape[0], he_ltf_size), dtype=complex)
    rx_he_ltf[:, he_ltf_data_indices] = rx_he_ltf_data
    rx_he_ltf[:, he_ltf_pilot_indices] = rx_he_ltf_pilot

    tx_he_ltf = constant_features['txHeltfFftOut'][()]

    rx_he_ltf_trimmed = rx_he_ltf[:, tx_he_ltf != 0]
    tx_he_ltf_trimmed = tx_he_ltf[tx_he_ltf != 0]

    he_ltf_trimmed_gain = rx_he_ltf_trimmed / tx_he_ltf_trimmed

    # Data and pilot extraction.
    rx_he_ppdu_pilot = np.array(data['rx_pilot'])
    tx_he_ppdu_pilot = np.array(data['tx_pilot'])
    he_ppdu_pilot_gain = rx_he_ppdu_pilot / tx_he_ppdu_pilot

    rx_he_ppdu_data = np.array(data['rx_data'][:, tone_start:tone_stop])
    # tx_he_ppdu_data = np.array(data['rx_ref_data'][:, tone_start:tone_stop])
    tx_he_ppdu_data = np.array(data['tx_data'][:, tone_start:tone_stop])    # TODO: Make this configurable.

    # Combine data.
    if use_combined:
        if use_gain:
            X = np.hstack([
                l_ltf_1_trimmed_gain,
                l_ltf_2_trimmed_gain,
                he_ltf_trimmed_gain,
                he_ppdu_pilot_gain
            ])
        else:
            X = np.hstack([
                rx_l_ltf_1,
                rx_l_ltf_2,
                np.tile(tx_l_ltf, (rx_l_ltf_1.shape[0], 1)),
                rx_he_ltf,
                np.tile(tx_he_ltf, (rx_he_ltf.shape[0], 1)),
                rx_he_ppdu_pilot,
                tx_he_ppdu_pilot
            ])

        if use_pca:
            components = X.shape[1] if components is None else components
            if pca is None:
                pca = complex_pca.ComplexPCA(components)
                pca.fit(X)

            X = pca.transform(X)[:, 0:components]

        x = [X, rx_he_ppdu_data]
    else:
        if use_gain:
            x = [
                l_ltf_1_trimmed_gain,
                l_ltf_2_trimmed_gain,
                he_ltf_trimmed_gain,
                he_ppdu_pilot_gain,
                rx_he_ppdu_data
            ]
        else:
            x = [
                rx_l_ltf_1,
                rx_l_ltf_2,
                tx_l_ltf,
                rx_he_ltf,
                tx_he_ltf,
                rx_he_ppdu_pilot,
                tx_he_ppdu_pilot,
                rx_he_ppdu_data
            ]

    y = tx_he_ppdu_data

    return x, y, pca


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', help='seed for Numpy RNG', default=42, type=int)
    parser.add_argument('--output_dir', help='output directory', default=None, type=str)

    parser.add_argument('--model_config', help='model config file', default='config/default.yaml', type=str)
    parser.add_argument('--load_checkpoint', help='load weights from checkpoint file', default=None, type=str)

    parser.add_argument('--train', help='training dataset', default='default.h5', type=str)
    parser.add_argument('--batch_size', help='batch size', default=100, type=int)
    parser.add_argument('--shuffle_buffer_size', help='shuffle buffer size', default=10000, type=int)
    parser.add_argument('--epochs', help='training epochs', default=100, type=int)
    parser.add_argument('--loss_function', help='loss function', default='mean_squared_error', type=str)
    parser.add_argument('--learning_rate', help='learning rate', default=0.00001, type=float)
    parser.add_argument('--cpu', help='CPU only operation (disable GPU)', action='store_true')

    parser.add_argument('--use_gain', help='transform (rx, tx) -> rx / tx', action='store_true')
    parser.add_argument('--use_pca', help='use PCA', action='store_true')
    parser.add_argument('--components', help='number of PCA components', default=None, type=int)

    args = parser.parse_args()

    # Disable info messages from TensorFlow.
    tf.get_logger().setLevel(logging.WARNING)

    # Seed the PRNG.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Disable GPU acceleration.
    if args.cpu:
        util.disable_gpu()

    # Build the model.
    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)

    # Compute the number of input parameters (excluding the RX HE-PPDU data tones).
    if not args.use_pca or args.components is None:
        if args.use_gain:
            # L-LTF-1 gain + L-LTF-2 gain + HE-LTF gain + HE-PPDU pilot gain.
            num_inputs = 52 + 52 + 242 + 8
        else:
            # RX L-LTF-1 + RX L-LTF-2 + TX L-LTF + RX HE-LTF + TX HE-LTF + RX HE-PPDU pilot + TX HE-PPDU pilot.
            num_inputs = 64 + 64 + 64 + 256 + 256 + 8 + 8
    else:
        num_inputs = args.components

    model_config['model']['parameters']['num_inputs'] = num_inputs

    model = models.MODELS[model_config['model']['name']](**model_config['model']['parameters']).build()
    adam = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=adam, loss=args.loss_function)

    # Load the model weights.
    if args.load_checkpoint is not None:
        model.load_weights(args.load_checkpoint)

    # Create output directory.
    output_dir = f'output/{model_config["model"]["name"]}/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}'
    output_dir = args.output_dir or output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load dataset.
    num_pred_tones = model_config['model']['parameters']['num_pred_tones']
    tone_start = model_config['model']['parameters']['tone_start']
    tone_stop = tone_start + num_pred_tones

    data_args = {
        'tone_start': tone_start,
        'tone_stop': tone_stop,
        'use_gain': args.use_gain,
        'use_pca': args.use_pca,
        'use_combined': model_config['model']['name'] != 'ConvolutionalModel',
        'components': args.components
    }

    x_train, y_train, pca = load_data(args.train, **data_args)

    # Save PCA.
    with open(f'{output_dir}/pca.pkl', 'wb') as file:
        pickle.dump(pca, file)

    # Training.
    result = model.fit(
        x=x_train,
        y=y_train,
        validation_split=0.2,
        batch_size=args.batch_size,
        shuffle=True,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{output_dir}/weights_epoch_{{epoch}}.h5',
                save_weights_only=True,
                save_freq='epoch',
                save_best_only=False
            )
        ],
    )
    history = result.history

    # Save arguments.
    path = f'{output_dir}/arguments.pkl'
    with open(path, 'wb') as file:
        pickle.dump(args, file)

    # Save model.
    path = f'{output_dir}/model.json'
    with open(path, 'w') as file:
        file.write(model.to_json())

    # Save weights.
    path = f'{output_dir}/weights.h5'
    model.save_weights(path)
    print('Saved model weights to disk')

    # Save history.
    path = f'{output_dir}/history.pkl'
    with open(path, 'wb') as file:
        pickle.dump(history, file)


if __name__ == '__main__':
    main()
