"""ber_curve.py -- Plot BER curve for a model.
"""

import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

import models
import qam_decode
import train


def compute_ber(model_dir: str, data_fmt: str) -> dict:
    # Use cached value if possible.
    result_path = f'{model_dir}/ber.pkl'
    if os.path.exists(result_path):
        with open(result_path, 'rb') as file:
            result = pickle.load(file)
        return result

    # Load the model.
    arguments_path = f'{model_dir}/arguments.pkl'
    with open(arguments_path, 'rb') as file:
        args = pickle.load(file)

    config_path = args.model_config
    with open(config_path, 'r') as file:
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

    weights_path = f'{model_dir}/weights.h5'
    model.load_weights(weights_path)

    # Compute the BER for each SNR.
    if args.use_pca:
        pca_path = f'{model_dir}/pca.pkl'
        with open(pca_path, 'rb') as file:
            pca = pickle.load(file)
    else:
        pca = None

    num_pred_tones = model_config['model']['parameters']['num_pred_tones']
    tone_start = model_config['model']['parameters']['tone_start']
    tone_stop = tone_start + num_pred_tones

    data_args = {
        'tone_start': tone_start,
        'tone_stop': tone_stop,
        'use_gain': args.use_gain,
        'use_pca': args.use_pca,
        'use_combined': model_config['model']['name'] != 'ConvolutionalModel',
        'components': args.components,
        'pca': pca
    }

    result = {'snr': [], 'ber': [], 'ber_wbb': []}
    for snr in [10, 15, 20, 25, 30, 35, 40, 45]:
        data_path = data_fmt.format(snr)
        x, y, _ = train.load_data(data_path, **data_args)
        y_hat = model.predict(x)

        data = h5py.File(data_path, 'r')
        y_wbb = np.array(data['/rx_ref_data'][:, tone_start:tone_stop])

        bits = qam_decode.decode(y, 7)
        bits_hat = qam_decode.decode(y_hat, 7)
        bits_wbb = qam_decode.decode(y_wbb, 7)

        result['snr'].append(snr)
        result['ber'].append(np.mean(bits_hat != bits))
        result['ber_wbb'].append(np.mean(bits_wbb != bits))

        print(result['snr'][-1], result['ber'][-1], result['ber_wbb'][-1])

    with open(result_path, 'wb') as file:
        pickle.dump(result, file)

    return result


def plot_ber(model_dirs: list[str], names: list[str], data_fmt: str) -> None:
    results = [compute_ber(model_dir, data_fmt) for model_dir in model_dirs]

    plt.figure()
    for result in results:
        plt.plot(result['snr'], result['ber'])
    plt.plot(results[0]['snr'], results[0]['ber_wbb'])
    plt.xlabel('SNR [dB]')
    plt.ylabel('BER')
    plt.xlim([10, 20])
    plt.ylim([1e-2, 1])
    plt.yscale('log')
    plt.title('BER Curve')
    plt.legend(names + ['Freq. Smoothing LS Equalizer'])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_ber(
        model_dirs=[
            # 'output/dense_model_no_gain',
            # 'output/convolutional_model',
            'output/dense_model_gain',
            'output/dense_model_gain_trimmed_25',
            # 'output/dense_model_gain_pca_10',
            # 'output/dense_model_gain_pca_25',
            # 'output/dense_model_gain_pca_50',
            # 'output/dense_model_gain_pca_100',
            # 'output/dense_model_gain_regularized_l2',
            # 'output/dense_model_gain_regularized_l1_l2'
        ],
        names=[
            # 'Naive Dense Model',
            # 'Convolutional Model',
            'Dense Model',
            'Trimmed Dense Model ($k = 25$)',
            # 'Dense Model + PCA ($k = 10$)',
            # 'Dense Model + PCA ($k = 25$)',
            # 'Dense Model + PCA ($k = 50$)',
            # 'Dense Model + PCA ($k = 100$)',
            # 'Dense Model + L2',
            # 'Dense Model + L1/L2'
        ],
        data_fmt=r'D:\EE 364D\dataset\synthetic_data\channel_specific\test_indoor_{0}dB\test_indoor_{0}dB_channel_e_flat.h5'
    )
