"""flatten.py -- Flattens HDF5 dataset.
"""

import glob
import math
import os

import h5py
import numpy as np
import scipy.io


def flatten(source: str, dest: str, packets_per_chunk: int = 1000, synthetic: bool = False,
            silent: bool = True, constant_features_path: str = 'constant_features.mat') -> (int, int):
    """Flatten an HDF5 dataset.
    :param source: path to original dataset.
    :param dest: path to store flattened dataset.
    :param packets_per_chunk: number of packets to process at a time. Ideally the number of symbols per chunk is < 1000.
    :param synthetic: flag the source dataset as being synthetic (changes data input format).
    :param silent: suppress output.
    :param constant_features_path: path to constant features.
    :return: tuple containing the number of (symbols, packets) converted.
    """
    data = h5py.File(source, 'r')
    num_packets = len(data['rxLltfFftOut1'])

    # Load constant features. This contains the indices for data and pilot tones.
    constant_features = scipy.io.loadmat(constant_features_path, squeeze_me=True)
    constant_features = constant_features['constant']

    # Values in `constant_features.mat` are one-indexed so we subtract one for zero-indexed.
    he_ltf_data_indices = constant_features['iMDataTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_pilot_indices = constant_features['iMPilotTone_Heltf'][()].astype(np.int32) - 1
    data_indices = constant_features['iMDataTone_HePpdu'][()].astype(np.int32) - 1
    pilot_indices = constant_features['iMPilotTone_HePpdu'][()].astype(np.int32) - 1

    # Create the file to write the data to.
    output = h5py.File(dest, 'w')

    for chunk in range(math.ceil(num_packets / packets_per_chunk)):
        chunk_indices = np.arange(chunk * packets_per_chunk, min((chunk + 1) * packets_per_chunk, num_packets))

        # Fixes bug with h5py version on TACC.
        chunk_indices = list(chunk_indices)

        # Load L-LTF fields.
        rx_l_ltf_1 = np.array(data['rxLltfFftOut1'][chunk_indices])
        rx_l_ltf_2 = np.array(data['rxLltfFftOut2'][chunk_indices])

        # Load HE-LTF and split into data and pilot.
        rx_he_ltf = np.array(data['rxHeltfFftOut'][chunk_indices])
        rx_he_ltf_data = rx_he_ltf[:, he_ltf_data_indices]
        rx_he_ltf_pilot = rx_he_ltf[:, he_ltf_pilot_indices]

        # Load RX and TX data symbols and split into data and pilot, keeping track of the corresponding LTF fields.
        # FIXME: This is very slow. On the other hand, I can't be bothered to make it faster since it only runs once.
        ltf_indices = []
        rx_symbols = []
        rx_ref_data = []
        tx_symbols = []

        if synthetic:
            for i, j in enumerate(chunk_indices):
                iterator = (data['rxHePpduFftOut'][j], data['rxHePpdutoneMappedFreq'][j], data['txConstellation'][j])
                for rx_symbol, rx_ref_symbol, tx_symbol in zip(*iterator):
                    ltf_indices.append(i)
                    rx_symbols.append(rx_symbol)
                    rx_ref_data.append(rx_ref_symbol)
                    tx_symbols.append(tx_symbol)
        else:
            for i, j in enumerate(chunk_indices):
                iterator = (data[f'rxHePpduFftOut{j}'], data[f'rxHePpdutoneMappedFreq{j}'], data[f'txConstellation{j}'])
                for rx_symbol, rx_ref_symbol, tx_symbol in zip(*iterator):
                    ltf_indices.append(i)
                    rx_symbols.append(rx_symbol)
                    rx_ref_data.append(rx_ref_symbol)
                    tx_symbols.append(tx_symbol)

        num_symbols = len(ltf_indices)

        ltf_indices = np.array(ltf_indices)
        rx_symbols = np.array(rx_symbols)
        rx_ref_data = np.array(rx_ref_data)
        tx_symbols = np.array(tx_symbols)

        rx_data = rx_symbols[:, data_indices]
        rx_pilot = rx_symbols[:, pilot_indices]
        tx_data = tx_symbols[:, data_indices]
        tx_pilot = tx_symbols[:, pilot_indices]

        # Duplicate the entries in the LTF fields so that we have `num_symbols` entries.
        rx_l_ltf_1 = np.array([rx_l_ltf_1[ltf_indices[i]] for i in range(num_symbols)])
        rx_l_ltf_2 = np.array([rx_l_ltf_2[ltf_indices[i]] for i in range(num_symbols)])
        rx_he_ltf_data = np.array([rx_he_ltf_data[ltf_indices[i]] for i in range(num_symbols)])
        rx_he_ltf_pilot = np.array([rx_he_ltf_pilot[ltf_indices[i]] for i in range(num_symbols)])

        # Lump the arrays together and save them.
        arrays = {
            'rx_l_ltf_1': rx_l_ltf_1,
            'rx_l_ltf_2': rx_l_ltf_2,
            'rx_he_ltf_data': rx_he_ltf_data,
            'rx_he_ltf_pilot': rx_he_ltf_pilot,
            'rx_data': rx_data,
            'rx_pilot': rx_pilot,
            'rx_ref_data': rx_ref_data,
            'tx_data': tx_data,
            'tx_pilot': tx_pilot
        }

        for key, value in arrays.items():
            if key not in output:
                output.create_dataset(name=key, data=value, maxshape=(None, *value.shape[1:]), chunks=True)
            else:
                output[key].resize(output[key].shape[0] + value.shape[0], axis=0)
                output[key][-value.shape[0]:] = value

        if not silent:
            print(f'Wrote {num_symbols} symbols in {len(chunk_indices)} packets.')

    if not silent:
        print(f'TOTAL: {output["rx_l_ltf_1"].shape[0]} symbols in {num_packets} packets.')

    return output['rx_l_ltf_1'].shape[0], num_packets


def flatten_dir(source_dir: str, dest_dir: str, packets_per_chunk: int = 1000, synthetic: bool = False,
                silent: bool = True, force: bool = False,
                constant_features_path: str = 'constant_features.mat') -> (int, int):
    """Flatten all HDF5 files in a directory and place them in another.
    :param source_dir: source directory.
    :param dest_dir: destination directory.
    :param packets_per_chunk: number of packets to process at a time.
    :param synthetic: flag the source dataset as being synthetic (changes data input format).
    :param silent: suppress output.
    :param force: force reprocessing of already-processed files.
    :param constant_features_path: path to constant features.
    :return: tuple containing the number of (symbols, packets) converted.
    """
    total_symbols = 0
    total_packets = 0

    for source in glob.glob(f'{source_dir}/*.h5'):
        if 'flat' in source:
            if not silent:
                print(f'{source} is already flattened. Skipping.')
                print()
            continue

        dest = f'{dest_dir}/{os.path.splitext(os.path.basename(source))[0]}_flat.h5'

        if not force and os.path.exists(dest):
            if not silent:
                print(f'{dest_dir} exists. Skipping.')
                print()
            continue

        num_symbols, num_packets = flatten(source, dest, packets_per_chunk, synthetic=synthetic, silent=silent,
                                           constant_features_path=constant_features_path)

        total_symbols += num_symbols
        total_packets += num_packets

        if not silent:
            print()

    if not silent:
        print(f'TOTAL: {total_symbols} symbols in {total_packets} packets.')


def main():
    import argparse

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source', help='directory to flatten', type=str)
    parser.add_argument('dest', help='directory to store result', type=str)
    parser.add_argument('--synthetic', help='flag the source dataset as being synthetic', action='store_true')
    args = parser.parse_args()

    flatten_dir(args.source, args.dest, synthetic=args.synthetic, silent=False)


if __name__ == '__main__':
    main()
