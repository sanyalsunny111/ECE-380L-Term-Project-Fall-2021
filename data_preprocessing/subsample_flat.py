"""subsample_flat.py -- Subsample flattened dataset.
"""

import h5py
import numpy as np
import tqdm


def subsample(source: str, destination: str, proportion: float, batch_size: int = 1000, seed: int = None,
              silent: bool = True) -> (int, int):
    """Subsample a flattened HDF5 dataset.
    :param source: path to original dataset.
    :param destination: path to store subsample of dataset.
    :param proportion: proportion of the original dataset to include (actual proportion will be upper bounded by this).
    :param batch_size: batch size.
    :param seed: seed for the random number generator.
    :param silent: suppress output.
    :return: tuple containing the number of symbols included in the subsample and total symbols.
    """
    if not silent:
        print(f'Source: {source}')
        print(f'Destination: {destination}')
        print(f'Proportion: {proportion * 100:.2f}%')
        print(f'Batch Size: {batch_size}')
        print()

    # Open source dataset.
    data = h5py.File(source, 'r')

    # Set seed if specified.
    np.random.seed(seed)

    # Choose the indices at random without replacement.
    total_symbols = data['rx_data'].shape[0]
    included_symbols = int(total_symbols * proportion)
    indices = np.sort(np.random.choice(np.arange(total_symbols), included_symbols, replace=False))

    # Open destination dataset.
    output = h5py.File(destination, 'w')

    # Create empty fields for each type.
    fields = [
        'rx_l_ltf_1',
        'rx_l_ltf_2',
        'rx_he_ltf_data',
        'rx_he_ltf_pilot',
        'rx_data',
        'rx_pilot',
        'rx_ref_data',
        'tx_data',
        'tx_pilot'
    ]

    for field in fields:
        output.create_dataset(field, shape=(included_symbols, *data[field].shape[1:]), dtype=data[field].dtype)

    # Store data.
    iterator = zip(
        np.array_split(np.arange(included_symbols), included_symbols // batch_size),
        np.array_split(indices, included_symbols // batch_size)
    )

    for i, j in iterator if silent else tqdm.tqdm(iterator, total=included_symbols // batch_size):
        for field in fields:
            output[field][i] = data[field][j]

    return included_symbols, total_symbols


if __name__ == '__main__':
    import argparse

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source', help='file to subsample', type=str)
    parser.add_argument('dest', help='file to store result', type=str)
    parser.add_argument('proportion', help='proportion of data to keep', type=float)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1000)
    parser.add_argument('--seed', help='seed for PRNG', type=int, default=None)
    parser.add_argument('--silent', help='suppress output', action='store_true')
    args = parser.parse_args()

    subsample(args.source, args.dest, args.proportion, batch_size=args.batch_size, seed=args.seed, silent=args.silent)
