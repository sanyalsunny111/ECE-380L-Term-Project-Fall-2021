"""animate_weights.py -- Visualize weights over multiple steps.
"""

import glob
import math
import pathlib
import re

import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def animate(title, images):
    figure = plt.figure()
    plt.title(title)
    plt.xlabel('Input Space')
    plt.ylabel('Output Space')

    print(f'Animation has {len(images)} frames.')

    frames = []
    for image in images:
        frame = plt.imshow(image, animated=True, cmap='inferno', aspect=600)
        frames.append([frame])

    a = animation.ArtistAnimation(figure, frames, interval=500, repeat=False)
    a.save('animation.gif', writer='pillow')
    plt.show()


def main():
    basedir = 'output/dense_model_gain_regularized_l1_l2_test_2'
    weights = h5py.File(f'{basedir}/weights_epoch_1.h5', 'r')
    layers = [layer for layer in weights.keys() if 'dense' in layer]

    for layer in layers:
        images = []
        paths = list(glob.glob(f'{basedir}/weights_epoch_*.h5'))

        def get_order(file):
            pattern = re.compile(r'.*?(\d+).*?')
            match = pattern.match(pathlib.Path(file).name)
            if not match:
                return math.inf
            return int(match.groups()[0])

        paths = sorted(paths, key=get_order)

        for path in paths:
            print(pathlib.Path(path).name)
            weights = h5py.File(path, 'r')
            kernel = weights[layer][layer]['kernel:0'][()]
            if 'conv1d' not in layer:
                kernel = kernel.T
            kernel = np.sqrt(np.sum(np.abs(kernel) ** 2, axis=0).reshape((1, 708)))
            images.append(abs(kernel))

        animate(layer, images)
        break


if __name__ == '__main__':
    main()
