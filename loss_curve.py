import pickle

import matplotlib.pyplot as plt
import numpy as np


def loss_curve(path: str) -> None:
    with open(path, 'rb') as file:
        history = pickle.load(file)

    loss = history['loss']
    val_loss = history['val_loss']
    epochs = np.arange(len(loss)) + 1

    plt.figure()
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.grid()


if __name__ == '__main__':
    loss_curve('output/dense_model_gain_pca_100/history.pkl')
    plt.show()
