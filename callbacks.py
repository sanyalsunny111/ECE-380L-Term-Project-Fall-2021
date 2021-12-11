"""callbacks.py -- Various callbacks.
"""

import math
import signal
import types

import h5py
import numpy as np
import tensorflow as tf

from qam_decode import decode


class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, path: str, tone_start: int, tone_end: int, batch_size: int):
        super().__init__()
        self._tone_start = tone_start
        self._tone_end = tone_end
        self._batch_size = batch_size

        data = h5py.File(path, 'r')

        self._x = [
            np.array(data['rx_l_ltf_1']),
            np.array(data['rx_l_ltf_2']),
            np.array(data['rx_he_ltf_data']),
            np.array(data['rx_he_ltf_pilot']),
            np.array(data['rx_data'][:, self._tone_start:self._tone_end]),
            np.array(data['rx_pilot']),
            np.array(data['tx_pilot'])
        ]

        self._y = np.array(data['tx_data'][:, self._tone_start:self._tone_end])
        self._bits = decode(self._y, 7)

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        # TODO: Keep a running number of bit error instead if we have memory issues.
        y_hat = np.zeros_like(self._y)
        for i in range(math.ceil(len(y_hat) / self._batch_size)):
            indices = np.arange(self._batch_size * i, min(self._batch_size * (i + 1), len(y_hat)))
            y_hat[indices] = self.model.predict([self._x[j][indices] for j in range(len(self._x))])

        bits_hat = decode(y_hat, 7)
        logs['ber'] = np.mean(self._bits != bits_hat)


# TODO: Why doesn't this work all the time?
class SignalHandlerCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_dir: str, save_on_exit: bool = False):
        super().__init__()
        self._output_dir = output_dir
        self._save_on_exit = save_on_exit
        self._exit = False

        signal.signal(signal.SIGINT, self.handle_signal)

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        if self._exit:
            if self._save_on_exit:
                path = f'{self._output_dir}/weights_abort.hdf5'
                self.model.save_weights(path)
            self.model.stop_training = True

    def handle_signal(self, _signals: signal.Signals, _frame: types.FrameType) -> None:
        self._exit = True
