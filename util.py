"""util.py -- Various utility functions.
"""

import tensorflow as tf


def disable_gpu():
    try:
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except (AssertionError, ValueError, RuntimeError):
        raise RuntimeError('Unable to disable GPU')
