"""qam_decode.py -- Decode QAM symbols from I/Q to binary.

MIT License

Copyright (c) 2020 Yi Zhang, Cisco Systems Inc. and The University of Texas at Austin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


# # TODO: Implement the QAM decoder without hard coding all the boundaries.
# def decode(iq: np.ndarray, mcs: int) -> np.ndarray:
#     ...


def decode(c, mcs):
    if mcs == 7:
        c = np.array(c)
        ret = np.zeros([c.shape[0], c.shape[-1] * 6], dtype=np.uint8)
        factor = np.sqrt(float(42.))
        i = np.arange(c.shape[-1])
        re = c[:, i].real * factor
        im = c[:, i].imag * factor
        ret[:, i*6+0] = re > 0
        ret[:, i*6+1] = (-np.abs(re) + 4) > 0
        ret[:, i*6+2] = (-np.abs((-np.abs(re) + 4)) + 2) > 0
        ret[:, i*6+3] = im > 0
        ret[:, i*6+4] = (-np.abs(im) + 4) > 0
        ret[:, i*6+5] = (-np.abs((-np.abs(im) + 4)) + 2) > 0
    elif mcs == 9:
        c = np.array(c)
        ret = np.zeros([c.shape[0], c.shape[-1] * 8], dtype=np.uint8)
        factor = np.sqrt(float(170.))
        i = np.arange(c.shape[-1])
        re = c[:, i].real * factor
        im = c[:, i].imag * factor
        ret[:, i*8+0] = re > 0
        ret[:, i*8+1] = (-np.abs(re) + 8) > 0
        ret[:, i*8+2] = (-np.abs((-np.abs(re) + 8)) + 4) > 0
        ret[:, i*8+3] = (-np.abs((-np.abs((-np.abs(re) + 8)) + 4)) + 2) > 0
        ret[:, i*8+4] = im > 0
        ret[:, i*8+5] = (-np.abs(im) + 8) > 0
        ret[:, i*8+6] = (-np.abs((-np.abs(im) + 8)) + 4) > 0
        ret[:, i*8+7] = (-np.abs((-np.abs((-np.abs(im) + 8)) + 4)) + 2) > 0
    elif mcs == 11:
        c = np.array(c)
        ret = np.zeros([c.shape[0], c.shape[-1] * 10], dtype=np.uint8)
        factor = np.sqrt(float(682.))
        i = np.arange(c.shape[-1])
        re = c[:, i].real * factor
        im = c[:, i].imag * factor
        ret[:, i*10+0] = re > 0
        ret[:, i*10+1] = (-np.abs(re) + 10) > 0
        ret[:, i*10+2] = (-np.abs((-np.abs(re) + 10)) + 8) > 0
        ret[:, i*10+3] = (-np.abs((-np.abs((-np.abs(re) + 10)) + 8)) + 4) > 0
        ret[:, i*10+4] = (-np.abs((-np.abs((-np.abs((-np.abs(re) + 10)) + 8)) + 4))+2) > 0
        ret[:, i*10+5] = im > 0
        ret[:, i*10+6] = (-np.abs(im) + 10) > 0
        ret[:, i*10+7] = (-np.abs((-np.abs(im) + 10)) + 8) > 0
        ret[:, i*10+8] = (-np.abs((-np.abs((-np.abs(im) + 10)) + 8)) + 4) > 0
        ret[:, i*10+9] = (-np.abs((-np.abs((-np.abs((-np.abs(im) + 10)) + 8)) + 4))+2) > 0
    else:
        raise RuntimeError('Illegal MCS')
    return ret
