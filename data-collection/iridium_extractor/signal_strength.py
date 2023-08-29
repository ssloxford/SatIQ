"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class blk(gr.decim_block):
    def __init__(self, batch_length = 64000):
        gr.decim_block.__init__(self,
            name="Decimating Max",
            in_sig=[np.complex64],
            out_sig=[np.float32],
            decim = batch_length)

        self.set_relative_rate(1.0/batch_length)
        self.batch_length = batch_length

    def work(self, input_items, output_items):
        n_out_items = len(input_items[0]) // self.batch_length
        for i in range(0, n_out_items):
            output_items[0][i] = np.max([
                np.max(np.abs(input_items[0][i*self.batch_length:(i+1)*self.batch_length].real)),
                np.max(np.abs(input_items[0][i*self.batch_length:(i+1)*self.batch_length].imag))
            ])
        print(output_items)
        return len(output_items[0])
