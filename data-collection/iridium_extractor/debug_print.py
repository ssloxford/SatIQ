from gnuradio import gr
import pmt
import time


class blk(gr.basic_block):

    def __init__(self):

        gr.sync_block.__init__(
            self,
            name='Debug Printer',
            in_sig=None,
            out_sig=None
        )

        self.count_a = 0
        self.count_b = 0
        self.count_c = 0

        self.time_0 = time.time()
        self.time_1 = time.time()

        self.port_name_a = "pdus_a"
        self.port_name_b = "pdus_b"
        self.port_name_c = "pdus_c"

        self.message_port_register_in(pmt.intern(self.port_name_a))
        self.set_msg_handler(pmt.intern(self.port_name_a), self.handle_msg_a)

        self.message_port_register_in(pmt.intern(self.port_name_b))
        self.set_msg_handler(pmt.intern(self.port_name_b), self.handle_msg_b)

        self.message_port_register_in(pmt.intern(self.port_name_c))
        self.set_msg_handler(pmt.intern(self.port_name_c), self.handle_msg_c)

    def print_stats(self):
        time_diff = time.time() - self.time_0

        freq_a = format(self.count_a / time_diff, '.2f')
        freq_b = format(self.count_b / time_diff, '.2f')
        freq_c = format(self.count_c / time_diff, '.2f')

        print(self.count_a, self.count_b, self.count_c, freq_a, freq_b, freq_c)
        self.time_1 = time.time()

    def handle_msg_a(self, msg):
        self.count_a += 1
        
        if time.time() - self.time_1 > 1.0:
            self.print_stats()

    def handle_msg_b(self, msg):
        self.count_b += 1

        if time.time() - self.time_1 > 1.0:
            self.print_stats()

    def handle_msg_c(self, msg):
        self.count_c += 1

        if time.time() - self.time_1 > 1.0:
            self.print_stats()
