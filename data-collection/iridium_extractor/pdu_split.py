from gnuradio import gr
import pmt


class blk(gr.basic_block):

    def __init__(self, num_outputs=2):

        gr.sync_block.__init__(
            self,
            name='Debug Printer',
            in_sig=None,
            out_sig=None
        )

        self.num_outputs = num_outputs

        self.state = 0

        self.port_name_in = "in"
        self.port_names_out = [ "out{}".format(i) for i in range(self.num_outputs) ]

        self.message_port_register_in(pmt.intern(self.port_name_in))
        self.set_msg_handler(pmt.intern(self.port_name_in), self.handle_msg)

        for port_name_out in self.port_names_out:
            self.message_port_register_out(pmt.intern(port_name_out))

    def handle_msg(self, msg):
        self.message_port_pub(pmt.intern(self.port_names_out[self.state]), msg)

        self.state = (self.state + 1) % self.num_outputs
