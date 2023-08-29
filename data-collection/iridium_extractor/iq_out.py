from gnuradio import gr
import pmt
import zmq
import base64
import time


class blk(gr.basic_block):

    def __init__(self, zmq_address, num_samples=None, debug=False):

        gr.sync_block.__init__(
            self,
            name='PDU IQ Sink',
            in_sig=None,
            out_sig=None
        )

        self.debug = debug
        self.count = 0

        self.num_samples = num_samples

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(zmq_address)

        self.port_name = "pdus"
        self.message_port_register_in(pmt.intern(self.port_name))
        self.set_msg_handler(pmt.intern(self.port_name), self.handle_msg)

    def handle_msg(self, msg):
        msg = pmt.to_python(msg)
        data = msg[0]
        iq = msg[1]

        if self.num_samples is not None:
            iq = iq[:self.num_samples]

        if self.debug:
            pass #print("Received message:", msg)

        data['msg_id'] = data.pop('id')
        data['sample_count'] = iq.size
        iq_b64 = base64.b64encode(iq).decode('UTF-8')
        data['samples'] = iq_b64
        data['timestamp_global'] = int(time.time_ns())

        if self.debug:
            data_copy = data.copy()
            del data_copy['samples']
            print("Sending:", data_copy)

        self.socket.send_json(data)
