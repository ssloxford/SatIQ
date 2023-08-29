from gnuradio import gr
import pmt
import zmq
import time
from iridium_toolkit import bitsparser


class blk(gr.basic_block):

    def __init__(self, zmq_address, debug=False):

        gr.sync_block.__init__(
            self,
            name='PDU Byte Sink',
            in_sig=None,
            out_sig=None
        )

        self.debug = debug
        self.count = 0

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(zmq_address)

        self.port_name = "pdus"
        self.message_port_register_in(pmt.intern(self.port_name))
        self.set_msg_handler(pmt.intern(self.port_name), self.handle_msg)


    def get_msg_data(self, iridium_message):
        output = dict()
        output['msg'] = iridium_message.pretty()
        output['msg_type'] = iridium_message.msgtype
        if iridium_message.msgtype == 'RA':
            print(":)")
            output['ra_lat'] = float(iridium_message.ra_lat)
            output['ra_lon'] = float(iridium_message.ra_lon)
            output['ra_alt'] = float(iridium_message.ra_alt)
            output['ra_sat'] = int(iridium_message.ra_sat)
            output['ra_cell'] = int(iridium_message.ra_cell)
        return output


    def decode_msg(self, msg):
        timestamp = float(msg['timestamp']) / 1000000
        timestamp_global = int(msg['timestamp_global'])
        frequency = int(msg['center_frequency'])
        snr = float(msg['magnitude'])
        noise = float(msg['noise'])
        id = int(msg['msg_id'])
        confidence = int(msg['confidence'])
        level = float(msg['level'])
        bitstream = msg['bytes']

        message_interface = bitsparser.MessageInterface(timestamp, timestamp_global, frequency, snr, noise, id, confidence, level, bitstream)
        message_interface_upgrade = message_interface.upgrade()
        if message_interface.error:
            if self.debug:
                print("Decoding errors:")
                for e in message_interface.error_msg:
                    print("    {}".format(e))
            del message_interface_upgrade
            del message_interface
            return dict(
                success=False,
                msg = "",
                msg_type = "",
            )
        else:
            if self.debug:
                print(message_interface_upgrade.pretty())
            output = self.get_msg_data(message_interface_upgrade)
            output['success'] = True
            del message_interface_upgrade
            del message_interface
            return output


    def handle_msg(self, msg):
        msg = pmt.to_python(msg)
        data = msg[0]
        bytes = msg[1]
        if self.debug:
            print("Received message:", msg)

        data['msg_id'] = data.pop('id')
        data['bytes'] = "".join([str(b) for b in bytes])
        data['timestamp_global'] = int(time.time_ns())

        msg_decoded = self.decode_msg(data)
        # Combine data with decoded message
        data = {**data, **msg_decoded}

        if self.debug:
            print("Sending:", data)

        self.socket.send_json(data)
