#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import zmq
from argparse import ArgumentParser
import time

class Database:
    def __init__(self, db_path, backup_path=None, verbose=0):
        self.db_path = db_path
        self.backup_path = backup_path
        self.verbose = verbose or 0

        self.conn = sqlite3.connect(self.db_path)
        self.__create_tables()
        self.run_id = self.__get_run_id() + 1


    # Create tables for iq samples and bytes, if they do not already exist
    def __create_tables(self):
        if self.verbose >= 1:
            print("Creating tables")

        self.conn.execute('''CREATE TABLE IF NOT EXISTS iq_samples
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            magnitude REAL NOT NULL,
            noise REAL NOT NULL,
            msg_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            timestamp_global INTEGER NOT NULL,
            uw_start REAL NOT NULL,
            direction INTEGER NOT NULL,
            center_frequency INTEGER NOT NULL,
            sample_rate INTEGER NOT NULL,
            sample_count INTEGER NOT NULL,
            samples BLOB NOT NULL)'''
        )

        self.conn.execute('''CREATE TABLE IF NOT EXISTS bytes
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            direction INTEGER NOT NULL,
            n_symbols INTEGER NOT NULL,
            magnitude REAL NOT NULL,
            noise REAL NOT NULL,
            level REAL NOT NULL,
            confidence INTEGER NOT NULL,
            msg_id INTEGER NOT NULL,
            center_frequency INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            timestamp_global INTEGER NOT NULL,
            msg_type TEXT NOT NULL,
            msg TEXT NOT NULL,
            bytes BLOB NOT NULL)'''
        )

        self.conn.execute('''CREATE TABLE IF NOT EXISTS ira_messages
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            msg_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            timestamp_global INTEGER NOT NULL,
            direction INTEGER NOT NULL,
            msg_type TEXT NOT NULL,
            msg TEXT NOT NULL,
            ra_lat REAL NOT NULL,
            ra_lon REAL NOT NULL,
            ra_alt REAL NOT NULL,
            ra_sat INTEGER NOT NULL,
            ra_cell INTEGER NOT NULL)'''
        )


    # Get the current max value of run_id in the database
    def __get_run_id(self):
        if self.verbose >= 1:
            print("Getting run_id")
        cursor = self.conn.execute('SELECT MAX(run_id) FROM iq_samples')
        return cursor.fetchone()[0] or 0


    # Write iq samples to database
    def write_iq_samples(self, iq_samples):
        iq_samples['run_id'] = self.run_id

        if self.verbose == 1:
            print("Received IQ samples")
        elif self.verbose == 2:
            iq_samples_copy = iq_samples.copy()
            del iq_samples_copy['samples']
            print("Received IQ samples:", iq_samples_copy)
        elif self.verbose > 3:
            print("Received IQ samples:", iq_samples)

        self.conn.execute('''INSERT INTO iq_samples
            (run_id, magnitude, noise, msg_id, timestamp, timestamp_global, uw_start, direction, center_frequency, sample_rate, sample_count, samples)
            VALUES (:run_id, :magnitude, :noise, :msg_id, :timestamp, :timestamp_global, :uw_start, :direction, :center_frequency, :sample_rate, :sample_count, :samples)''',
            iq_samples
        )
        self.conn.commit()


    # Write bytes to database
    def write_bytes(self, bytes):
        bytes['run_id'] = self.run_id

        if self.verbose == 1:
            print("Received bytes")
        elif self.verbose == 2:
            bytes_copy = bytes.copy()
            del bytes_copy['bytes']
            print("Received bytes:", bytes)
        elif self.verbose > 2:
            print("Received bytes:", bytes)

        self.conn.execute('''INSERT INTO bytes
            (run_id, direction, n_symbols, magnitude, noise, level, confidence, msg_id, center_frequency, timestamp, timestamp_global, msg_type, msg, bytes)
            VALUES (:run_id, :direction, :n_symbols, :magnitude, :noise, :level, :confidence, :msg_id, :center_frequency, :timestamp, :timestamp_global, :msg_type, :msg, :bytes)''',
            bytes
        )
        self.conn.commit()

    # Write IRA messages to database
    def write_ira_messages(self, ira_messages):
        ira_messages['run_id'] = self.run_id

        if self.verbose >= 1:
            print("    IRA message, writing to database")

        self.conn.execute('''INSERT INTO ira_messages
            (run_id, msg_id, timestamp, timestamp_global, direction, msg_type, msg, ra_lat, ra_lon, ra_alt, ra_sat, ra_cell)
            VALUES (:run_id, :msg_id, :timestamp, :timestamp_global, :direction, :msg_type, :msg, :ra_lat, :ra_lon, :ra_alt, :ra_sat, :ra_cell)''',
            ira_messages
        )
        self.conn.commit()


    def backup_database(self):
        if self.backup_path is not None:
            if self.verbose >= 1:
                print("Backing up database")
            backup = sqlite3.connect(self.backup_path)
            with backup:
                self.conn.backup(backup, pages=1)
            backup.close()

            self.conn.commit()
            self.conn.close()
            self.conn = sqlite3.connect(self.db_path)
        else:
            if self.verbose >= 1:
                print("No backup path specified, not backing up database")


# Buffer for IQ samples and messages
class Buffer:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.iq_buffer = []
        self.byte_buffer = []

    def add_iq_samples(self, iq_samples):
        self.iq_buffer.append(iq_samples)
        while len(self.iq_buffer) > self.buffer_size:
            self.iq_buffer.pop(0)

    def add_bytes(self, bytes):
        self.byte_buffer.append(bytes)
        while len(self.byte_buffer) > self.buffer_size:
            self.byte_buffer.pop(0)

    # Check if any pairs of iq_samples or bytes have the same msg_id, write to database if so
    def check(self):
        iq_sample_out = None
        bytes_out = None
        for iq_sample in self.iq_buffer:
            for bytes in self.byte_buffer:
                if iq_sample['msg_id'] == bytes['msg_id']:
                    iq_sample_out = iq_sample
                    bytes_out = bytes
                    break
            if iq_sample_out is not None:
                break
        if iq_sample_out is not None:
            self.iq_buffer.remove(iq_sample_out)
            self.byte_buffer.remove(bytes_out)
            return (iq_sample_out, bytes_out)
        else:
            return None


class ZMQInterface:
    def __init__(self, verbose, db_path, backup_path, receive_messages, send_messages, zmq_address_iq, zmq_address_bytes, zmq_address_send, keep_malformed, filter_matching, filter_ira, buffer_size):
        self.verbose = verbose or 0
        # Verbosity levels:
        # 0: No output
        # 1: Alert on new samples and bytes
        # 2: Alert on new samples and bytes, helper functions
        # 3: Alert on new samples and bytes, helper functions, full iq_samples and bytes
        self.db_path = db_path
        self.backup_path = backup_path
        self.receive_messages = receive_messages or False
        self.send_messages = send_messages or False
        self.zmq_address_iq = zmq_address_iq
        self.zmq_address_bytes = zmq_address_bytes
        self.zmq_address_send = zmq_address_send
        self.keep_malformed = keep_malformed
        self.filter_matching = filter_matching
        self.filter_ira = filter_ira
        self.buffer_size = buffer_size

        self.db = Database(db_path, backup_path=backup_path, verbose=verbose)
        self.buffer = Buffer(buffer_size)

        # Connect to ZeroMQ
        self.context = zmq.Context()
        self.socket_iq = self.context.socket(zmq.PULL)
        self.socket_iq.bind(zmq_address_iq)
        self.socket_bytes = self.context.socket(zmq.PULL)
        self.socket_bytes.bind(zmq_address_bytes)

        # Set up ZeroMQ poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket_iq, zmq.POLLIN)
        self.poller.register(self.socket_bytes, zmq.POLLIN)

        self.count = 0
        self.count_total = 0
        self.count_iq = 0
        self.count_bytes = 0
        self.time_0 = time.time()
        self.time_1 = time.time()

    # Return True if the message should be saved
    def __check_filters(self, iq_samples, bytes):
        if self.filter_ira:
            return bytes['msg_type'] == 'RA'
        if not self.keep_malformed:
            return iq_samples['uw_start'] >= 0
        return True

    # Receive and write to database (no filtering)
    def __process_messages_simple(self):
        # Poll ZeroMQ
        socks = dict(self.poller.poll())

        # Process IQ samples
        if self.socket_iq in socks and socks[self.socket_iq] == zmq.POLLIN:
            iq_samples = self.socket_iq.recv_json()
            self.db.write_iq_samples(iq_samples)

        # Process bytes
        if self.socket_bytes in socks and socks[self.socket_bytes] == zmq.POLLIN:
            bytes = self.socket_bytes.recv_json()
            self.db.write_bytes(bytes)
            if bytes.get('msg_type') == 'RA':
                self.db.write_ira_messages(bytes)

    # Receive and write to database (with filtering)
    def __process_messages_filtered(self):
        # Poll ZeroMQ
        socks = dict(self.poller.poll())

        # Process IQ samples
        if self.socket_iq in socks and socks[self.socket_iq] == zmq.POLLIN:
            iq_samples = self.socket_iq.recv_json()
            self.buffer.add_iq_samples(iq_samples)
            self.count_iq += 1

        # Process bytes
        if self.socket_bytes in socks and socks[self.socket_bytes] == zmq.POLLIN:
            bytes = self.socket_bytes.recv_json()
            self.buffer.add_bytes(bytes)
            self.count_bytes += 1

        # Check if any pairs of iq_samples or bytes have the same msg_id, write to database if so
        iq_samples_bytes = self.buffer.check()
        if iq_samples_bytes:
            iq_samples, bytes = iq_samples_bytes
            self.count_total += 1
            if self.__check_filters(iq_samples, bytes):
                self.count += 1
                self.db.write_iq_samples(iq_samples)
                self.db.write_bytes(bytes)
                if bytes.get('msg_type') == 'RA':
                    self.db.write_ira_messages(bytes)

        if time.time() - self.time_1 > 1.0:
            self.print_debug()

    def print_debug(self):
        rate = self.count / (time.time() - self.time_0)
        rate_total = self.count_total / (time.time() - self.time_0)
        rate_ratio = rate / rate_total if rate_total != 0 else 0
        print("{} {} {}, {:.2f} {:.2f} (ratio {:.2f})".format(self.count_iq, self.count_bytes, self.count, rate_total, rate, rate_ratio))
        self.time_1 = time.time()

    def process_messages(self):
        if self.filter_matching:
            self.__process_messages_filtered()
        else:
            self.__process_messages_simple()

    def backup_database(self):
        self.db.backup_database()


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('-d', '--database', dest='database', required=True, help='Database file')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', help='Verbose output')
    parser.add_argument('-r', '--receive-messages', dest='receive_messages', action='store_true', help='Receive ZeroMQ messages using addresses zmq-address-iq and zmq-address-bytes')
    parser.add_argument('-s', '--send-messages', dest='send_messages', action='store_true', help='Send ZeroMQ messages using address zmq-address-send')
    parser.add_argument('--zmq-address-iq', dest='zmq_address_iq', required=False, help='ZeroMQ address for iq samples')
    parser.add_argument('--zmq-address-bytes', dest='zmq_address_bytes', required=False, help='ZeroMQ address for bytes')
    parser.add_argument('--zmq-address-send', dest='zmq_address_send', required=False, help='ZeroMQ address for sending messages')
    parser.add_argument('--database-backup', dest='database_backup', type=str, default=None, help='Path to database backup')
    parser.add_argument('--backup-interval', dest='backup_interval', type=int, default=3600, help='Backup interval in seconds')
    parser.add_argument('-k', '--keep-malformed', dest='keep_malformed', action='store_true', help='Keep malformed samples')
    parser.add_argument('--filter-matching', dest='filter_matching', action='store_true', help='Keep only samples which match valid messages')
    parser.add_argument('--filter-ira', dest='filter_ira', action='store_true', help='Keep only IRA messages')
    parser.add_argument('--buffer-size', dest='buffer_size', type=int, default=10, help='Buffer size when filtering to keep only IRA messages')
    return parser.parse_args()


def main():
    args = argument_parser()

    zmq_interface = ZMQInterface(
        args.verbose,
        args.database,
        args.database_backup,
        args.receive_messages,
        args.send_messages,
        args.zmq_address_iq,
        args.zmq_address_bytes,
        args.zmq_address_send,
        args.keep_malformed,
        args.filter_matching,
        args.filter_ira,
        args.buffer_size
    )

    last_backup_time = time.time()
    backup_interval = args.backup_interval

    while True:
        # If backup_interval has passed, backup the database
        if time.time() - last_backup_time > backup_interval:
            zmq_interface.backup_database()
            last_backup_time = time.time()

        zmq_interface.process_messages()


if __name__ == "__main__":
    main()
