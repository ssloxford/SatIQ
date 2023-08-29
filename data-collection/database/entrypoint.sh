#!/bin/sh

exec /code/database/database.py --database $DB_FILE --zmq-address-iq $ZMQ_ADDRESS_IQ --zmq-address-bytes $ZMQ_ADDRESS_BYTES ${ARGS} "$@"
