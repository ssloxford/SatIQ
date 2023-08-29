FROM ubuntu:focal

# Avoid questions in apt
ARG DEBIAN_FRONTEND=noninteractive

# Python tools
RUN apt update && apt install -y \
    python3-pip

# Install python packages
RUN pip3 install \
    pyzmq