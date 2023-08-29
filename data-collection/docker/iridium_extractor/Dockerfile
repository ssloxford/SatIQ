FROM ubuntu:focal

# Avoid questions in apt
ARG DEBIAN_FRONTEND=noninteractive

# Base system stuff
RUN apt update && apt install -y \
    software-properties-common \
    git \
    build-essential \
    cmake \
    nano \
    mousepad \
    wget \
    python3-pip

# GNURadio Dependencies
RUN apt update && apt install -y \
    git \
    cmake \
    g++ \
    libboost-all-dev \
    libgmp-dev \
    swig \
    python3-numpy \
    python3-mako \
    python3-sphinx \
    python3-lxml \
    doxygen \
    libfftw3-dev \
    libsdl1.2-dev \
    libgsl-dev \
    libqwt-qt5-dev \
    libqt5opengl5-dev \
    python3-pyqt5 \
    liblog4cpp5-dev \
    libzmq3-dev \
    python3-yaml \
    python3-click \
    python3-click-plugins \
    python3-zmq \
    python3-scipy \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    libcodec2-dev \
    libgsm1-dev \
    pybind11-dev \
    python3-matplotlib \
    libsndfile1-dev \
    python3-pip \
    libsoapysdr-dev \
    soapysdr-tools \
    libiio-dev \
    libad9361-dev \
    libspdlog-dev \
    python3-packaging \
    python3-jsonschema

RUN pip3 install pygccxml
RUN pip3 install pyqtgraph

# Other dependencies
RUN apt update && apt install -y \
    gobject-introspection \
    liborc-0.4-dev \
    libuhd-dev \
    libuhd3.15.0 \
    uhd-host \
    ccache

RUN apt update && apt install -y \
    libudev-dev \
    qtbase5-dev \
    qt5-default \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    python3-pygccxml \
    python3-pybind11 \
    hackrf

RUN apt update && apt install -y \
    libsndfile1 \
    libsndfile1-dev \
    libspdlog-dev \
    libspdlog1

# Install GNURadio
RUN add-apt-repository ppa:gnuradio/gnuradio-releases && \
    apt update && \
    apt install -y gnuradio

# Download, build, and install gr-iridium
WORKDIR /
RUN mkdir gr-iridium && wget -qO- https://github.com/jsmailes/gr-iridium/archive/refs/tags/v1.0.1.tar.gz | tar -xz -C gr-iridium --strip-components=1
WORKDIR gr-iridium
RUN mkdir build
WORKDIR build
RUN cmake ..
RUN make -j $(nproc --all)
RUN make install
RUN ldconfig

RUN uhd_images_downloader

# Install miscellaneous python dependencies
RUN pip3 install crcmod

# Set python path correctly so we can import the new modules
ENV PYTHONPATH="/usr/local/lib/python3/dist-packages:/usr/local/lib/python3/site-packages:/usr/local/lib64/python3/site-packages"
RUN echo "export PYTHONPATH=/usr/local/lib/python3/dist-packages:/usr/local/lib/python3/site-packages:/usr/local/lib64/python3/site-packages" >> /root/.bashrc
RUN echo "export PATH=/usr/lib/ccache:${PATH}" >> /root/.bashrc

ENTRYPOINT [ "/bin/bash" ]
