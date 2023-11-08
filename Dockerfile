#FROM tensorflow/tensorflow:latest
FROM tensorflow/tensorflow:latest-gpu

# avoid questions when installing things in apt-get
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install git wget nano

RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install pandas keras h5py zmq
RUN pip3 install tqdm
RUN pip3 install tensorflow-datasets

RUN pip install tensorflow-addons

RUN pip install scipy
RUN pip install scikit-learn
RUN pip install notebook

RUN pip install seaborn

WORKDIR /
