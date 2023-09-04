# SatIQ

This repository contains all the data collection, model training, and analysis code for the `SatIQ` system, described in the paper "Watch This Space: Securing Satellite Communication through Resilient Transmitter Fingerprinting".
This system can be used to authenticate Iridium satellite transmitters using high sample rate message headers.

The full dataset can be found at the following URL: https://zenodo.org/record/8220494

The trained model weights can be found at the following URL: https://zenodo.org/record/8298532

When using this code, please cite the following paper: "Watch This Space: Securing Satellite Communication through Resilient Transmitter Fingerprinting".
The BibTeX entry is given below:
```
@inproceedings{smailesWatch2023,
  author = {Smailes, Joshua and K{\"o}hler, Sebastian and Birnbach, Simon and Strohmeier, Martin and Martinovic, Ivan},
  title = {{Watch This Space}: {Securing Satellite Communication through Resilient Transmitter Fingerprinting}},
  year = {2023},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
  location = {Copenhagen, Denmark},
  series = {CCS '23}
}
```


## Setup

To clone the repository:
```bash
git clone --recurse-submodules https://github.com/ssloxford/SatIQ.git
cd SatIQ
```

A Docker container is provided for ease of use, with all dependencies installed.
A recent version of Docker must be installed on your system to use this.

To run scripts locally, the following packages are required:
```
python3
```

The following Python packages are also required:
```
numpy
matplotlib
pandas
keras
h5py
zmq
tqdm
tensorflow
tensorflow-datasets
tensorflow-addons==0.13.0
scipy
seaborn
```

A GPU is recommended (with all necessary drivers installed), and a moderate amount of RAM will be required to run the data preproccessing and model training.


## Usage

### TensorFlow Container

The script `tf-container.sh` provides a Docker container with the required dependencies for data processing, model training, and the analysis code.
Run the script from inside the repository's root directory to ensure volumes are correctly mounted.

If your machine has no GPUs:
- Modify `Dockerfile` to use the `tensorflow/tensorflow:latest` image.
- Modify `tf-container.sh`, removing `--gpus all`.


### SatIQ


The `util` directory contains the main data processing and model code:
- `data.py` contains utilities for data loading and preprocessing.
- `models.py` contains the main model code.
- `model_utils.py` contains various helper classes and functions used during model construction and training.

See the data collection, training, and analysis scripts for examples on how to use these files.


### Data Collection

The `data-collection` directory contains a `docker-compose` pipeline to receive signals from an SDR, extract Iridium messages, and save the data to a database file.
To run under its default configuration, connect a USRP N210 via Ethernet to the host machine, and run the following (from inside the `data-collection` directory:

```bash
docker-compose up
```

Data will be stored in `data/db.sqlite3`.

If a different SDR is used, the `iridium_extractor` configuration may need to be altered.
Change the `docker-compose.yml` to ensure the device is mounted in the container, and modify `iridium_extractor/iridium_extractor.py` to use the new device as a source.


### Data Preprocessing

The scripts in the `preprocessing` directory process the database file(s) into NumPy files, and then TFRecord datasets.
It is recommended to run these scripts from within the TensorFlow container described above.

Please note that these scripts load the full datasets into memory, and will consume large amounts of RAM.
It is recommended that you run them on a machine with at least 128GB of RAM.

#### db-to-np-multiple.py

This script extracts the database files into NumPy files.
To run, adjust `path_base` if appropriate (this should point to your `data` directory), and `db_indices` to point to the databases that need extracting.

The script itself runs with no arguments:
```bash
python3 db-to-np-multiple.py
```

The resulting files will be placed in `code/processed` (ensure this directory already exists).

#### np-filter.py

This script normalizes the IQ samples, and filters out unusable data.
To run, once again adjust `path_base` if appropriate, and set `suffixes` to the NumPy suffixes that need filtering -- this will likely be the same as `db_indices` from the previous step.

The script runs with no arguments:
```bash
python3 np-filter.py
```

The resulting files will be placed in `code/filtered` (ensure this directory already exists).

#### np-to-tfrecord.py

This script converts NumPy files into the TFRecord format, for use in model training.
To run, `path_base` and `suffixes` are once again set as above.
The `chunk_size` `shuffle`, `by_id`, and `id_counts` options may also be set to adjust how the dataset is generated -- the default options should be fine, unless alternative datasets (e.g. with transmitters removed) are required.

The script runs with no arguments:
```bash
python3 np-filter.py
```

The resulting files will be placed in `code/tfrecord` (ensure this directory already exists).

Please note that this script in particular will use a large amount of RAM.

#### Noise

The `noise` directory contains modified versions of the above scripts that filter the dataset to remove the messages with the highest noise.
Use in the same way as above.

Ensure that all the requisite directories have been created before these scripts are executed.


### Model Training

The script for training the `SatIQ` model is found in `code/training/ae-triplet-conv.py`.
Ensure that data is placed in the `data` directory before running.

Adjust the arguments at the top of the script to ensure the data and output directories are set correctly (these should be fine if running inside the TensorFlow Docker container), then run the script with no arguments:
```bash
python3 ae-triplet-conv.py
```

This will take a long time.
The checkpoints should appear in `data/models`.


### Analysis

The `analysis` directory contains Jupyter notebooks for loading the trained models, processing the data, and producing the plots and numbers used in the paper.
The notebook may be opened without running to see the results in context, or executed to reproduce the results.

The TensorFlow Docker container should contain all the required dependencies to run the notebooks.
See [Setup](#Setup) for requirements to run outside docker.

Note that these also require a large amount of RAM, and a GPU is recommended in order to run the models.

The `plots-data.ipynb` notebook contains plots relating to the raw samples.

The `plots-models.ipynb` notebook contains all the analysis of the trained models.


## Contribute

This code, alongside the datasets and trained models, has been made public to aid future research in this area.
However, this reposistory is no longer actively developed.
Any contributions (documentation, bug fixes, etc.) should be made as pull requests, and may be accepted.

