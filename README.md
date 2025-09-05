# SatIQ

This repository contains all the data collection, model training, and analysis code for the `SatIQ` system, described in the paper "Watch This Space: Securing Satellite Communication through Resilient Transmitter Fingerprinting".
This system can be used to authenticate Iridium satellite transmitters using high sample rate message headers.

> [!NOTE]
> This version of the repository contains the code used for the paper "SatIQ: Extensible and Stable Satellite Authentication using Hardware Fingerprinting".
> The code used for "Watch This Space: Securing Satellite Communication through Resilient Transmitter Fingerprinting" can be found [here](https://github.com/ssloxford/SatIQ/tree/v1.0.2).

Additional materials (SatIQ):
- "SatIQ" paper: https://www.cs.ox.ac.uk/files/14805/main.pdf
- Full dataset (UK): https://doi.org/10.7910/DVN/P5FUAW
- Full dataset (Germany): https://doi.org/10.7910/DVN/RXWV1M
- Full dataset (Switzerland): https://doi.org/10.7910/DVN/OSSJ68
- Trained model weights: https://doi.org/10.7910/DVN/GANMDZ

Additional materials (Watch This Space):
- "Watch This Space" paper: https://arxiv.org/abs/2305.06947
- Full dataset: https://zenodo.org/record/8220494
- Trained model weights: https://zenodo.org/record/8298532

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
scikit-learn
notebook
```

A GPU is recommended (with all necessary drivers installed), and a moderate amount of RAM will be required to run the data preproccessing and model training.


### Downloading Data (SatIQ)

The full dataset for "SatIQ" is stored on the Harvard Dataverse at the following URL: https://dataverse.harvard.edu/dataverse/satiq.

This includes three datasets for each of the three locations (UK, Germany, Switzerland), and trained model weights.

These can be downloaded from the site directly, but the following script may be preferable due to the large file size:
```bash
TODO https://eamonnbell.webspace.durham.ac.uk/2023/03/07/bulk-downloading-from-dataverse/
```

> [!WARNING]
> The files are very large (approximately 1TB total).
> Ensure you have enough disk space before downloading.

### Downloading Data (Watch This Space)

The full dataset for "Watch This Space" is stored on Zenodo at the following URL: https://zenodo.org/record/8220494.

These can be downloaded from the site directly, but the following script may be preferable due to the large file size:
```bash
#!/bin/bash

for i in $(seq -w 0 5 165); do
  printf -v j "%03d" $((${i#0} + 4))
  wget https://zenodo.org/records/8220494/files/data_${i}_${j}.tar.gz
done
```

> [!WARNING]
> These files are very large (4.0GB each, 135.4GB total).
> Ensure you have enough disk space before downloading.

To extract the files:
```bash
#!/bin/bash

for i in $(seq -w 0 5 165); do
  printf -v j "%03d" $((${i#0} + 4))
  tar xzf data_${i}_${j}.tar.gz
done
```

See the instructions below on processing the resulting files for use.


### Directory Structure

The training and analysis scripts expect the repository to be laid out as follows:
```
SatIQ
├── ...
└── data
    ├── models
    │   ├── downsample
    │   │   └── ...
    │   └── ...
    ├── tfrecord
    │   ├── ...
    │   ├── germany
    │   │   └── ...
    │   ├── switzerland
    │   │   └── ...
    │   └── uk-switzerland
    └── test
        ├── embeddings
        │   └── ...
        └── labels
            └── ...
```

Any downloaded model/loss files with `downsample` in the name should be placed in `data/models/downsample`, and any other model files should be placed in `data/models`.

The `uk-switzerland` directory can be populated using `preprocessing/dataset-combine.sh`, and the `embeddings` and `labels` directories using `preprocessing/generate-embeddings.py`.
These are described in greater detail below.


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
- `processing.py` contains utilities for processing and analysis of results.
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

The `autorun.sh` and `restart.sh` scripts are provided for convenience, in order to automate the process of stopping the container and moving the resulting database files to a permanent storage location.


### Data Preprocessing

The scripts in the `preprocessing` directory process the database file(s) into NumPy files, and then TFRecord datasets.
It is recommended to run these scripts from within the TensorFlow container described above.

> [!NOTE]
> Converting databases to NumPy files and filtering is only necessary if you are doing your own data collection.
> If the "SatIQ" dataset is used, no preprocessing is required.
> If the "Watch This Space" dataset is used, only the `np-to-tfrecord.py` script is required.

> [!IMPORTANT]
> Please note that these scripts load the full datasets into memory, and will consume large amounts of RAM.
> It is recommended that you run them on a machine with at least 128GB of RAM.


#### db-to-tfrecord.py

This script extracts database files and processes them directly into TFRecord files, optionally adding weather data if provided.
This should be used preferentially over the legacy scripts described below.
To run this script, use the command-line arguments as directed by the script itself.

```bash
python3 db-to-tfrecord.py --help
```


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
To run this script, ensure your data has been processed into NumPy files with the following format:
- `samples_<suffix>.npy`
- `ra_sat_<suffix>.npy`
- `ra_cell_<suffix>.npy`

> [!NOTE]
> The `db-to-np-multiple.py` script will produce files in this format.
> The dataset available from Zenodo is also in this format.

The script can be used as follows:
```bash
python3 np-to-tfrecord.py --path-in <INPUT PATH> --path-out <OUTPUT PATH>
```

There are also the following optional parameters:
- `--chunk-size <CHUNK SIZE>`: number of records in each chunk. Default is 50000, set to a smaller value for smaller files.
- `-v`, `--verbose`: display progress.
- `--max-files <MAX FILES>`: stop after processing the specified number of input files.
- `--skip-files <SKIP FILES>`: skip a specified number of input files.
- `--no-shuffle`: do not shuffle the data.
- `--by-id`: see below.

The `by_id` option creates 9 datasets.
The first of these contains only the most common 10% of transmitter IDs.
The second contains 20%, and so on.
Be careful using this option, as it creates a much larger number of files, and takes significantly longer to run.

> [!WARNING]
> This script in particular will use a large amount of RAM, since it loads the entire dataset into memory at once.
> Processing may be done in batches by using the `--max-files` and `--skip-files` command-line arguments, or the script below.

#### np-to-tfrecord-parallel.sh

This script can run multiple instances of `np-to-tfrecord.py` in parallel, allowing preprocessing to be sped up and/or less RAM to be used.

Usage:
```bash
np-to-tfrecord-parallel.sh <NUM PROCESSES> <FILES PER PROCESS> <INPUT PATH> <OUTPUT PATH>
```
Where:
- `INPUT PATH` contains your `.npy` files, as above.
- `OUTPUT PATH` is the desired output directory.
- `NUM PROCESSES` is the number of CPU cores to use.
- `FILES PER PROCESS` is the number of files each thread should load at once.

Ensure that `NUM_PROCESSES * FILES_PER_PROCESS` input files can fit comfortably in RAM.

> [!NOTE]
> Shuffling is disabled by default in this script - if shuffled data is desired, the `--no-shuffle` flag should be removed from the script.
> If this flag is removed, shuffling will only be done on a per-process level - that is, each process will shuffle the files it has loaded, but not the dataset as a whole.


#### sqlite3-compress.py

This script converts database files directly into the NumPy arrays in the same format as provided in the Zenodo dataset.
This includes all columns provided by the data collection pipeline.

The script can be used as follows:
```bash
python3 sqlite3-compress.py <INPUT PATH> <OUTPUT PATH>
```


#### dataset-combine.sh

This script builds the combined UK-Switzerland dataset out of the two separate datasets by linking files.
This script has no configuration options and is used as follows:
```bash
./dataset-combine.sh
```


#### generate-embeddings.py

This script takes a trained model (or multiple models) and generates the embeddings of the dataset produced by that model, to enable faster analysis.
To use this script, modify `data_dir`, `model_dir`, and `output_dir` to point to the relevant input/output directories, and ensure `model_names` contains the correct names of the models from which the embeddings should be generated.
The script should then be run with no arguments
```bash
python3 generate-embeddings.py
```


#### Noise

> [!NOTE]
> These scripts are only used for "Watch This Space", and should not be used with the "SatIQ" data.

The `noise` directory contains modified versions of the above scripts that filter the dataset to remove the messages with the highest noise.
Use in the same way as above.

Ensure that all the requisite directories have been created before these scripts are executed.


### Model Training

The scripts for model training can be found in the `training` directory.
Ensure that data is placed in the `data` directory before running.
The `ae-triplet-conv-dataset-slices.py` script is used to train models from "SatIQ", and `ae-triplet-conv` to train models from "Watch This Space".
Additionally, `train-___.sh` scripts are provided as examples for training multiple models sequentially under different configurations.

Adjust the arguments at the top of the script to ensure the data and output directories are set correctly (these should be fine if running inside the TensorFlow Docker container), then run the script with no arguments:
```bash
python3 ae-triplet-conv-dataset-slices.py
```

Additional command-line arguments can be used to adjust characteristics of the model.

Training will take a long time.
The checkpoints will appear in `data/models`.


### Analysis

The `analysis` directory contains Jupyter notebooks for loading the trained models, processing the data, and producing the plots and numbers used in the paper.
The notebook may be opened without running to see the results in context, or executed to reproduce the results.

The TensorFlow Docker container should contain all the required dependencies to run the notebooks.
See [Setup](#Setup) for requirements to run outside docker.

Note that these also require a large amount of RAM, and a GPU is recommended in order to run the models.

The `satiq-data.ipynb` notebook contains plots relating to the raw samples.

The `satiq-models.ipynb` notebook contains all the analysis of the trained models.

> [!NOTE]
> The `past-ai-___.pdf` plots require access to the dataset from the paper "PAST-AI: Physical-layer authentication of satellite transmitters via deep learning".
> Please contact the authors of this paper for access if needed.

> [!NOTE]
> The `wts-data.ipynb` and `wts-models.ipynb` are also included for legacy purposes -- these are the equivalent analysis scripts from "Watch This Space".


## Contribute

This code, alongside the datasets and trained models, has been made public to aid future research in this area.
However, this reposistory is no longer actively developed.
Any contributions (documentation, bug fixes, etc.) should be made as pull requests, and may be accepted.

