# Set Interdependence Transformer
This repository contains all code and links to datasets required for repeated experiments. The Set Interdependence Transformer (SIT) is a modularized set encoder, lending itself particularly well to permutation learning and structure prediction challenges. It uses attention mechanisms to learn a permutation invariant representation of the input set as well as a permutation equivariant representation of its elements. These representations are learned concurrently, such that higher-order dependencies between them can be learned in a single layer of SIT.

## Setup

The following section explains how to obtain the freely available datasets and prepare the virtual environment.

### Data
Two datasets need to be downloaded prior to running the experiments, available under the following links:
    
- [ROCStory](https://cs.rochester.edu/nlp/rocstories) sentence ordering dataset.
- [PROCAT](https://doi.org/10.6084/m9.figshare.14709507) catalog structure prediction dataset.

The PROCAT dataset requires additional preprocessing as provided in the `procat_preprocess_for_bert.py` file, which produces 3 csv files. The final directory tree should look like this:

```
|-- README.md
|-- danish-bert-botxo
|-- data
|   |-- PROCAT
|   |   |-- PROCAT.test.csv
|   |   |-- PROCAT.train.csv
|   |   |-- PROCAT.validation.csv
|   |   |-- procat_dataset_loader.py
|   |   `-- procat_preprocess_for_bert.py
|   `-- rocstory
|       |-- ROCStory.test.csv
|       |-- ROCStory.train.csv
|       |-- ROCStory.validation.csv
|       `-- rocstory_ordering.py
|-- grammars.py
|-- logs
|-- metrics.py
|-- model_configs
|   |-- grammar.json
|   |-- procat.json
|   |-- sentence_ordering.json
|   |-- synthetic.json
|   `-- tsp.json
|-- models
|-- models.py
|-- plots
|-- procat.py
|-- requirements.txt
|-- scratchpad.py
|-- sentence_ordering.py
|-- synthetic.py
|-- tsp.py
|-- utils.py
`-- utils_data.py
```

Both PROCAT and ROCStory experiments employ a language specific version for the BERT model. The PROCAT model is available for download [here](https://github.com/certainlyio/nordic_bert),
and should be placed in its `danish-bert-botxo` directory.

### Environment

The code requires **python 3.6.5**. All requirements are listed in the `requirements.txt` and can be installed in the following way (on a Linux system):

```
python -m venv /path/to/new/virtual/environment
source <ven_pathv>/bin/activate
python -m pip install -r requirements.txt
```

If you wish to store & view repeated experimental results, you will also need to set up a Mongo database (instructions [here](https://docs.mongodb.com/manual/installation/#std-label-tutorial-installation)) and have a local omniboard instance running (more information [here](https://github.com/vivekratnavel/omniboard)).

## Usage

After finishing the preceding steps, you should be able to run each experiment through its corresponding `.py` file. Each experiment requires model parameters to be specified in the corresponding `model_configs/*.json` file. The majority of the model architecture is defined in the `SetToSequence()` class from `models.py`. For example:

```
# Run TSP Experiment, with model_configs/tsp.json file prepared beforehand
python tsp.py
```

The Travelling Salesman Problem (TSP), Formal Grammar and Synthetic Structure Prediction experiments generate their own datasets using parameters specified in their corresponding scripts. 

When a training script is running, comprehensive progress logs will be visible in the console and saved to a log file in the `./logs` directory, e.g.:

```
2022-01-07 12:54:33,655 | INFO | EXPERIMENT run started, id: 8259183078
2022-01-07 12:54:33,655 | INFO | Time started (UTC): 2022-01-07 11:54:33.653905+00:00
2022-01-07 12:54:33,655 | INFO | Seeding for reproducibility with: 2679
(...)
```

