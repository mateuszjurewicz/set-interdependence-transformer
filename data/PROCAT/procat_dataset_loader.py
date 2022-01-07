# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""PROCAT dataset."""

from __future__ import absolute_import, division, print_function

import csv
import os
import pathlib

import datasets
import numpy as np

CITATION = """
@inproceedings{jurewicz2021procat,
  title={PROCAT: Product Catalogue Dataset for Implicit Clustering, Permutation Learning and Structure Prediction},
  author={Jurewicz, Mateusz Maria and Derczynski, Leon},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  year={2021}
}
"""
_DESCRIPTION = """
"""

_PATH = "data/PROCAT/"

_SENTENCES = "sentences"
_SHUFFLED_SENTENCES = "shuffled_sentences"
_LABEL = "label"


class PROCATOrdering(datasets.GeneratorBasedBuilder):
    """PROCAT ordering dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _SENTENCES: datasets.Sequence(datasets.Value("string")),
                    _SHUFFLED_SENTENCES: datasets.Sequence(datasets.Value("string")),
                    _LABEL: datasets.Sequence(datasets.Value("int64")),
                }
            ),
            supervised_keys=None,
            homepage="https://figshare.com/articles/dataset/PROCAT_Product_Catalogue_Dataset_for_Implicit_Clustering_Permutation_Learning_and_Structure_Prediction/14709507",
            citation=CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = os.path.join(pathlib.Path().absolute(), _PATH)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_path, "PROCAT.train.csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_path, "PROCAT.validation.csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(data_path, "PROCAT.test.csv")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, "r", encoding='UTF-8') as f:
            csv_reader = csv.reader(f, delimiter=";")
            for i, elems in enumerate(csv_reader):
                if len(elems) != 201:
                    continue
                sentences = elems[-200:]

                shuffled_sentences, label = self.shuffle_sentences(sentences)
                yield i, {
                    _SENTENCES: sentences,
                    _SHUFFLED_SENTENCES: shuffled_sentences,
                    _LABEL: label,
                }

    def shuffle_sentences(self, sentences):
        sentences = np.array(sentences)
        permutation = np.random.permutation(len(sentences))
        return sentences[permutation].tolist(), np.argsort(permutation).tolist()
