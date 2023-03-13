# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""
This is modified version of MovieLens 100k dataset for sequence recommendation. 
This script is not for the redistribution purpose. It is only for the ease of use.
The script automatically downloads from original link and prepares the dataset for you.
The original dataset is available at https://grouplens.org/datasets/movielens/100k/
"""


import csv
import json
import os
import pandas as pd
import random

import datasets


_CITATION = """\
@article{10.1145/2827872, author = {Harper, F. Maxwell and Konstan, Joseph A.}, 
title = {The MovieLens Datasets: History and Context}, 
year = {2015}, 
issue_date = {January 2016}, 
publisher = {Association for Computing Machinery}, 
address = {New York, NY, USA}, 
volume = {5}, 
number = {4}, 
issn = {2160-6455}, 
url = {https://doi.org/10.1145/2827872}, 
doi = {10.1145/2827872}, 
abstract = {The MovieLens datasets are widely used in education, research, and industry. They are downloaded hundreds of thousands of times each year, reflecting their use in popular press programming books, traditional and online courses, and software. These datasets are a product of member activity in the MovieLens movie recommendation system, an active research platform that has hosted many experiments since its launch in 1997. This article documents the history of MovieLens and the MovieLens datasets. We include a discussion of lessons learned from running a long-standing, live research platform from the perspective of a research organization. We document best practices and limitations of using the MovieLens datasets in new research.}, 
journal = {ACM Trans. Interact. Intell. Syst.}, 
month = {dec}, 
articleno = {19}, 
numpages = {19}, 
keywords = {Datasets, MovieLens, ratings, recommendations} }
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This is modified version of MovieLens 100k dataset for sequence recommendation. 
This script is not for the redistribution purpose. It is only for the ease of use.
The script automatically downloads from original link and prepares the dataset for you.

Following is the description of the original dataset:

MovieLens data sets were collected by the GroupLens Research Project
at the University of Minnesota.
 
This data set consists of:
	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
	* Each user has rated at least 20 movies. 
        * Simple demographic info for the users (age, gender, occupation, zip)

The data was collected through the MovieLens web site
(movielens.umn.edu) during the seven-month period from September 19th, 
1997 through April 22nd, 1998. This data has been cleaned up - users
who had less than 20 ratings or did not have complete demographic
information were removed from this data set. Detailed descriptions of
the data file can be found at the end of this file.

Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.
"""

_HOMEPAGE = "https://grouplens.org/datasets/movielens/100k/"

_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class ML100kSeq(datasets.GeneratorBasedBuilder):
    """This is modified version of MovieLens 100k dataset. It is a sequence of items rated by users."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="100k", version=VERSION, description="This is the 100k version of the dataset"),
    ]

    # DEFAULT_CONFIG_NAME = "100k"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == "100k":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "uid": datasets.Value("string"),
                    "seq": datasets.Sequence(datasets.Value("string")),
                    "candidates": datasets.Sequence(datasets.Value("string")),
                    "target": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'ml-100k', 'u.data'),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'ml-100k', 'u.data'),
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'ml-100k', 'u.data'),
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].strip().split('\t')
            df = pd.DataFrame(lines, index=None, columns=['uid', 'iid', 'rating', 'timestamp'])
        iid_list = df["iid"].unique()
        for uid in df["uid"].unique():
            sub_df = df.loc[df['uid'] == uid].sort_values(by='timestamp')
            sub_iid_list = list(sub_df['iid'])

            candidates = random.sample([item for item in iid_list if item not in sub_iid_list], 10)
            
            # list(sub.loc[sub['rating'] == '1']['iid'])
            if split == "train":
                yield uid, {
                        "uid": uid,
                        "seq": sub_iid_list[-10:-2],
                        "target": None,
                        "candidates": candidates
                    }
            elif split == "validation":
                yield uid, {
                        "uid": uid,
                        "seq": sub_iid_list[-10:-2],
                        "target": sub_iid_list[-2],
                        "candidates": candidates
                    }
            elif split == "test":
                yield uid, {
                        "uid": uid,
                        "seq": sub_iid_list[-10:-1],
                        "target": sub_iid_list[-1],
                        "candidates": candidates
                    }

