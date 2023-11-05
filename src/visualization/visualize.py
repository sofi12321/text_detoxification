from src.models.predict import *
from src.data.build_dataset import *

# vizualization library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# pytorch library
import torch  # the main pytorch library
import torch.nn.functional as f  # the sub-library containing different functions for manipulating with tensors

# huggingface's transformers library
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

from annoy import AnnoyIndex

import torch

from evaluate import load
import numpy as np
# to avoid warnings
import warnings
warnings.filterwarnings('ignore')


def visualize_sentences(sentences):
    set_seed(42)
    bertscore = load("bertscore")

    for i in sentences:
        print("Initial text:", i)
        res = process_sentence(i)
        print("Result:", res)
        score = bertscore.compute(predictions=[res], references=[i], lang="en")
        print("Bart score is ", score)

        print()


def random_visualize(k=3):
    _, _, test_df = build_dataset()
    ind = np.random.choice(list(test_df.index.values), size=k, replace=False)
    visualize_sentences(test_df.iloc[ind])
