# Imports
# vizualization library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# pytorch library
import torch # the main pytorch library
import torch.nn.functional as f # the sub-library containing different functions for manipulating with tensors

# huggingface's transformers library
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

from annoy import AnnoyIndex

import torch

import pandas as pd
import zipfile
from urllib.request import urlretrieve

from nltk.tokenize import word_tokenize
import string
import tqdm
import pickle
import gc

from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string

import nltk
from transformers import T5ForConditionalGeneration,T5Tokenizer
import sentencepiece

from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import  AdamW 
from transformers import pipeline
#to avoid warnings
import warnings

from src.models.paraphraser import paraphrase_sent
from src.models.similarity import sentence_similarity
from src.models.synonyms import get_vector_index, get_kNN_embeddings
from src.models.toxic_classifier import toxisity
from src.data.build_dataset import build_word_embeddings,  build_dataset, build_dataset_toxicity_classifier, make_set
from src.data.preprocessing import preprocess‎, embed
from src.data.models.evaluation import evaluate
from evaluate import load

bertscore = load("bertscore")

def text_evaluate(pred, true_label):
    global bertscore
    return bertscore.compute(predictions=[pred], references=[true_label], lang="en")
