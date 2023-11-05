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
from src.data.models.text_evaluator import text_evaluate
from evaluate import load
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def process_sentence(sent, max_synonyms=3):
    words = preprocess(sent)
    result = []
    for ind, word in enumerate(words):
        if toxisity(word) <= 0.5:
            result.append(word)
            continue

        best_synonym = ""
        best_simil_score = -1
        # May be add analyzing skipping !!!
        counter = 0
        for synonym in get_synonyms(word):
            toxic = toxisity(synonym)
            if toxic <= 0.5:
                # Create a sentence as initial
                potential_sentence = words.copy()
                # And replace toxic word with a synonym
                potential_sentence[ind] = synonym
                potential_similarity = sentence_similarity(sent, " ".join(potential_sentence))

                # Better synonym in the context
                if potential_similarity > best_simil_score:
                    best_simil_score = potential_similarity
                    best_synonym = synonym
                    print(f"Better synonym for {word} is {synonym}")

                # Non-toxic word was analyzed
                counter += 1

            if counter >= max_synonyms:
                # Analyze only top max_synonyms
                break
        result.append(best_synonym)
    print("Before paraphrasing:", " ".join(result))
    # Paraphrase
    return paraphrase_sent(" ".join(result))

def predict(sentences):
    predictions = [process_sentence(s) for s in sentences]
    return predictions

if __name__ == "__main__":
    # Prepare for work
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    warnings.filterwarnings('ignore')
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bertscore = load("bertscore")
    model_similar = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 
    model_similarity = model_similar
    paraphraser = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    paraphrase_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    paraphraser = paraphraser.to(device)
    train_df, eval_df, test_df = build_dataset()
    print("Evaluation result: ", evaluate(predict(train_df[:10]), train_df[:10]))
    
    
