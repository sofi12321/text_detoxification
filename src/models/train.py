from src.models.predict import *
from src.data.build_dataset import *
from src.data.preprocessing import *

from annoy import AnnoyIndex
import zipfile
# to avoid warnings
import warnings
warnings.filterwarnings('ignore')

def train():
    words, embeddings = build_word_embeddings()
    # Train Annoy indexing
    index = get_vector_index(embeddings)
    return index

def load():
    index = AnnoyIndex(embed("hello").shape[0], 'angular')

    with zipfile.ZipFile("models/annoy_index.zip", 'r') as zip_file:
        zip_file.extract("annoy_index.ann", "models/")

    index.load('models/annoy_index.ann')
    return index