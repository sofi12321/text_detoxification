from sentence_transformers import SentenceTransformer, util

model_similar = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 

def sentence_similarity(sent1, sent2):
    # Calculate cosine similarity between sentences
    global model_similar
    sentences = [
        sent1,
        sent2
    ]
    embedding = model_similar.encode(sentences, convert_to_tensor=False)

    cosine_scores = util.cos_sim(embedding, embedding)

    return cosine_scores[0][1].item()

