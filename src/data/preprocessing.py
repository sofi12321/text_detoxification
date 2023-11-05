def preprocess(sent):
    """
    Preprocess name of the dataset point
    Lowercased without punctuation and stop word
    Return list of preprocessed words from the sent
    """
    res = []

    try:
        words = word_tokenize(sent)
    except:
        print(f"\nTokenization fails for {sent}")
        return []

    for word in words:
        # Delete punctuation
        sent = sent.translate(str.maketrans("", "", string.punctuation))
        # Split by a free space
        word = word.strip()
        # Lowercase text
        word = word.lower()

        # Ignore free space
        if len(word) > 0:
            res.append(word)

    # Return list of preprocessed words from the sent
    return res

model_similarity = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model

def embed(text):
    # global nlp
    # return nlp.vocab.get_vector(text)

    global model_similarity
    return model_similarity.encode([text], convert_to_tensor=False)[0]

