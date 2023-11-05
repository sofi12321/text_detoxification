def get_vector_index(embeddings, start=0):
    # Initialize index
    annoy3 = AnnoyIndex(embeddings.shape[1], 'angular')
    i = start
    for embedding in tqdm.tqdm(embeddings):
        try:
            # Add non-zero embeddings in the index
            # Because points with zero embedding
            # is given to "unknown" words and phrases
            if np.sum(np.abs(embedding)) != 0:
                annoy3.add_item(i, embedding)
        except:
            pass
        i += 1

    # Build 37 trees
    annoy3.build(37)
    print("Index is constructed")
    annoy3.save('annoy_index.ann')

    # Return resulting index
    return annoy3


def get_kNN_embeddings(embedding, k, index):
    # Obtain nearest neighbours
    return index.get_nns_by_vector(embedding, k)
