## Best model description

### Goal of the model

I defined the condition for the best model as changing the sentence so that the result is non-toxic and similar in meaning to the original text. Therefore, I was looking for a solution that satisfies the criterion of non-toxicity and similarity.

### Model pippeline
The basic idea behind my implementation is that a paraphraser can make a meaningful sentence based on a toxic sentence if the toxic words are replaced with neutral synonyms. The solution consists of 4 main elements: synonym search, toxicity scoring, similarity scoring, paraphrasing (Image 1). 

![main_model](reports/main_model.png)

#### Sentence Preprocessing
First of all, the function "preprocess" obtain a sentence and return list of lowercased tokens without free spaces. 

#### Toxicity scoring
Then for each word toxicity function is applied. If word is non-toxic it stays without changes, otherwise the top-k non-toxic synonym is found. The toxicity score is given by a pre-trained zero-shot-classification model.

#### Synonyms choise
Synonyms are found using Annoy Index from Spotify. It is trained on embeddings of  150 000 the most frequent english words. The best synonym is a word with the smallest cosine distance to the embedding of a given word. Embeddings are created using pre-trained  Sentence Transformer(paraphrase-multilingual-MiniLM-L12-v2). 

Also this model is used to choose how different the sentence become after substituting initial word with its synonym. The synonym with the maximal similarity is chosen as a result. Note that "" (just skip initial word) is consideres as well.

#### Paraphraser
After all toxic words are substituded by their synonyms, paraphraser is used to create a more realistic result. Ideally, it should substitute not really properly chosen words (that may happen sometime) and the result should became more logically structured. Also it is useful for creating a human-like text format (for example, without whitespace before punctuation). 


### Evaluation
I chose 2 metrics to evaluate the model: BERTScore and toxicity. 

BERTScore is an automatic evaluation metric for text generation that computes a similarity score for each token in the candidate sentence with each token in the reference sentence. I chose it because it leverages the pre-trained contextual embeddings from BERT models and matches words in candidate and reference sentences by cosine similarity. Also, it is noted as one of the best evaluation model for text detoxification [14]. 

Toxicity is a a pre-trained zero-shot-classification model [12]. It gives a probability of a text to be toxic.