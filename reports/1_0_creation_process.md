# Solution findings

## Dataset analysis 
First of all, I opened and analyzed the given dataset [1]. I found out that not all "reference" sentences are toxic, just like not all "translations" are non-toxic. Therefore, I processed the data so that I ended up with a dataframe with pairs of toxic ("input_text") - non-toxic ("target_text") text. In the resulting dataset, all pairs have similarity > 0.5, toxicity(input_text) > 0.5 and toxicity(target_text) < 0.5.

## Pre-trained Paraphraser
The first idea I tried out is to use a pre-trained model to paraphrase text. The hypothesis was that  the vocabulary of such a model does not contain abusive words, so model does not know them and consequently cannot predict. 

Model that I used was bart-paraphrase for BartForConditionalGeneration [2]. However, the hypothesis failed: this model not only kept toxic words, but made the sentence even more rude. Therefore, I rejected this hypothesis.

### Fine-tune Paraphraser
Idea that comes after using pre-trained model is to fune-tune it. This was my second hypothesis. I took the model [2] that I described before bart-paraphrase for paraphrasing (seq2seq) task. To make training a bit lighter, I freeze most of model layers. However training such a big (even with a frozen layers) model was really hard and ineffecient - results didn't improve much while my CUDA runs out of memory too many times. So I rejected that hypothesis as well.

### Synonyms substitution

Idea behind this topic is to change toxic words with their non-toxic synonyms. This is a more unusual hypothesis that became part of the main solution. 

Synonyms are found using Annoy Index from Spotify [3]. I trained it on embeddings of a set of words. The best synonyms are words with the smallest cosine distance to the embedding of a given word, that is a result of getting k nearest neighbour from Annoy Index.

To obtain better results, I tried several words datasets and embeddings models.

#### Dataset
For dataset I firstly chose a set of words with over 466k English words. However, it contains poorly used words, which interfered with the search for neutral synonyms.

Thus, I took "English Word Frequency" dataset that contains the 333,333 most commonly-used single words on the English language web [5]. Predictions based on this dataset was much better. 

#### Embeddings

At the beginning I tried to use the ready-made embedding from spicy. However, spicy.nlp.vocab.get_vector [6] did not show the good results, as well as gensim.Word2Vec [7]. 

That's why I decided to revisit pre-trained models. The model I chose is specifically designed to compare two texts (including words) - pre-trained Sentence Transformer(paraphrase-multilingual-MiniLM-L12-v2) [8]. 
My hypothesis was that this model would perform better than spicy.nlp and word2vec, and it did.


### Toxicity scoring
In order not to break the sentence and leave at least part of the original text, to save computational time and to keep logic involved, I decided to create a toxicity function. This function returns a number between 0 and 1, how toxic the word is. 

I tried different approaches, such as fine-tuning model [9], using ready for a task model [11] or pre-trained zero-shot-classification model [12]. 
The best accuracy has task-specific model, however it's prediction time is relatively big, so in the final solution I prefer to use zero-shot model.

### Paraphraser
Hypothesis: paraphraser may give a more realistic result given a toxic sentence with substituted neutral synonyms instead of toxic words. Ideally, paraphraser should substitute not really properly chosen words (that may happen sometime) and the result should became more logically structured. Also it may be useful for creating a human-like text format (for example, without whitespace before punctuation). 

I tried 2 pre-trained paraphrasers [2, 13], their perfomance is around the same. In the final solution I used model [2].


### Evaluation

Expected result of the solution is a non-toxic sentence similar in meaning to the original text. Therefore, to evaluate the perfomance of the solution criterions of non-toxicity and similarity should be satisfied.

I chose 2 metrics to evaluate the model: BERTScore and toxicity. 

BERTScore is an automatic evaluation metric for text generation that computes a similarity score for each token in the candidate sentence with each token in the reference sentence. I chose it because it leverages the pre-trained contextual embeddings from BERT models and matches words in candidate and reference sentences by cosine similarity. Also, it is noted as one of the best evaluation model for text detoxification [14]. 

Toxicity is a a pre-trained zero-shot-classification model [12] that was discussed earlier. It gives a probability of a text to be toxic.

