from transformers import pipeline

def toxisity(text):
    # Returns toxicity of the text
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    labels = ["toxic", "non-toxic"]
    hypothesis_template = 'This text is {}.'
    prediction = classifier(text, labels, hypothesis_template=hypothesis_template, multi_class=True)
    return prediction
