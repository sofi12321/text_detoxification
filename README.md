# Text Detoxification

### Creator
Sofi Shulepina

B21-DS-02

s.zaitseva@innopolis.university

## Task Deskription

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.


## Data Description

The main dataset is a filtered subset of the ParaNMT corpus (50M sentence pairs). 

Additionally, this solution uses English Word Frequency Dataset.

All data is presented in /data folder.


## Repository structure

```
text_detoxification
├── README.md # The top-level README
│
├── data
│   ├── external # Data from third party sources
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│   ├── annoy_index.ann # Annoy index model
│   └── model_toxicity # checkpoint for toxicity model
│
├── notebooks    #  Jupyter notebooks
│   ├── 1_0_data_analysis.ipynb # Download and prepare data
│   └── 2_0_main_model_solution.ipynb # Final solution workflow        
│
├── references   
│   └── references.md  # Used resources
│
├── reports      
│   ├── 1_0_creation_process.md # First report on solution findings
│   ├── 2_0_final_solution_description.md # Second report on solution explanation
│   └── main_model.png  # Final solution workflow chart
│
├── requirements.txt # The requirements file for reproducing the analysis environment
└── src                 # Source code for use
    │                 
    ├── data            # Scripts to download or generate data
    │   ├── build_dataset.py # Dataset related functions
    │   └── preprocessing.py # Preprocessing function
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict.py # Run solution prediction
    │   ├── train.py # Train model
    │   ├── paraphraser.py # Paraphraser block
    │   ├── similarity.py # Sentence similarity calculation
    │   ├── synonyms.py # Find k synonyms
    │   ├── text_evaluator.py # Evaluation function
    │   └── toxic_classifier.py # Text toxicity scoring
    │   
    └── visualization   # Results oriented visualizations
        └── visualize.py
```


### Requirements 

Install requirements before start of the work
```python
pip install requirements.txt
```

### Run model

Let's run the model and look at predictions

```python
from src.models.predict import *
from src.data.build_dataset import *
from src.data.models.text_evaluator import *

train_df, eval_df, test_df = build_dataset()
predictions = predict(train_df[:10])
scores = evaluate(predict(predictions, train_df[:10]))
```


### References

All used materials are listed in file references/references.md
