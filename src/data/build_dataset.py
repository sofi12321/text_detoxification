def make_set(df):
    # Returns texts , labels
    # Where texts is a list of toxic and non-toxic sentences
    # labels is label 0 - non-toxic, 1 - toxic
    texts = []
    labels = []
    texts += list(df['input_text'])
    labels += list(np.ones(len(df['input_text'])))
    texts += list(df['target_text'])
    labels += list(np.zeros(len(df['target_text'])))
    res = pd.DataFrame({"texts": texts, "labels": labels})
    res = res.sample(frac=1).reset_index(drop=True)
    return res["texts"].to_numpy(), res["labels"].to_numpy()

def build_dataset_toxicity_classifier():
    # Returns X_train, y_train, X_eval, y_eval, X_test, y_test 
    # Where X_ is a list of toxic and non-toxic sentences
    # y_ is label 0 - non-toxic, 1 - toxic
    # Loading the zip file and extracting a zip object
    with zipfile.ZipFile("main_model_train.zip", 'r') as zip_file:
      zip_file.extract("main_model_train/main_model_train.csv", "../../data/interim/")
    
    # Read files
    train_df = pd.read_csv("../../data/interim/main_model_train.csv", index_col=0)
    eval_df = pd.read_csv("../../data/interim/main_model_eval.csv", index_col=0)
    test_df = pd.read_csv("../../data/interim/main_model_test.csv", index_col=0)
    
    X_train, y_train = make_set(train_df)
    X_eval, y_eval = make_set(eval_df)
    X_test, y_test = make_set(test_df)
    
    return X_train, y_train, X_eval, y_eval, X_test, y_test

def build_dataset():
    # Returns X_train, y_train, X_eval, y_eval, X_test, y_test 
    # Where X_ is a list of toxic and non-toxic sentences
    # y_ is label 0 - non-toxic, 1 - toxic
    # Loading the zip file and extracting a zip object
    with zipfile.ZipFile("../../data/interim/main_model_train.zip", 'r') as zip_file:
      zip_file.extract("main_model_train/main_model_train.csv", "../../data/interim/")
    
    # Read files
    train_df = pd.read_csv("../../data/interim/main_model_train.csv", index_col=0)
    eval_df = pd.read_csv("../../data/interim/main_model_eval.csv", index_col=0)
    test_df = pd.read_csv("../../data/interim/main_model_test.csv", index_col=0)

return train_df, eval_df, test_df

def build_word_embeddings():
    # # Load words embeddings
    with open('word_embeddings.pkl', "rb") as fIn:
        word_embeddings = pickle.load(fIn)
        
    words = word_embeddings['words']
    embeddings = word_embeddings['embedding']
    return words, embeddings
  
