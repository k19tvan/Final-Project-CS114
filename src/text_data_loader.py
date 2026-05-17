import re
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Binarizer

def load_and_preprocess_data(dataset_name, config):
    print(f"[*] Loading dataset: {dataset_name.upper()}...")
    
    n_samples = config["N_SAMPLES"]
    max_features = config["MAX_FEATURES"]
    
    dataset = load_dataset(dataset_name, split='train')
    
    if dataset_name == 'ag_news':
        texts = dataset['text'][:n_samples]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, sublinear_tf=True)
    else:
        texts = [f"{str(t)} {str(c)}" for t, c in zip(dataset['title'][:n_samples], dataset['content'][:n_samples])]
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=max_features, sublinear_tf=True)
    
    labels = np.array(dataset['label'][:n_samples])
    
    texts_cleaned = [re.sub(r'[^a-z\s]', ' ', text.lower()).strip() for text in texts]
    texts_cleaned = [re.sub(r'\s+', ' ', text) for text in texts_cleaned]
    
    X_tf = vectorizer.fit_transform(texts_cleaned).toarray()
    X_bin = Binarizer().fit_transform(X_tf)
    
    print(f"    -> Vectorized shape: {X_tf.shape}")
    return X_tf, X_bin, labels