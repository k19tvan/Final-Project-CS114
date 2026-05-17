from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

def get_selected_models(config, selected_names=None):
    alpha = config["NB_ALPHA"]
    max_iter = config["MAX_ITER"]
    seed = config["RANDOM_SEED"]
    
    all_models = {
        "Multinomial NB (GLA)": (MultinomialNB(alpha=alpha), 'tfidf'),
        "Bernoulli NB (GLA)": (BernoulliNB(alpha=alpha), 'binary'),
        "Linear SVM (DLA)": (LinearSVC(max_iter=max_iter, random_state=seed), 'tfidf'),
        "Logistic Regression (DLA)": (LogisticRegression(max_iter=max_iter, random_state=seed, solver='lbfgs'), 'tfidf')
    }
    
    if selected_names:
        return {k: v for k, v in all_models.items() if k in selected_names}
    return all_models