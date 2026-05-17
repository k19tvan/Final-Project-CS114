import streamlit as st
import numpy as np
import pandas as pd
import random
import re
from datasets import load_dataset
from src.text_data_loader import load_and_preprocess_data
from src.text_models import get_selected_models


st.set_page_config(page_title="DLA vs GLA Evaluation", layout="wide")


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


st.sidebar.title("Model Configuration")
st.sidebar.markdown("Adjust hyperparameters and data size to dynamically retrain the models.")

data_size = st.sidebar.slider("Training Data Size (Samples):", min_value=100, max_value=5000, value=1000, step=100)
alpha_val = st.sidebar.slider("Naive Bayes Laplace Smoothing (Alpha):", min_value=0.01, max_value=2.0, value=0.1, step=0.01)
svm_iter = st.sidebar.slider("SVM Maximum Iterations:", min_value=100, max_value=2000, value=1000, step=100)

current_config = {
    "N_SAMPLES": data_size,
    "NB_ALPHA": alpha_val,
    "MAX_ITER": svm_iter,
    "RANDOM_SEED": 42
}


@st.cache_resource
def load_base_data(n_samples):
    """Load base data utilizing src modules, cached based on n_samples."""
    mock_config = {"N_SAMPLES": n_samples, "MAX_FEATURES": 3000}
    X_tf, _, y = load_and_preprocess_data('ag_news', mock_config)
    
    dataset = load_dataset('ag_news', split=f'train[:{n_samples}]')
    raw_texts = dataset['text']
    raw_labels = dataset['label']
    
    return X_tf, y, raw_texts, raw_labels, mock_config

@st.cache_resource
def get_random_real_samples(_raw_texts, _raw_labels):
    """Extract random samples with their ground truth labels."""
    samples = list(zip(_raw_texts, _raw_labels))
    return random.sample(samples, 50)

X_train_tf, y_train, raw_texts, raw_labels, base_config = load_base_data(data_size)

@st.cache_resource
def train_dynamic_models(config, X, y):
    """Retrain models when configuration changes."""
    models_dict = get_selected_models(config)
    
    svm_model = models_dict["Linear SVM (DLA)"][0]
    nb_model = models_dict["Multinomial NB (GLA)"][0]
    
    # Re-initialize vectorizer for new inputs based on current Data Size
    n_samples = config["N_SAMPLES"]
    dataset = load_dataset('ag_news', split=f'train[:{n_samples}]')
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts_cleaned = [re.sub(r'[^a-z\s]', ' ', text.lower()).strip() for text in dataset['text']]
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, sublinear_tf=True)
    vectorizer.fit(texts_cleaned)
    
    with st.spinner(f"Training models on {n_samples} samples with new hyperparameters..."):
        svm_model.fit(X, y)
        nb_model.fit(X, y)
        
    return vectorizer, svm_model, nb_model

vectorizer, svm_model, nb_model = train_dynamic_models(current_config, X_train_tf, y_train)
labels_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def inject_missing_data(text, rate=0.4):
    words = text.split()
    noisy_words = ["___" if random.random() < rate and len(w) > 3 else w for w in words]
    return " ".join(noisy_words)

def inject_feature_noise(text, rate=0.15):
    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < rate and chars[i].isalpha() and chars[i+1].isalpha():
            chars[i], chars[i+1] = chars[i+1], chars[i] 
    return "".join(chars)


st.title("Discriminative vs Generative Models")
st.markdown("Interactive robustness evaluation using Support Vector Machines and Multinomial Naive Bayes.")
st.divider()

col_input, col_results = st.columns([1, 1.2])

with col_input:
    st.subheader("Data Input & Simulation")
    
    real_samples = get_random_real_samples(raw_texts, raw_labels)
    sample_options = [f"Sample {i+1}: {text[:60]}..." for i, (text, _) in enumerate(real_samples)]
    
    selected_option = st.selectbox("Select a ground-truth sample from AG News:", sample_options)
    selected_idx = sample_options.index(selected_option)
    base_text, true_label_idx = real_samples[selected_idx]
    
    # Session state initialization for text manipulation
    if 'current_base' not in st.session_state or st.session_state.current_base != base_text:
        st.session_state.current_base = base_text
        st.session_state.text_input = base_text
        
    user_input = st.text_area("Document Content:", value=st.session_state.text_input, height=150)
    
    st.markdown("**Noise Injection Experiments:**")
    st.caption("Note: Typo injection effectively acts as Feature Noise by destroying original TF-IDF features and creating Out-Of-Vocabulary (OOV) tokens.")
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("Apply Missing Data (40%)"):
            st.session_state.text_input = inject_missing_data(user_input)
            st.rerun()
    with btn_col2:
        if st.button("Apply Typo Noise (15%)"):
            st.session_state.text_input = inject_feature_noise(user_input)
            st.rerun()
    with btn_col3:
        if st.button("Reset to Original"):
            st.session_state.text_input = st.session_state.current_base
            st.rerun()

    st.markdown("---")
    st.markdown("#### Feature Space Representation (TF-IDF)")
    if user_input:
        input_cleaned = re.sub(r'[^a-z\s]', ' ', user_input.lower()).strip()
        X_test = vectorizer.transform([input_cleaned])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = X_test.toarray()[0]
        
        active_features = [(feature_names[i], tfidf_scores[i]) for i in np.nonzero(tfidf_scores)[0]]
        active_features.sort(key=lambda x: x[1], reverse=True)
        
        if active_features:
            st.caption(f"Active Features: {len(active_features)} / {len(feature_names)}")
            df_features = pd.DataFrame(active_features[:5], columns=["Token", "TF-IDF Weight"])
            st.dataframe(df_features, use_container_width=True, hide_index=True)
        else:
            st.warning("No valid features identified. The vector space is empty.")

with col_results:
    st.subheader("Classification Results")
    
    st.info(f"**Ground Truth (Actual Label):** {labels_map[true_label_idx]}")
    
    if user_input:
        input_cleaned = re.sub(r'[^a-z\s]', ' ', user_input.lower()).strip()
        X_test = vectorizer.transform([input_cleaned]).toarray()
        
        # --- PREDICTIONS & CONFIDENCE COMPUTATION ---
        # SVM (Discriminative)
        svm_pred_idx = svm_model.predict(X_test)[0]
        svm_margins = svm_model.decision_function(X_test)[0]
        svm_probs = softmax(svm_margins) # Convert margins to pseudo-probabilities for comparison
        
        # NB (Generative)
        nb_pred_idx = nb_model.predict(X_test)[0]
        nb_probs = nb_model.predict_proba(X_test)[0]
        
        # Handle zero-feature edge case
        if np.count_nonzero(X_test) == 0:
            svm_probs = np.array([0.25, 0.25, 0.25, 0.25])
            nb_probs = np.array([0.25, 0.25, 0.25, 0.25])
            
        # --- UI DISPLAY ---
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("##### Linear SVM (DLA)")
            st.metric(label="Prediction", value=labels_map[svm_pred_idx])
            st.caption(f"Pseudo-Confidence: {np.max(svm_probs)*100:.1f}%")
            
        with res_col2:
            st.markdown("##### Multinomial NB (GLA)")
            st.metric(label="Prediction", value=labels_map[nb_pred_idx])
            st.caption(f"Posterior Probability: {np.max(nb_probs)*100:.1f}%")
            
        # --- PROBABILITY DISTRIBUTION CHART ---
        st.markdown("#### Class Distribution Comparison")
        dist_df = pd.DataFrame({
            "Classes": list(labels_map.values()),
            "SVM (Softmax Margin)": svm_probs,
            "Naive Bayes (Posterior)": nb_probs
        }).set_index("Classes")
        
        st.bar_chart(dist_df, height=250)

        # --- MODEL EXPLAINABILITY ---
        with st.expander("Model Explainability (Top Class Weights)"):
            st.markdown(f"Top 5 learned features contributing to the prediction of **{labels_map[svm_pred_idx]}**:")
            
            svm_class_weights = svm_model.coef_[svm_pred_idx]
            top_svm_indices = svm_class_weights.argsort()[-5:][::-1]
            top_svm_features = [feature_names[i] for i in top_svm_indices]
            
            nb_class_weights = nb_model.feature_log_prob_[svm_pred_idx]
            top_nb_indices = nb_class_weights.argsort()[-5:][::-1]
            top_nb_features = [feature_names[i] for i in top_nb_indices]
            
            explain_df = pd.DataFrame({
                "SVM Highest Weights": top_svm_features,
                "NB Highest Probabilities": top_nb_features
            })
            st.dataframe(explain_df, use_container_width=True, hide_index=True)
            
        # --- ANALYTICAL INSIGHTS ---
        st.markdown("#### Analytical Insights")
        if np.count_nonzero(X_test) == 0:
            st.error("Model failure: Input document has completely collapsed into the null vector space.")
        elif np.max(svm_probs) - np.max(nb_probs) > 0.15:
            st.warning("DLA indicates higher resilience. Generative algorithm probabilities have degraded due to loss of evidence (sparse feature intersection).")
        elif svm_pred_idx != nb_pred_idx:
            st.warning("Decision boundary conflict: SVM and Naive Bayes evaluate the residual features differently.")
        else:
            st.success("Stable classification. Both algorithms converge on the same prediction with high confidence.")