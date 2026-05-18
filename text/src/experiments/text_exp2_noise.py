
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
from tqdm import tqdm  # Thêm thư viện tqdm
from src.text_visualization import plot_with_std

def inject_label_noise(y, noise_level):
    if noise_level == 0: return y.copy()
    y_noisy = y.copy()
    n_noisy = int(noise_level * len(y))
    idx = np.random.choice(len(y), n_noisy, replace=False)
    classes = np.unique(y)
    for i in idx:
        y_noisy[i] = np.random.choice(classes[classes != y_noisy[i]])
    return y_noisy

def inject_feature_noise(X, sigma):
    if sigma == 0: return X.copy()
    return np.clip(X + np.random.normal(0, sigma, X.shape), 0, None)

def run_exp_noise(models, X_tf, X_bin, y, dataset_name, config):
    print(f"\n--- EXPERIMENT 2: NOISE ROBUSTNESS ({dataset_name}) ---")
    
    subset_size = config["EXP_SUBSET_SIZE"]
    cv_splits = config["CV_SPLITS"]
    seed = config["RANDOM_SEED"]
    plot_dir = config["PLOT_DIR"]
    
    X_sub_tf, _, y_sub, _ = train_test_split(X_tf, y, train_size=subset_size, stratify=y, random_state=seed)
    X_sub_bin, _, _, _ = train_test_split(X_bin, y, train_size=subset_size, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    # 2A: LABEL NOISE
    print("  Running Label Noise evaluation...")
    label_levels = config["EXP2_LABEL_NOISE"]
    results_label = {name: ([], []) for name in models.keys()}
    
    for lvl in tqdm(label_levels, desc="Label Noise Levels", unit="lvl"):
        for name, (model, f_type) in models.items():
            X_use = X_sub_tf if f_type == 'tfidf' else X_sub_bin
            fold_accs = []
            
            for tr_idx, ts_idx in cv.split(X_use, y_sub):
                y_tr_noisy = inject_label_noise(y_sub[tr_idx], lvl)
                model.fit(X_use[tr_idx], y_tr_noisy)
                pred = model.predict(X_use[ts_idx])
                fold_accs.append(accuracy_score(y_sub[ts_idx], pred))
                
            results_label[name][0].append(np.mean(fold_accs))
            results_label[name][1].append(np.std(fold_accs))
            
    save_path_label = os.path.join(plot_dir, f"{dataset_name.replace(' ', '_')}_Label_Noise_{subset_size}.png")
    plot_with_std(results_label, label_levels, 'Label Noise Ratio', f'Label Noise Robustness ({dataset_name})', save_path_label)

    # 2B: FEATURE NOISE
    print("  Running Feature Noise evaluation...")
    feature_levels = config["EXP2_FEATURE_NOISE"]
    results_feature = {name: ([], []) for name in models.keys()}
    
    for sigma in tqdm(feature_levels, desc="Feature Noise Levels", unit="sigma"):
        for name, (model, f_type) in models.items():
            X_use = X_sub_tf if f_type == 'tfidf' else X_sub_bin
            fold_accs = []
            
            for tr_idx, ts_idx in cv.split(X_use, y_sub):
                X_tr_noisy = inject_feature_noise(X_use[tr_idx], sigma)
                
                if f_type == 'binary':
                    threshold_val = sigma / 2.0 if sigma > 0 else 0.0
                    X_tr_noisy = Binarizer(threshold=threshold_val).fit_transform(X_tr_noisy)
                    
                model.fit(X_tr_noisy, y_sub[tr_idx])
                pred = model.predict(X_use[ts_idx])
                fold_accs.append(accuracy_score(y_sub[ts_idx], pred))
                
            results_feature[name][0].append(np.mean(fold_accs))
            results_feature[name][1].append(np.std(fold_accs))
            
    save_path_feat = os.path.join(plot_dir, f"{dataset_name.replace(' ', '_')}_Feature_Noise_{subset_size}.png")
    plot_with_std(results_feature, feature_levels, 'Gaussian Noise Std (Sigma)', f'Feature Noise Robustness ({dataset_name})', save_path_feat)