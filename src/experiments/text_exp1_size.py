
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from tqdm import tqdm  # Thêm thư viện tqdm
from src.text_visualization import plot_with_std

def run_exp_size(models, X_tf, X_bin, y, dataset_name, config):
    print(f"\n--- EXPERIMENT 1: DATA SIZE ({dataset_name}) ---")
    
    sizes = config["EXP1_SIZES"]
    cv_splits = config["CV_SPLITS"]
    seed = config["RANDOM_SEED"]
    plot_dir = config["PLOT_DIR"]
    
    results = {name: ([], []) for name in models.keys()}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    
    for s in tqdm(sizes, desc=f"Evaluating Sizes for {dataset_name}", unit="size"):
        X_tr_tf, _, y_tr, _ = train_test_split(X_tf, y, train_size=s, stratify=y, random_state=seed)
        X_tr_bin, _, _, _ = train_test_split(X_bin, y, train_size=s, stratify=y, random_state=seed)
        
        for name, (model, f_type) in models.items():
            X_use = X_tr_tf if f_type == 'tfidf' else X_tr_bin
            scores = cross_val_score(model, X_use, y_tr, cv=cv)
            results[name][0].append(np.mean(scores))
            results[name][1].append(np.std(scores))
            
    save_path = os.path.join(plot_dir, f"{dataset_name.replace(' ', '_')}_Data_Size.png")
    plot_with_std(results, sizes, 'Training Set Size', f'Learning Curve ({dataset_name})', save_path)