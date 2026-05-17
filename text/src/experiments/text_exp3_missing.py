import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
from src.text_visualization import plot_with_std

def run_exp_missing(models, X_tf, X_bin, y, dataset_name, config):
    print(f"\n--- EXPERIMENT 3: MISSING DATA ({dataset_name}) ---")
    
    rates = config["EXP3_MISSING_RATES"]
    subset_size = config["EXP_SUBSET_SIZE"]
    cv_splits = config["CV_SPLITS"]
    seed = config["RANDOM_SEED"]
    plot_dir = config["PLOT_DIR"]
    
    results = {name: ([], []) for name in models.keys()}
    
    X_sub_tf, _, y_sub, _ = train_test_split(X_tf, y, train_size=subset_size, stratify=y, random_state=seed)
    X_sub_bin, _, _, _ = train_test_split(X_bin, y, train_size=subset_size, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    for r in rates:
        for name, (model, f_type) in models.items():
            X_use = X_sub_tf if f_type == 'tfidf' else X_sub_bin
            fold_accs = []
            
            for tr_idx, ts_idx in cv.split(X_use, y_sub):
                X_tr = X_use[tr_idx].copy()
                X_ts = X_use[ts_idx].copy()
                
                mask_tr = np.random.rand(*X_tr.shape) < r
                mask_ts = np.random.rand(*X_ts.shape) < r
                X_tr[mask_tr] = 0
                X_ts[mask_ts] = 0
                
                if f_type == 'binary': 
                    X_tr = Binarizer().fit_transform(X_tr)
                    X_ts = Binarizer().fit_transform(X_ts)
                    
                model.fit(X_tr, y_sub[tr_idx])
                pred = model.predict(X_ts)
                fold_accs.append(accuracy_score(y_sub[ts_idx], pred))
                
            results[name][0].append(np.mean(fold_accs))
            results[name][1].append(np.std(fold_accs))
            
    save_path = os.path.join(plot_dir, f"{dataset_name.replace(' ', '_')}_Missing_Data.png")
    plot_with_std(results, rates, 'Missing Data Rate', f'Missing Data Robustness ({dataset_name})', save_path)