import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from src.data_loader import load_dataset
from src.models import LDAClassifier_wrapper, SVMClassifier_wrapper
from src.lda_em import LinearDiscriminantAnalysis, LDA_EM
from src.utils import extract_hog_features, save_plot

def run_experiment_3_on_dataset(dataset_name):
    print(f"\n" + "="*60)
    print(f"🚀 BẮT ĐẦU THÍ NGHIỆM 3: SEMI-SUPERVISED LEARNING TRÊN {dataset_name.upper()}")
    print("="*60)
    
    start_total_time = time.time()
    
    # 1. Load data
    print(f"[1/5] Đang tải dữ liệu {dataset_name}...")
    (train_X, train_y), (test_X, test_y) = load_dataset(dataset_name)
    
    pixels_per_cell = (7, 7) if dataset_name in ['mnist', 'fashion_mnist'] else (8, 8)
    cells_per_block = (2, 2)
    orientations = 9
    
    # 2. Extract HOG
    print(f"[2/5] Đang trích xuất đặc trưng HOG...")
    t0 = time.time()
    train_X_hog = extract_hog_features(train_X, pixels_per_cell, cells_per_block, orientations)
    test_X_hog = extract_hog_features(test_X, pixels_per_cell, cells_per_block, orientations)
    print(f"      -> Trích xuất xong. Thời gian: {time.time() - t0:.2f}s")
    
    LABELED_RATIOS = [0.01, 0.05, 0.1, 0.2, 0.5]
    N_RUNS = 5
    
    results = []
    convergence_data = []

    print(f"[3/5] Bắt đầu huấn luyện và đánh giá mô hình...")
    for run in range(N_RUNS):
        print(f"\n  ▶ RUN {run+1}/{N_RUNS}")
        
        for ratio in LABELED_RATIOS:
            print(f"    Labeled Ratio: {ratio*100:.0f}%")
            
            # Split labeled/unlabeled from full train set
            X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
                train_X_hog, train_y, train_size=ratio, stratify=train_y, random_state=run
            )
            
            # Scale features (ONLY for SVM)
            scaler = StandardScaler()
            X_l_scaled = scaler.fit_transform(X_labeled)
            X_t_scaled_svm = scaler.transform(test_X_hog)

            # --- 1. Baseline: Supervised LDA ---
            # LDA uses raw HOG features (unscaled)
            lda_sup = LDAClassifier_wrapper()
            lda_sup.fit(X_labeled, y_labeled)
            acc_lda_sup = lda_sup.score(test_X_hog, test_y)
            results.append({'Run': run, 'Ratio': ratio, 'Method': 'LDA (Supervised)', 'Accuracy': acc_lda_sup})

            # --- 2. Baseline: Supervised SVM ---
            # SVM uses scaled features
            svm_sup = SVMClassifier_wrapper()
            svm_sup.train(X_l_scaled, y_labeled)
            acc_svm_sup = svm_sup.evaluate(X_t_scaled_svm, test_y)
            results.append({'Run': run, 'Ratio': ratio, 'Method': 'SVM (Supervised)', 'Accuracy': acc_svm_sup})

            # --- 3. LDA Self-Training ---
            # Uses raw features
            X_l_curr = X_labeled.copy()
            y_l_curr = y_labeled.copy()
            X_u_curr = X_unlabeled.copy()
            X_t_curr = test_X_hog
            
            for iter in range(10):
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                clf.fit(X_l_curr, y_l_curr)
                
                probs = clf.predict_proba(X_u_curr)
                max_probs = np.max(probs, axis=1)
                high_conf_mask = max_probs >= 0.9
                
                n_added = np.sum(high_conf_mask)
                if n_added < 10:
                    break
                
                pseudo_labels = np.argmax(probs[high_conf_mask], axis=1)
                X_l_curr = np.vstack([X_l_curr, X_u_curr[high_conf_mask]])
                y_l_curr = np.concatenate([y_l_curr, pseudo_labels])
                X_u_curr = X_u_curr[~high_conf_mask]
                
                if ratio == 0.05 and run == 0:
                    convergence_data.append({'Iter': iter, 'Value': n_added, 'Method': 'LDA Self-Training', 'Metric': 'Pseudo-labels Added'})
                
                if X_u_curr.shape[0] == 0:
                    break
            
            # Final eval for LDA Self-Training
            final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            final_clf.fit(X_l_curr, y_l_curr)
            acc_lda_st = np.mean(final_clf.predict(X_t_curr) == test_y)
            results.append({'Run': run, 'Ratio': ratio, 'Method': 'LDA Self-Training', 'Accuracy': acc_lda_st})

            # --- 4. SVM Self-Training ---
            # Reset current sets for SVM (using scaled)
            X_l_curr_svm = X_l_scaled.copy()
            y_l_curr_svm = y_labeled.copy()
            X_u_curr_svm = scaler.transform(X_unlabeled)
            X_t_curr_svm = X_t_scaled_svm
            
            for iter in range(10):
                base_svm = SVMClassifier_wrapper().model
                # Calibration needed for probabilities
                clf = CalibratedClassifierCV(base_svm, cv=3)
                clf.fit(X_l_curr_svm, y_l_curr_svm)
                
                probs = clf.predict_proba(X_u_curr_svm)
                max_probs = np.max(probs, axis=1)
                high_conf_mask = max_probs >= 0.9
                
                n_added = np.sum(high_conf_mask)
                if n_added < 10:
                    break
                
                pseudo_labels = np.argmax(probs[high_conf_mask], axis=1)
                X_l_curr_svm = np.vstack([X_l_curr_svm, X_u_curr_svm[high_conf_mask]])
                y_l_curr_svm = np.concatenate([y_l_curr_svm, pseudo_labels])
                X_u_curr_svm = X_u_curr_svm[~high_conf_mask]
                
                if ratio == 0.05 and run == 0:
                    convergence_data.append({'Iter': iter, 'Value': n_added, 'Method': 'SVM Self-Training', 'Metric': 'Pseudo-labels Added'})
                
                if X_u_curr_svm.shape[0] == 0:
                    break
            
            acc_svm_st = np.mean(clf.predict(X_t_curr_svm) == test_y)
            results.append({'Run': run, 'Ratio': ratio, 'Method': 'SVM Self-Training', 'Accuracy': acc_svm_st})

            # --- 5. LDA-EM ---
            # EM also uses raw features
            lda_em = LDA_EM(alpha=0.5, max_iter=50, tol=1e-4)
            X_u_sub = X_unlabeled[:10000] if X_unlabeled.shape[0] > 10000 else X_unlabeled
            lda_em.fit(X_labeled, y_labeled, X_u_sub)
            
            acc_lda_em = np.mean(lda_em.predict(test_X_hog) == test_y)
            results.append({'Run': run, 'Ratio': ratio, 'Method': 'LDA-EM', 'Accuracy': acc_lda_em})
            
            print(f"      Acc -> LDA: {acc_lda_sup:.4f}, SVM: {acc_svm_sup:.4f}, LDA-ST: {acc_lda_st:.4f}, SVM-ST: {acc_svm_st:.4f}, EM: {acc_lda_em:.4f}")
            
            if ratio == 0.05 and run == 0:
                for i, ll in enumerate(lda_em.convergence_log_likelihood):
                    convergence_data.append({'Iter': i, 'Value': ll, 'Method': 'LDA-EM', 'Metric': 'Log-Likelihood'})

    # 4. Lưu kết quả và vẽ biểu đồ
    output_dir = 'results/experiment3'
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"experiment3_{dataset_name}_results.csv"), index=False)
    
    # Plot 1: Accuracy vs Labeled Ratio
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Ratio', y='Accuracy', hue='Method', marker='o', errorbar='sd')
    plt.title(f'Semi-Supervised Learning Accuracy on {dataset_name.upper()}')
    plt.xlabel('Labeled Data Ratio')
    plt.ylabel('Test Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(f"experiment3_{dataset_name}_accuracy.png", directory=output_dir)
    
    # Plot 2: Delta Accuracy
    # Calculate improvement over supervised baseline
    df_mean = df.groupby(['Ratio', 'Method'])['Accuracy'].mean().unstack()
    df_delta = pd.DataFrame()
    df_delta['Ratio'] = df_mean.index
    df_delta['LDA Self-Training'] = df_mean['LDA Self-Training'].values - df_mean['LDA (Supervised)'].values
    df_delta['LDA-EM'] = df_mean['LDA-EM'].values - df_mean['LDA (Supervised)'].values
    df_delta['SVM Self-Training'] = df_mean['SVM Self-Training'].values - df_mean['SVM (Supervised)'].values
    
    df_delta_melted = df_delta.melt(id_vars='Ratio', var_name='Method', value_name='Delta Accuracy')
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_delta_melted, x='Ratio', y='Delta Accuracy', hue='Method', marker='s')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Improvement over Supervised Baseline ({dataset_name.upper()})')
    plt.xlabel('Labeled Data Ratio')
    plt.ylabel('Delta Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(f"experiment3_{dataset_name}_delta.png", directory=output_dir)
    
    # Plot 3: Convergence curves (Ratio 5%)
    conv_df = pd.DataFrame(convergence_data)
    
    # Self-training convergence
    st_conv = conv_df[conv_df['Metric'] == 'Pseudo-labels Added']
    if not st_conv.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=st_conv, x='Iter', y='Value', hue='Method', marker='o')
        plt.title(f'Self-Training Convergence (Ratio=5%, {dataset_name.upper()})')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Pseudo-labels Added')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(f"experiment3_{dataset_name}_st_convergence.png", directory=output_dir)
    
    # EM convergence
    em_conv = conv_df[conv_df['Method'] == 'LDA-EM']
    if not em_conv.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=em_conv, x='Iter', y='Value', marker='o')
        plt.title(f'LDA-EM Log-Likelihood Convergence (Ratio=5%, {dataset_name.upper()})')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(f"experiment3_{dataset_name}_em_convergence.png", directory=output_dir)

    total_time = time.time() - start_total_time
    print(f"\n✅ Hoàn thành Experiment 3 trên {dataset_name.upper()}! (Tổng thời gian: {total_time/60:.2f} phút)")

def run_experiment_3():
    for ds in ['mnist', 'fashion_mnist', 'cifar10']:
        run_experiment_3_on_dataset(ds)

if __name__ == "__main__":
    run_experiment_3()
