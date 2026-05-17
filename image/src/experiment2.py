import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_dataset
from src.models import LDAClassifier_wrapper, SVMClassifier_wrapper
from src.utils import extract_hog_features, add_label_noise, add_feature_noise, save_plot

def run_experiment_2_on_dataset(dataset_name):
    print(f"\n" + "="*50)
    print(f"🚀 BẮT ĐẦU THÍ NGHIỆM 2 (NOISE) TRÊN DATASET: {dataset_name.upper()}")
    print("="*50)
    
    start_total_time = time.time()
    
    # 1. Load Data
    print(f"[1/4] Đang tải dữ liệu {dataset_name}...")
    (train_X, train_y), (test_X, test_y) = load_dataset(dataset_name)
    
    pixels_per_cell = (7, 7) if dataset_name in ['mnist', 'fashion_mnist'] else (8, 8)
    cells_per_block = (2, 2)
    orientations = 9
    
    print(f"[2/4] Đang trích xuất đặc trưng HOG...")
    t0 = time.time()
    train_X_hog = extract_hog_features(train_X, pixels_per_cell, cells_per_block, orientations)
    test_X_hog = extract_hog_features(test_X, pixels_per_cell, cells_per_block, orientations)
    print(f"      -> Trích xuất HOG xong. Thời gian: {time.time() - t0:.2f}s")
    
    LABEL_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    FEATURE_NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    setups = [
        {'name': 'subset500', 'size': 500, 'runs': 5},
        {'name': 'full', 'size': None, 'runs': 1}
    ]

    # Helper cho việc chạy Train & Eval
    def run_trial(X_train, y_train, X_test, y_test):
        # --- Pipeline LDA (Chạy trực tiếp trên HOG) ---
        lda = LDAClassifier_wrapper()
        lda.fit(X_train, y_train)
        acc_lda = lda.score(X_test, y_test)
        
        # --- Pipeline SVM (Phải đi qua StandardScaler) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svm = SVMClassifier_wrapper()
        svm.train(X_train_scaled, y_train)
        acc_svm = svm.evaluate(X_test_scaled, y_test)
        
        return acc_lda, acc_svm

    print(f"[3/4] Bắt đầu đánh giá khả năng chống nhiễu...")
    for setup in setups:
        case_name = setup['name']
        subset_size = setup['size']
        n_runs = setup['runs']
        
        print(f"\n  =========================================")
        print(f"  ▶ ĐANG CHẠY CASE: {case_name.upper()} (Size: {subset_size if subset_size else 'All'}, Runs: {n_runs})")
        print(f"  =========================================")
        
        noise_results = []

        for run in range(n_runs):
            print(f"\n    [Run {run+1}/{n_runs}]")
            
            # Phân tách tập con nếu cần
            if subset_size is None:
                X_subset_hog, y_subset = train_X_hog, train_y
            else:
                X_subset_hog, _, y_subset, _ = train_test_split(
                    train_X_hog, train_y,
                    train_size=subset_size,
                    stratify=train_y,
                    random_state=run # Đổi seed theo run
                )

            # ----------------------------------------------------
            # PHẦN A: LABEL NOISE (Nhiễu Nhãn)
            # ----------------------------------------------------
            print(f"      >> Đang test Label Noise...", end=" ", flush=True)
            t_label = time.time()
            for level in LABEL_NOISE_LEVELS:
                y_noisy = add_label_noise(y_subset, level)
                acc_lda, acc_svm = run_trial(X_subset_hog, y_noisy, test_X_hog, test_y)
                
                noise_results.append({'Run': run, 'Noise_Type': 'Label', 'Level': level, 'Method': 'LDA', 'Accuracy': acc_lda})
                noise_results.append({'Run': run, 'Noise_Type': 'Label', 'Level': level, 'Method': 'SVM', 'Accuracy': acc_svm})
            print(f"Xong! ({time.time() - t_label:.2f}s)")

            # ----------------------------------------------------
            # PHẦN B: FEATURE NOISE (Nhiễu Đặc Trưng / Gaussian)
            # ----------------------------------------------------
            print(f"      >> Đang test Feature Noise...", end=" ", flush=True)
            t_feat = time.time()
            
            # Scale dữ liệu trước khi ném nhiễu vào (để sigma có ý nghĩa thống kê)
            scaler_noise_prep = StandardScaler()
            X_subset_prep = scaler_noise_prep.fit_transform(X_subset_hog)
            X_test_prep = scaler_noise_prep.transform(test_X_hog)

            for sigma in FEATURE_NOISE_LEVELS:
                X_noisy = add_feature_noise(X_subset_prep, sigma)
                acc_lda, acc_svm = run_trial(X_noisy, y_subset, X_test_prep, test_y)
                
                noise_results.append({'Run': run, 'Noise_Type': 'Feature', 'Level': sigma, 'Method': 'LDA', 'Accuracy': acc_lda})
                noise_results.append({'Run': run, 'Noise_Type': 'Feature', 'Level': sigma, 'Method': 'SVM', 'Accuracy': acc_svm})
            print(f"Xong! ({time.time() - t_feat:.2f}s)")

        # 4. Lưu và Vẽ Đồ Thị cho Case hiện tại
        output_dir = os.path.join('results/experiment2', case_name)
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(noise_results)
        df.to_csv(os.path.join(output_dir, f"experiment2_{dataset_name}_{case_name}_results.csv"), index=False)
        
        sns.set_theme(style="whitegrid")
        
        # Biểu đồ Label Noise
        plt.figure(figsize=(10, 6))
        label_df = df[df['Noise_Type'] == 'Label']
        sns.lineplot(data=label_df, x='Level', y='Accuracy', hue='Method', marker='o', errorbar='sd', linewidth=2)
        plt.title(f'Label Noise Robustness on {dataset_name.upper()} ({case_name})')
        plt.xlabel('Noise Level (Label Flip Probability)')
        plt.ylabel('Test Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot(f"experiment2_{dataset_name}_{case_name}_label_noise.png", directory=output_dir)
        
        # Biểu đồ Feature Noise
        plt.figure(figsize=(10, 6))
        feature_df = df[df['Noise_Type'] == 'Feature']
        sns.lineplot(data=feature_df, x='Level', y='Accuracy', hue='Method', marker='s', errorbar='sd', linewidth=2)
        plt.title(f'Feature Noise Robustness (Gaussian) on {dataset_name.upper()} ({case_name})')
        plt.xlabel('Noise Sigma (Standard Deviations)')
        plt.ylabel('Test Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot(f"experiment2_{dataset_name}_{case_name}_feature_noise.png", directory=output_dir)
    
    total_time = time.time() - start_total_time
    print(f"\n[4/4] ✅ Hoàn thành Experiment 2 trên {dataset_name.upper()}! (Tổng thời gian: {total_time/60:.2f} phút)")
    print("="*50)

def run_experiment_2():
    datasets = ['cifar10']
    for ds in datasets:
        run_experiment_2_on_dataset(ds)

if __name__ == "__main__":
    run_experiment_2()