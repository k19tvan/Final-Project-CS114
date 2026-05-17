import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_dataset
from src.models import DimensionReduction, LDAClassifier_wrapper, SVMClassifier_wrapper
from src.utils import extract_hog_features, save_plot

def run_experiment_1_on_dataset(dataset_name):
    print(f"--- Starting Experiment 1 on {dataset_name} ---")
    
    # 1. Load Data
    (train_X, train_y), (test_X, test_y) = load_dataset(dataset_name)
    
    # 2. HOG Parameters
    if dataset_name in ['mnist', 'fashion_mnist']:
        pixels_per_cell = (7, 7)
    else: # cifar10
        pixels_per_cell = (8, 8)
    
    cells_per_block = (2, 2)
    orientations = 9
    
    print(f"Extracting HOG features for {dataset_name}...")
    train_X_hog = extract_hog_features(train_X, pixels_per_cell, cells_per_block, orientations)
    test_X_hog = extract_hog_features(test_X, pixels_per_cell, cells_per_block, orientations)
    
    print(f"HOG feature dimension: {train_X_hog.shape[1]}")
    
    # 3. Experimental Setup
    pca_dims = [2, 5, 10, 15, 20, 50, 100, 150, 200, 250, 300, 324]
    if dataset_name == 'cifar10':
        pca_dims = [d for d in pca_dims if d <= 144]
        train_subsets = [20, 50, 100, 200, 1000, 5000, 10000, 20000, 40000, 50000]
    else:
        train_subsets = [20, 50, 100, 200, 1000, 5000, 10000, 20000, 40000, 60000]
    
    results = []
    
    # 4. Training Loop
    for size in train_subsets:
        print(f"Processing Subset Size: {size}")
        # Use full dataset if size matches total samples, otherwise use stratified split
        if size == len(train_X_hog):
            X_subset_hog, y_subset = train_X_hog, train_y
        else:
            X_subset_hog, _, y_subset, _ = train_test_split(
                train_X_hog, train_y,
                train_size=size,
                stratify=train_y,
                random_state=42
            )
        
        for dim in pca_dims:
            if dim > size: continue
            
            # Reduce dimension using PCA (fit only on train subset)
            dr = DimensionReduction(n_components=dim)
            X_train_pca = dr.fit_transform(X_subset_hog)
            X_test_pca = dr.transform(test_X_hog)
            
            # --- LDA ---
            lda = LDAClassifier_wrapper()
            lda.fit(X_train_pca, y_subset)
            acc_lda = lda.score(X_test_pca, test_y)
            
            # --- SVM (requires scaling) ---
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_pca)
            X_test_scaled = scaler.transform(X_test_pca)
            
            svm = SVMClassifier_wrapper()
            svm.train(X_train_scaled, y_subset)
            acc_svm = svm.evaluate(X_test_scaled, test_y)
            
            results.append({
                'Dataset': dataset_name,
                'Subset_Size': size,
                'Dimension': dim,
                'LDA_Accuracy': acc_lda,
                'SVM_Accuracy': acc_svm
            })
            print(f"  Dim {dim:3d} | LDA: {acc_lda:.4f} | SVM: {acc_svm:.4f}")

    # 5. Save and Plot
    output_dir = os.path.join('results', 'experiment1', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"experiment1_{dataset_name}_results.csv"), index=False)
    
    # Plot accuracy vs PCA dimension for each subset size
    # Convert Subset_Size to string for consistent discrete colors
    df_plot = df.copy()
    df_plot['Subset_Size'] = df_plot['Subset_Size'].astype(str)
    
    # LDA Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot, x='Dimension', y='LDA_Accuracy', hue='Subset_Size', palette='viridis', marker='o')
    plt.title(f'LDA Performance vs PCA Dimension on {dataset_name.upper()}')
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Test Accuracy')
    plt.legend(title='Subset Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(f"experiment1_{dataset_name}_lda_pca.png", directory=output_dir)
    
    # SVM Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot, x='Dimension', y='SVM_Accuracy', hue='Subset_Size', palette='magma', marker='o')
    plt.title(f'SVM Performance vs PCA Dimension on {dataset_name.upper()}')
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Test Accuracy')
    plt.legend(title='Subset Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(f"experiment1_{dataset_name}_svm_pca.png", directory=output_dir)
    
    # Comparison of Optimal Dimensions
    best_lda = df.loc[df.groupby('Subset_Size')['LDA_Accuracy'].idxmax()]
    best_svm = df.loc[df.groupby('Subset_Size')['SVM_Accuracy'].idxmax()]
    
    best_lda = best_lda[['Subset_Size', 'LDA_Accuracy']].rename(columns={'LDA_Accuracy': 'Accuracy'})
    best_lda['Model'] = 'LDA'
    best_svm = best_svm[['Subset_Size', 'SVM_Accuracy']].rename(columns={'SVM_Accuracy': 'Accuracy'})
    best_svm['Model'] = 'SVM'
    
    best_df = pd.concat([best_lda, best_svm])
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=best_df, x='Subset_Size', y='Accuracy', hue='Model', marker='o')
    plt.xscale('log')
    plt.title(f'LDA vs SVM Optimal Accuracy on {dataset_name}')
    plt.xlabel('Training Subset Size (log scale)')
    plt.ylabel('Test Accuracy')
    save_plot(f"experiment1_{dataset_name}_comparison.png", directory=output_dir)
    
    print(f"Experiment 1 on {dataset_name} complete.")

def run_experiment_1():
    datasets = ['mnist', 'fashion_mnist']
    for ds in datasets:
        run_experiment_1_on_dataset(ds)

if __name__ == "__main__":
    run_experiment_1()
