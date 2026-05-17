import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from skimage.transform import rotate, AffineTransform, warp
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from src.data_loader import load_dataset
from src.models import LDAClassifier_wrapper, SVMClassifier_wrapper
from src.utils import extract_hog_features, save_plot

def augment_image(image, intensity='medium'):
    if intensity == 'low':
        rot, trans, zoom = random.uniform(-5, 5), (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)), random.uniform(0.95, 1.05)
    elif intensity == 'medium':
        rot, trans, zoom = random.uniform(-15, 15), (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)), random.uniform(0.9, 1.1)
    else: # high
        rot, trans, zoom = random.uniform(-30, 30), (random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)), random.uniform(0.8, 1.2)

    img_2d = image[:, :, 0] if len(image.shape) == 3 and image.shape[2] == 1 else image
    aug_img = rotate(img_2d, rot, mode='edge')
    h, w = img_2d.shape
    t_x, t_y = trans[0] * w, trans[1] * h
    c_x, c_y = w / 2, h / 2
    shift_x, shift_y = c_x - c_x * zoom, c_y - c_y * zoom
    tf = AffineTransform(translation=(t_x + shift_x, t_y + shift_y), scale=(zoom, zoom))
    aug_img = warp(aug_img, tf.inverse, mode='edge')
    return aug_img[:, :, np.newaxis] if len(image.shape) == 3 and image.shape[2] == 1 else aug_img

def get_tta_predictions(images, model, scaler, pixels_per_cell, cells_per_block, orientations, n_augmentations=10, intensity='medium', use_scaler=False):
    """
    Đánh giá TTA đảm bảo không làm lệch nhãn.
    """
    all_probs = []
    all_single_preds = []
    all_consistency = []
    
    # Không shuffle ở đây để giữ đúng thứ tự với nhãn truyền vào
    for i in range(len(images)):
        img = images[i]
        
        # 1. Dự đoán đơn (Single)
        feat_single = extract_hog_features(np.array([img]), pixels_per_cell, cells_per_block, orientations)
        if use_scaler and scaler:
            feat_single = scaler.transform(feat_single)
        
        prob_single = model.predict_proba(feat_single)[0]
        pred_single = np.argmax(prob_single)
        all_single_preds.append(pred_single)
        
        # 2. Dự đoán TTA
        tta_probs = []
        tta_preds = []
        for _ in range(n_augmentations):
            aug_img = augment_image(img, intensity)
            feat_aug = extract_hog_features(np.array([aug_img]), pixels_per_cell, cells_per_block, orientations)
            if use_scaler and scaler:
                feat_aug = scaler.transform(feat_aug)
            
            p = model.predict_proba(feat_aug)[0]
            tta_probs.append(p)
            tta_preds.append(np.argmax(p))
            
        all_probs.append(np.mean(tta_probs, axis=0))
        all_consistency.append(np.mean(np.array(tta_preds) == pred_single))
        
        if (i+1) % 100 == 0:
            print(f"        Processed {i+1}/{len(images)} samples...")
            
    return np.array(all_probs), np.array(all_single_preds), np.array(all_consistency)

def visualize_tta_samples(image, intensity, dataset_name, output_dir):
    """
    Visualize original image and its 10 augmented versions.
    """
    n_augmentations = 10
    plt.figure(figsize=(15, 5))
    
    # Original
    plt.subplot(2, 6, 1)
    img_display = image[:, :, 0] if len(image.shape) == 3 and image.shape[2] == 1 else image
    plt.imshow(img_display, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Augmented versions
    for i in range(n_augmentations):
        aug_img = augment_image(image, intensity)
        aug_display = aug_img[:, :, 0] if len(aug_img.shape) == 3 and aug_img.shape[2] == 1 else aug_img
        plt.subplot(2, 6, i + 2)
        plt.imshow(aug_display, cmap='gray')
        plt.title(f"Aug {i+1}")
        plt.axis('off')
        
    plt.suptitle(f"TTA Augmentations - Intensity: {intensity.upper()} ({dataset_name.upper()})")
    plt.tight_layout()
    save_plot(f"tta_samples_{intensity}.png", directory=output_dir)

def run_experiment_4_on_dataset(dataset_name):
    print(f"\n" + "="*60)
    print(f"🚀 BẮT ĐẦU THÍ NGHIỆM 4 (SỬA LỖI): LDA VS SVM TRÊN {dataset_name.upper()}")
    print("="*60)
    
    (train_X, train_y), (test_X, test_y) = load_dataset(dataset_name)
    pixels_per_cell = (7, 7) if dataset_name in ['mnist', 'fashion_mnist'] else (8, 8)
    cells_per_block, orientations = (2, 2), 9
    
    # Sử dụng toàn bộ tập test
    X_test_sub = test_X
    y_test_sub = test_y

    print(f"[1/4] Huấn luyện mô hình...")
    train_X_hog = extract_hog_features(train_X, pixels_per_cell, cells_per_block, orientations)
    
    # LDA
    print("      >> Training LDA model...")
    lda = LDAClassifier_wrapper()
    lda.fit(train_X_hog, train_y)
    print("      >> LDA trained.")
    
    # SVM
    print("      >> Training SVM model (with Calibration)...")
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X_hog)
    base_svm = SVMClassifier_wrapper().model
    svm_calibrated = CalibratedClassifierCV(base_svm, cv=3)
    svm_calibrated.fit(train_X_scaled, train_y)
    print("      >> SVM trained.")
    
    intensities = ['low', 'medium', 'high']
    results = []
    
    for intensity in intensities:
        print(f"\n  Intensity Level: {intensity.upper()}")
        
        # LDA TTA
        print(f"      [LDA] Evaluating Test-Time Augmentation...")
        probs_l, preds_s_l, consist_l = get_tta_predictions(X_test_sub, lda, None, pixels_per_cell, cells_per_block, orientations, intensity=intensity, use_scaler=False)
        acc_s_l, acc_t_l = np.mean(preds_s_l == y_test_sub), np.mean(np.argmax(probs_l, axis=1) == y_test_sub)
        results.append({'Intensity': intensity, 'Method': 'LDA', 'Single_Acc': acc_s_l, 'TTA_Acc': acc_t_l, 'Consistency': np.mean(consist_l)})
        
        # SVM TTA
        print(f"      [SVM] Evaluating Test-Time Augmentation...")
        probs_s, preds_s_s, consist_s = get_tta_predictions(X_test_sub, svm_calibrated, scaler, pixels_per_cell, cells_per_block, orientations, intensity=intensity, use_scaler=True)
        acc_s_s, acc_t_s = np.mean(preds_s_s == y_test_sub), np.mean(np.argmax(probs_s, axis=1) == y_test_sub)
        results.append({'Intensity': intensity, 'Method': 'SVM', 'Single_Acc': acc_s_s, 'TTA_Acc': acc_t_s, 'Consistency': np.mean(consist_s)})
        
        print(f"      Acc -> LDA -> Single: {acc_s_l:.4f}, TTA: {acc_t_l:.4f}, Consist: {np.mean(consist_l):.4f}")
        print(f"      Acc -> SVM -> Single: {acc_s_s:.4f}, TTA: {acc_t_s:.4f}, Consist: {np.mean(consist_s):.4f}")

    # 4. Lưu và Vẽ biểu đồ
    output_dir = os.path.join('results/experiment4', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize samples for each intensity
    for intensity in intensities:
        visualize_tta_samples(test_X[0], intensity, dataset_name, output_dir)
        
    pd.DataFrame(results).to_csv(os.path.join(output_dir, f"experiment4_{dataset_name}_results.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pd.DataFrame(results), x='Intensity', y='TTA_Acc', hue='Method', marker='o')
    plt.title(f'TTA Accuracy (Fixed): LDA vs SVM ({dataset_name.upper()})')
    save_plot(f"experiment4_{dataset_name}_tta_accuracy.png", directory=output_dir)

def run_experiment_4():
    for ds in ['mnist', 'fashion_mnist']:
        run_experiment_4_on_dataset(ds)

if __name__ == "__main__":
    run_experiment_4()
