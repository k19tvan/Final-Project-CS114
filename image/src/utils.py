import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.feature import hog

def extract_hog_features(data, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    hog_feats = []
    for img in data:
        fd = hog(img, orientations=orientations,
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block,
                 visualize=False)
        hog_feats.append(fd)
    return np.array(hog_feats)

def add_label_noise(y, level):
    if level == 0: return y.copy()
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(level * n_samples)
    idx_to_flip = np.random.choice(n_samples, n_noisy, replace=False)
    y_noisy[idx_to_flip] = np.random.randint(0, 10, size=n_noisy)
    return y_noisy

def add_feature_noise(X, sigma):
    if sigma == 0: return X.copy()
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

def save_plot(filename, directory='results'):
    """
    Saves the matplotlib plot to a file instead of showing it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.close()
