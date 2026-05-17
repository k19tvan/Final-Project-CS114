import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

def load_dataset(name):
    """
    Loads specified dataset.
    name: 'mnist', 'fashion_mnist', or 'cifar10'
    Returns: (train_X, train_y), (test_X, test_y)
    """
    print(f"Loading {name} data...")
    if name == 'mnist':
        return mnist.load_data()
    elif name == 'fashion_mnist':
        return fashion_mnist.load_data()
    elif name == 'cifar10':
        (train_X, train_y), (test_X, test_y) = cifar10.load_data()
        # Convert to grayscale: gray = 0.2989*R + 0.5870*G + 0.1140*B
        train_X_gray = 0.2989 * train_X[:,:,:,0] + 0.5870 * train_X[:,:,:,1] + 0.1140 * train_X[:,:,:,2]
        test_X_gray = 0.2989 * test_X[:,:,:,0] + 0.5870 * test_X[:,:,:,1] + 0.1140 * test_X[:,:,:,2]
        return (train_X_gray.astype(np.uint8), train_y.flatten()), (test_X_gray.astype(np.uint8), test_y.flatten())
    else:
        raise ValueError("Unknown dataset name")

def load_mnist_data():
    return mnist.load_data()

def preprocess_data(train_X, test_X):
    """
    Standard preprocessing: flattening or other common steps if needed.
    """
    # For now, we keep them as 28x28 for HOG, but flattened versions might be needed for PCA/LDA
    return train_X, test_X
