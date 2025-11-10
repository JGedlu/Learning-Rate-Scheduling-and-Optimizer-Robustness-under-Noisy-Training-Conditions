import tensorflow_datasets as tfds
import numpy as np

def load_fashion_mnist(normalize=True):
    """Loads Fashion-MNIST dataset from TensorFlow Datasets."""
    train_ds = tfds.as_numpy(tfds.load('fashion_mnist', split='train', batch_size=-1))
    test_ds = tfds.as_numpy(tfds.load('fashion_mnist', split='test', batch_size=-1))

    x_train, y_train = train_ds['image'], train_ds['label']
    x_test, y_test = test_ds['image'], test_ds['label']

    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)


def add_label_noise(y, noise_ratio, num_classes=10, seed=None):
    """Randomly replaces a percentage of labels with random ones."""
    rng = np.random.default_rng(seed)
    y_noisy = np.copy(y)
    n_noisy = int(noise_ratio * len(y))
    idx = rng.choice(len(y), n_noisy, replace=False)
    y_noisy[idx] = rng.integers(0, num_classes, n_noisy)
    return y_noisy


def get_noisy_datasets(noise_levels=(0.0, 0.1, 0.2), seed=42):
    """Returns datasets with multiple noise levels."""
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    datasets = {}

    for noise in noise_levels:
        y_noisy = add_label_noise(y_train, noise, seed=seed)
        datasets[noise] = (x_train, y_noisy)

    return datasets, (x_test, y_test)


if __name__ == "__main__":
    datasets, (x_test, y_test) = get_noisy_datasets()
    for noise_level, (x, y) in datasets.items():
        print(f"{int(noise_level*100)}% noise dataset: x={x.shape}, y={y.shape}")