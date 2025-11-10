import matplotlib.pyplot as plt
from Data.datasets import get_noisy_datasets
from Models.cnn_model import build_cnn
import tensorflow as tf
import os, json

# Loads datasets with varying noise levels
datasets, (x_test, y_test) = get_noisy_datasets(noise_levels=(0.0, 0.1, 0.2))
x_train, y_train = datasets[0.0]

# Reshapes for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# Builds & trains model
model = build_cnn()

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,            # increase to 15–20 for better accuracy
    batch_size=64,
    verbose=2
)

# Evaluates & plots data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy (SGD, clean data): {test_acc:.4f}")

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Baseline CNN – SGD (Constant LR)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Verifies CNN AI model
from Models.cnn_model import build_cnn
model = build_cnn()
model.summary()

os.makedirs("Results", exist_ok=True)
plt.savefig("Results/baseline_sgd_clean.png")

with open("Results/baseline_history.json", "w") as f:
    json.dump(history.history, f)

# Display dataset shapes for verification
for noise_level, (x_train, y_train) in datasets.items():
    print(f"Noise level: {int(noise_level * 100)}%")
    print(f"Train set shape: {x_train.shape}, Labels shape: {y_train.shape}")