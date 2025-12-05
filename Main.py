import matplotlib.pyplot as plt
from Data.datasets import get_noisy_datasets
from Models.cnn_model import build_cnn
import tensorflow as tf
import os, json

# Define learning-rate schedules and optimizers
def get_schedulers_and_optimizers():
    # Constant LR (control)
    sgd_const = tf.keras.optimizers.SGD(learning_rate=0.01)
    adam_const = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Step Decay: LR reduced by factor every few epochs
    step_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=2000,
        decay_rate=0.5
    )
    sgd_step = tf.keras.optimizers.SGD(learning_rate=step_decay)

    step_decay_adam = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=2000,
        decay_rate=0.5
    )
    adam_step = tf.keras.optimizers.Adam(learning_rate=step_decay_adam)

    # Cosine Annealing (smooth cyclical decay)
    cosine_sgd = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.01,
            first_decay_steps=2000
        )
    )
    cosine_adam = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001,
            first_decay_steps=2000
        )
    )

    return {
        'SGD_Constant': sgd_const,
        'Adam_Constant': adam_const,
        'SGD_StepDecay': sgd_step,
        'Adam_StepDecay': adam_step,
        'SGD_Cosine': cosine_sgd,
        'Adam_Cosine': cosine_adam
    }

# Loads datasets with varying noise levels
datasets, (x_test, y_test) = get_noisy_datasets(noise_levels=(0.0, 0.1, 0.2))
x_train, y_train = datasets[0.0]

# Reshapes for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# Loops through optimizers/schedulers and train on clean data
optimizers = get_schedulers_and_optimizers()
results = {}

for name, optimizer in optimizers.items():
    # Builds & trains model
    print(f"\n --- Training with {name} ")
    model = build_cnn(optimizer=optimizer)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,          # testing: increase to 15–20 for better accuracy
        batch_size=64,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"--- Final Test Accuracy ({name}): {test_acc:.4f}")

    results[name] = {
        'history': history.history,
        'test_acc': float(test_acc),
        'test_loss': float(test_loss)
    }

    # Plots each training curve data
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'CNN – {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    os.makedirs("Results", exist_ok=True)
    plt.savefig(f"Results/{name}_curve.png")
    plt.close()

# Saves summarized results
with open("Results/clean_data_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Prints summary table
print("\n SUMMARY (Clean Data)")
for name, res in results.items():
    print(f"{name:15s}  Test Acc: {res['test_acc']:.4f}")