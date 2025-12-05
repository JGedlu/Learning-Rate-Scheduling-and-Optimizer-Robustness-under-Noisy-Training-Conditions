import matplotlib.pyplot as plt
from Data.datasets import get_noisy_datasets
from Models.cnn_model import build_cnn
import tensorflow as tf
import os, json
import numpy as np
import seaborn as sns

tf.config.run_functions_eagerly(True)

# ------------------------------
# Optimizer & LR Schedule Factory
# ------------------------------
def get_schedulers_and_optimizers():
    return {
        'SGD_Constant': tf.keras.optimizers.SGD(learning_rate=0.01),
        'Adam_Constant': tf.keras.optimizers.Adam(learning_rate=0.001),

        'SGD_StepDecay': tf.keras.optimizers.SGD(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=2000,
                decay_rate=0.5
            )
        ),

        'Adam_StepDecay': tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=2000,
                decay_rate=0.5
            )
        ),

        'SGD_Cosine': tf.keras.optimizers.SGD(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.01,
                first_decay_steps=2000
            )
        ),

        'Adam_Cosine': tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.001,
                first_decay_steps=2000
            )
        )
    }


# ------------------------------
# Load datasets: 0%, 10%, 20% noise
# ------------------------------
datasets, (x_test, y_test) = get_noisy_datasets(noise_levels=(0.0, 0.1, 0.2))

x_test = x_test.reshape(-1, 28, 28, 1)

# ------------------------------
# CLEAN TRAINING (0% noise)
# ------------------------------
print("\n====== CLEAN DATA TRAINING (0% noise) ======\n")

clean_x, clean_y = datasets[0.0]
clean_x = clean_x.reshape(-1, 28, 28, 1)

clean_results = {}
optimizers_clean = get_schedulers_and_optimizers()  # fresh factory

for name, optimizer in optimizers_clean.items():
    print(f"\n --- Training {name} on CLEAN data ---")
    model = build_cnn(optimizer=optimizer)

    history = model.fit(
        clean_x, clean_y,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=64,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final CLEAN Accuracy ({name}): {test_acc:.4f}")

    clean_results[name] = float(test_acc)

# Save
os.makedirs("Results", exist_ok=True)
with open("Results/clean_results.json", "w") as f:
    json.dump(clean_results, f, indent=4)


# ------------------------------
# NOISY TRAINING (10% & 20%)
# ------------------------------
noise_results = {}

for noise_level in (0.1, 0.2):
    print(f"\n====== TRAINING ON {int(noise_level*100)}% NOISE ======\n")

    x_noise, y_noise = datasets[noise_level]
    x_noise = x_noise.reshape(-1, 28, 28, 1)

    noise_results[noise_level] = {}

    # NEW optimizer factory for this noise level
    optimizers_noise = get_schedulers_and_optimizers()

    for name, optimizer in optimizers_noise.items():
        print(f"\n --- Training {name} on {int(noise_level*100)}% noise ---")

        model = build_cnn(optimizer=optimizer)

        model.fit(
            x_noise, y_noise,
            validation_data=(x_test, y_test),
            epochs=10,
            batch_size=64,
            verbose=2
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Final Accuracy ({name}) on {int(noise_level*100)}% noise: {test_acc:.4f}")

        noise_results[noise_level][name] = float(test_acc)

# Save noisy results
with open("Results/noisy_results.json", "w") as f:
    json.dump(noise_results, f, indent=4)


# ------------------------------
# GRAPH GENERATION
# ------------------------------

os.makedirs("Results", exist_ok=True)

# Combine into single structure for plotting
all_noise_levels = [0.0, 0.1, 0.2]
full_results = {
    0.0: clean_results,
    0.1: noise_results[0.1],
    0.2: noise_results[0.2]
}


# 1. Accuracy vs Noise (line plot)
plt.figure(figsize=(10, 6))
for opt_name in clean_results.keys():
    accs = [full_results[n][opt_name] for n in all_noise_levels]
    plt.plot([n*100 for n in all_noise_levels], accs, marker='o', linewidth=2, label=opt_name)

plt.title("Accuracy vs Label Noise – All Optimizers")
plt.xlabel("Noise Level (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("Results/accuracy_vs_noise.png")
plt.close()


# 2. Accuracy bars per noise level
for noise in all_noise_levels:
    plt.figure(figsize=(10,6))
    plt.bar(full_results[noise].keys(), full_results[noise].values())
    plt.title(f"Accuracy at {int(noise*100)}% Noise")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Results/accuracy_bar_{int(noise*100)}.png")
    plt.close()


# 3. Robustness drop (clean → 20%)
plt.figure(figsize=(10,6))
for opt_name in clean_results.keys():
    drop = full_results[0.0][opt_name] - full_results[0.2][opt_name]
    plt.bar(opt_name, drop)

plt.title("Accuracy Drop from 0% → 20% Noise (Robustness)")
plt.ylabel("Drop in Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Results/robustness_drop.png")
plt.close()


# 4. Heatmap of accuracy
heatmap_data = np.array([
    [full_results[n][opt] for opt in clean_results.keys()]
    for n in all_noise_levels
])

plt.figure(figsize=(10,6))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="viridis",
    xticklabels=clean_results.keys(),
    yticklabels=[f"{int(n*100)}%" for n in all_noise_levels],
    fmt=".3f"
)
plt.title("Accuracy Heatmap – Optimizer × Noise Level")
plt.savefig("Results/accuracy_heatmap.png")
plt.close()


# Summary
print("\n=== FINAL SUMMARY ===")
for noise in all_noise_levels:
    print(f"\n{int(noise*100)}% Noise:")
    for opt, acc in full_results[noise].items():
        print(f"  {opt:15s} Accuracy: {acc:.4f}")
