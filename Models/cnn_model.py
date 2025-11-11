import tensorflow as tf

def build_cnn(optimizer=None, input_shape=(28, 28, 1), num_classes=10):
    # Simple baseline CNN for Fashion-MNIST
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Explicit constant-LR SGD
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model