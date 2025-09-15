from tensorflow.keras.datasets import reuters
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


(train_data, train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)


configs = [
    [64, 64],
    [128, 64],
    [128, 128, 64],
    [256, 128, 64],
    [512, 256, 128, 64]
]

results_summary = []


for i, config in enumerate(configs, 1):
    print(f"\n🔧 Config {i}: {config}")
    model = keras.Sequential()
    model.add(layers.Input(shape=(10000,)))
    for units in config:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(46, activation="softmax"))

    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        x_train, y_train,
        epochs=9,
        batch_size=512,
        verbose=0,
        validation_split=0.2
    )

    val_acc = history.history["val_accuracy"][-1]
    results_summary.append((config, val_acc))
    print(f"✅ Validation Accuracy: {val_acc:.4f}")


print("\n📊 Final Evaluation Report")
print("-" * 40)
for i, (config, acc) in enumerate(results_summary, 1):
    print(f"Config {i}: {config} => Validation Accuracy: {acc:.4f}")
