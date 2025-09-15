from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers, optimizers, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)


def build_model(hidden_units=512, learning_rate=0.001, dropout_rate=0.0):
    model = models.Sequential()
    model.add(Input(shape=(784,)))  
    model.add(layers.Dense(hidden_units, activation='relu'))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


configs = [
    {"hidden_units": 256, "learning_rate": 0.001,
        "dropout_rate": 0.0, "batch_size": 128},
    {"hidden_units": 512, "learning_rate": 0.001,
        "dropout_rate": 0.3, "batch_size": 128},
    {"hidden_units": 512, "learning_rate": 0.0005,
        "dropout_rate": 0.3, "batch_size": 64},
    {"hidden_units": 1024, "learning_rate": 0.001,
        "dropout_rate": 0.5, "batch_size": 256}
]

results = []


for i, config in enumerate(configs):
    print(f"\n🧪 Training config {i+1}: {config}")
    model = build_model(config["hidden_units"],
                        config["learning_rate"], config["dropout_rate"])
    history = model.fit(train_images, train_labels_cat,
                        epochs=10,
                        batch_size=config["batch_size"],
                        validation_split=0.2,
                        verbose=0)
    val_acc = history.history["val_accuracy"][-1]
    results.append((config, val_acc))
    print(f"✅ Validation Accuracy: {val_acc:.4f}")


print("\n📊 Summary of Results:")
for i, (config, acc) in enumerate(results):
    print(f"Config {i+1}: {config}")
    print(f"Validation Accuracy: {acc:.4f}")
    print("-" * 40)


best_config = max(results, key=lambda x: x[1])
print("\n📈 Final Analysis:")
print(f"Best validation accuracy: {best_config[1]:.4f}")
print(f"Best configuration: {best_config[0]}")


dropout_effect = [r for r in results if r[0]["dropout_rate"] > 0]
no_dropout = [r for r in results if r[0]["dropout_rate"] == 0]

if dropout_effect and no_dropout:
    avg_with_dropout = sum([r[1]
                           for r in dropout_effect]) / len(dropout_effect)
    avg_no_dropout = sum([r[1] for r in no_dropout]) / len(no_dropout)
    if avg_with_dropout > avg_no_dropout:
        print("📌 Using dropout helped improve model performance.")
    else:
        print("📌 Dropout had little or negative impact on performance.")


print("\n📊 Impact of the number of hidden units:")
for config, acc in results:
    print(f"  - {config['hidden_units']} units → Accuracy: {acc:.4f}")
