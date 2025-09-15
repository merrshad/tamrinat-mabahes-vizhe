import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam


num_words = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


def build_model(config_id):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=32))  
    model.add(GlobalAveragePooling1D())

    if config_id == 1:
        model.add(Dense(16, activation='relu'))

    elif config_id == 2:
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))

    elif config_id == 3:
        model.add(Dense(16, activation='tanh'))

    elif config_id == 4:
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))

    elif config_id == 5:
        model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

   
    if config_id == 5:
        optimizer = Adam(learning_rate=0.0005)
    else:
        optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


batch_sizes = {1: 512, 2: 512, 3: 512, 4: 512, 5: 128}
epochs = 10
results = []


for config_id in range(1, 6):
    print(f"\n🔧 Training Configuration {config_id}")
    model = build_model(config_id)
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_sizes[config_id],
        validation_split=0.2,
        verbose=0  
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Config {config_id} - Test Accuracy: {acc:.4f}")
    results.append((config_id, acc))


print("\n📋 Final Report (Test Accuracy):")
print("| Config ID | Description                       | Test Accuracy |")
print("|-----------|-----------------------------------|---------------|")

descriptions = {
    1: "Base model (16 relu)",
    2: "2 layers (32 relu + 16 relu)",
    3: "Tanh activation",
    4: "Dropout 0.5 added",
    5: "lr=0.0005 & batch=128"
}

for config_id, acc in results:
    desc = descriptions[config_id]
    print(f"|     {config_id}     | {desc:<33} |     {acc:.4f}     |")
