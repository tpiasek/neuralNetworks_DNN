# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Parametry
global_config = {
    "image_size": (128, 128),  # Rozmiar obrazu
    "batch_size": 32,         # Rozmiar mini-batch
    "learning_rate": 0.001,    # Początkowy współczynnik uczenia
    "momentum": 0.9,          # Wartość dla momentum
    "validation_split": 0.2,  # Proporcja danych walidacyjnych
    "error_target": 0.01      # Docelowy błąd MSE
}

# Przygotowanie danych (zakładamy strukturę katalogów: train/with_tumor, train/without_tumor)
data_dir = "train/"  # Zmień na ścieżkę do zbioru danych

image_size = global_config["image_size"]
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=global_config["validation_split"],
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15,
)

train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=global_config["batch_size"],
    class_mode="binary",
    subset="training",
)

val_data = data_gen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=global_config["batch_size"],
    class_mode="binary",
    subset="validation",
)

# Budowa modelu konwolucyjnego (CNN)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

# Kompilacja modelu CNN
optimizer_cnn = SGD(learning_rate=global_config["learning_rate"], momentum=global_config["momentum"])
cnn_model.compile(optimizer='Adam', loss="mse", metrics=["mse"])

# Budowa modelu w pełni gęstego (Dense)
input_dim = image_size[0] * image_size[1] * 3
dense_model = Sequential([
    Flatten(input_shape=(image_size[0], image_size[1], 3)),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid"),
])

# Kompilacja modelu Dense
optimizer_dense = SGD(learning_rate=global_config["learning_rate"], momentum=global_config["momentum"])
dense_model.compile(optimizer=optimizer_dense, loss="mse", metrics=["mse"])

# Callback dla wcześniejszego zatrzymania
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("mse") <= global_config["error_target"]:
            print(f"\nOsiągnięto docelowy błąd MSE: {logs.get('mse'):.4f}. Zatrzymywanie uczenia.")
            self.model.stop_training = True

# Trenowanie modelu CNN
print("\nTrenowanie modelu CNN...")
cnn_callbacks = [
    CustomEarlyStopping(),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
]

cnn_history = cnn_model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=cnn_callbacks,
    verbose=1,
)

# Trenowanie modelu Dense
print("\nTrenowanie modelu Dense...")
dense_callbacks = [
    CustomEarlyStopping(),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
]

dense_history = dense_model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=dense_callbacks,
    verbose=1,
)

# Zapis modeli
cnn_model.save("cnn_brain_tumor_model.keras")
dense_model.save("dense_brain_tumor_model.keras")

# Testowanie modeli
def evaluate_model(model, test_data, model_name):
    print(f"\nTestowanie modelu {model_name}...")
    test_loss, test_mse = model.evaluate(test_data)
    print(f"Błąd na zbiorze testowym (MSE) dla {model_name}: {test_mse:.4f}")
    return test_mse

# Wczytanie testowych danych (zakładamy strukturę katalogów: test/with_tumor, test/without_tumor)
test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_data = test_data_gen.flow_from_directory(
    data_dir.replace("train", "test"),
    target_size=image_size,
    batch_size=global_config["batch_size"],
    class_mode="binary",
)

cnn_test_mse = evaluate_model(cnn_model, test_data, "CNN")
dense_test_mse = evaluate_model(dense_model, test_data, "Dense")

# Wizualizacja wyników
plt.figure(figsize=(12, 6))

# Wykres błędu dla CNN
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history["mse"], label="CNN MSE trenowanie")
plt.plot(cnn_history.history["val_mse"], label="CNN MSE walidacja")
plt.title("Błąd MSE dla modelu CNN")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.legend()

# Wykres błędu dla Dense
plt.subplot(1, 2, 2)
plt.plot(dense_history.history["mse"], label="Dense MSE trenowanie")
plt.plot(dense_history.history["val_mse"], label="Dense MSE walidacja")
plt.title("Błąd MSE dla modelu Dense")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.legend()

plt.tight_layout()
plt.show()

# Porównanie skuteczności modeli
predictions_cnn = cnn_model.predict(test_data)
predictions_dense = dense_model.predict(test_data)

predicted_classes_cnn = (predictions_cnn > 0.5).astype(int)
predicted_classes_dense = (predictions_dense > 0.5).astype(int)

true_classes = test_data.classes

accuracy_cnn = np.mean(predicted_classes_cnn.flatten() == true_classes) * 100
accuracy_dense = np.mean(predicted_classes_dense.flatten() == true_classes) * 100

print(f"Ostateczna skuteczność modelu CNN: {accuracy_cnn:.2f}%")
print(f"Ostateczna skuteczność modelu Dense: {accuracy_dense:.2f}%")
