import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


# =========================
# CONFIGURATION
# =========================

DATASET_PATH = "dataset"
MODEL_PATH = "models/drawing_model.keras"

CLASSES = [
    "circle",
    "square",
    "triangle",
    "star"
]

SAMPLES_PER_CLASS = 2000

IMAGE_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 15


# =========================
# LOAD DATASET
# =========================

X = []
y = []

print("\nLoading dataset...\n")


for label, class_name in enumerate(CLASSES):

    file_path = os.path.join(
        DATASET_PATH,
        f"{class_name}.npy"
    )

    print(f"Loading {class_name}...")

    data = np.load(file_path)

    # Randomly select samples
    np.random.seed(42)

    indices = np.random.choice(
        len(data),
        SAMPLES_PER_CLASS,
        replace=False
    )

    data = data[indices]

    # Convert 784 pixels → 28 × 28
    data = data.reshape(
        -1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    X.append(data)

    # Create labels
    y.extend(
        [label] * SAMPLES_PER_CLASS
    )


# =========================
# COMBINE DATA
# =========================

X = np.concatenate(X)

y = np.array(y)

print("\nDataset loaded!")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# NORMALIZATION
# =========================

X = X.astype("float32") / 255.0


# =========================
# ADD CHANNEL
# =========================

X = X.reshape(
    -1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    1
)


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# BUILD CNN
# =========================

model = models.Sequential([

    layers.Input(
        shape=(28, 28, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        4,
        activation="softmax"
    )
])


# =========================
# COMPILE
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# SHOW MODEL
# =========================

model.summary()


# =========================
# TRAIN
# =========================

print("\nStarting training...\n")


history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================
# EVALUATE
# =========================

print("\nEvaluating model...\n")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================
# SAVE MODEL
# =========================

os.makedirs(
    "models",
    exist_ok=True
)

model.save(MODEL_PATH)

print(
    f"\nModel saved to: {MODEL_PATH}"
)