"""
Train both models and save them.
Run this script ONCE from the project directory:
    python train_models.py
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# ─────────────────────────
# 1. Train Crop Model
# ─────────────────────────
print("=" * 50)
print("Training Crop Recommendation Model...")
print("=" * 50)

df = pd.read_csv("Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Crop model accuracy: {accuracy * 100:.2f}%")

with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ crop_model.pkl saved!\n")

# ─────────────────────────
# 2. Train Disease Model
# ─────────────────────────
print("=" * 50)
print("Training Plant Disease Detection Model...")
print("=" * 50)

import tensorflow as tf

IMG_SIZE = 224

if not os.path.exists("dataset"):
    print("❌ 'dataset/' folder not found! Please extract your dataset first.")
    exit(1)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32
)

class_names = dataset.class_names
print(f"Classes found: {class_names}")

with open("plant_classes.json", "w") as f:
    json.dump(class_names, f, indent=2)

print("plant_classes.json saved!")

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(dataset, epochs=5)
model.save("plant_model.h5")
print("✅ plant_model.h5 saved!\n")

print("=" * 50)
print("🎉 Both models trained and saved successfully!")
print("=" * 50)
