import os
import io
import pickle
import logging
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import json

# ── Load env ──
load_dotenv()

# ── Validate env ──
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file. Please set it before running.")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
CORS(app)

# ── Logging ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# TRAIN & SAVE MODELS (run once)
# ─────────────────────────────────────────

def train_crop_model():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open("crop_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Crop model saved!")

def train_disease_model():
    IMG_SIZE = 224

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32
    )

    class_names = dataset.class_names
    print(f"Classes found: {class_names}")

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
    print("Disease model saved!")

# ── Uncomment these two lines ONCE to train, then comment back ──
# train_crop_model()
# train_disease_model()

# ─────────────────────────────────────────
# LOAD MODELS (with safety checks)
# ─────────────────────────────────────────

crop_model = None
disease_model = None

# Load class labels
if os.path.exists("plant_classes.json"):
    with open("plant_classes.json", "r") as f:
        CLASSES = json.load(f)
else:
    CLASSES = [
        "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
        "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    ]
    logger.warning("⚠️ plant_classes.json not found. Using hardcoded CLASSES list.")

IMG_SIZE = 224

if os.path.exists("crop_model.pkl"):
    with open("crop_model.pkl", "rb") as f:
        crop_model = pickle.load(f)
    logger.info("✅ Crop model loaded successfully.")
else:
    logger.warning("⚠️ crop_model.pkl not found. /predict_crop will not work. Run train_crop_model() first.")

if os.path.exists("plant_model.h5"):
    disease_model = tf.keras.models.load_model("plant_model.h5")
    logger.info("✅ Disease model loaded successfully.")

    output_classes = disease_model.output_shape[-1]
    if output_classes != len(CLASSES):
        raise ValueError(
            f"Model outputs {output_classes} classes, but CLASSES has {len(CLASSES)} labels."
        )

else:
    logger.warning("⚠️ plant_model.h5 not found. /predict_disease will not work. Run train_disease_model() first.")
# ─────────────────────────────────────────
# 0. Health Check
# ─────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "crop_model_loaded": crop_model is not None,
        "disease_model_loaded": disease_model is not None
    })

# ─────────────────────────────────────────
# 1. Crop Recommendation
# ─────────────────────────────────────────

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        if crop_model is None:
            return jsonify({"error": "Crop model not loaded. Train the model first."}), 503

        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        features = [[
            float(data["N"]), float(data["P"]), float(data["K"]),
            float(data["temperature"]), float(data["humidity"]),
            float(data["ph"]), float(data["rainfall"])
        ]]
        result = crop_model.predict(features)
        return jsonify({"recommended_crop": result[0]})

    except ValueError as e:
        return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error in /predict_crop: {str(e)}")
        return jsonify({"error": "Internal server error."}), 500

# ─────────────────────────────────────────
# 2. Disease Detection
# ─────────────────────────────────────────

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        if disease_model is None:
            return jsonify({"error": "Disease model not loaded. Train the model first."}), 503

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded. Send an image file with key 'image'."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename. Please select a valid image."}), 400

        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # IMPORTANT:
        # Do NOT divide by 255 here because the model already has Rescaling(1./255)
        img_array = np.asarray(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = disease_model.predict(img_array, verbose=0)[0]

        top_index = int(np.argmax(prediction))
        result = CLASSES[top_index]
        confidence = round(float(prediction[top_index]) * 100, 2)

        top_predictions = []
        for i in np.argsort(prediction)[::-1][:5]:
            top_predictions.append({
                "disease": CLASSES[int(i)],
                "confidence": round(float(prediction[int(i)]) * 100, 2)
            })

        return jsonify({
            "disease": result,
            "confidence": confidence,
            "top_predictions": top_predictions
        })

    except Exception as e:
        logger.error(f"Error in /predict_disease: {str(e)}")
        return jsonify({"error": "Failed to process image. Please try again."}), 500

# ─────────────────────────────────────────
# 3. Chatbot (Groq LLM)
# ─────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "No message provided."}), 400

        user_message = data["message"].strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        history = data.get("history", [])

        messages = [{
            "role": "system",
            "content": "You are a helpful agriculture assistant. Help farmers with crop recommendations, plant diseases, soil health, irrigation, and farming advice. Keep responses concise and practical."
        }]
        messages += history
        messages.append({"role": "user", "content": user_message})

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=300
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({"error": "Chatbot failed. Please try again."}), 500

# ─────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)