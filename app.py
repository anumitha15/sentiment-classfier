from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import joblib
from pymongo import MongoClient
import certifi
import os

app = Flask(__name__)
CORS(app)

# Load ML model
model = joblib.load("model.pkl")

# MongoDB connection (SSL FIX)
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(
    mongo_uri,
    tlsCAFile=certifi.where()
)
db = client["ai_text_db"]
collection = db["logs"]

# -------------------------
# 🔥 PREDICT + SAVE
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"prediction": "No input received"})

        # ML prediction
        prediction = model.predict([text])[0]
        prediction = str(prediction)

        # Save to MongoDB
        collection.insert_one({
            "text": text,
            "prediction": prediction
        })

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        print("🔥 FULL ERROR:", str(e))
        return jsonify({
            "prediction": "Server error"
        })

# -------------------------
# 📊 HISTORY
# -------------------------
@app.route("/history", methods=["GET"])
def history():
    logs = list(collection.find({}, {"_id": 0}))
    return jsonify(logs)

# -------------------------
# HOME
# -------------------------

# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5001)
