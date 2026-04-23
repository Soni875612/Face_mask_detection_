"""
app.py - Flask Web Server for Face Mask Detection
"""

import os
import io
import base64
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import cv2
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("FaceMaskApp")


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL = None
FACE_CASCADE = None
CONFIG = {
    "IMAGE_SIZE": (224, 224),
    "CLASSES": ["WithMask", "WithoutMask", "MaskWornIncorrectly"],
    "COLORS": {
        "WithMask": "#00c864",
        "WithoutMask": "#e03030",
        "MaskWornIncorrectly": "#ff9900",
    },
    "RISK": {
        "WithMask": "LOW",
        "WithoutMask": "HIGH",
        "MaskWornIncorrectly": "MEDIUM",
    },
    "CONFIDENCE_THRESHOLD": 0.60,
}

SESSION = {
    "frames": 0,
    "total_faces": 0,
    "with_mask": 0,
    "without_mask": 0,
    "incorrect": 0,
    "alerts": 0,
    "start_time": time.time(),
}


def load_model_once(model_path: str = None):
    global MODEL, FACE_CASCADE

    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    FACE_CASCADE = cv2.CascadeClassifier(xml)
    log.info("Haar cascade loaded.")

    if model_path and os.path.exists(model_path):
        log.info(f"Loading model: {model_path}")
        MODEL = tf.keras.models.load_model(model_path)
        log.info("Model loaded successfully.")
        return True
    else:
        log.warning("No trained model found. Running in DEMO mode.")
        MODEL = None
        return False


def decode_image(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def detect_faces(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    return faces if len(faces) > 0 else []


def classify_face(frame: np.ndarray, bbox) -> dict:
    x, y, w, h = bbox
    pad = 15
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return {"label": "Unknown", "confidence": 0.0}

    if MODEL is None:
        import random
        label = random.choices(CONFIG["CLASSES"], weights=[0.6, 0.3, 0.1])[0]
        conf = round(random.uniform(0.70, 0.97), 3)
    else:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resz = cv2.resize(roi_rgb, tuple(CONFIG["IMAGE_SIZE"]))
        roi_norm = roi_resz.astype("float32") / 255.0
        roi_exp = np.expand_dims(roi_norm, axis=0)
        probs = MODEL.predict(roi_exp, verbose=0)[0]
        
        # Model sirf 2 classes jaanta hai: 0=WithMask, 1=WithoutMask
        with_mask_prob = float(probs[0])
        without_mask_prob = float(probs[1])
        
        if max(probs) < CONFIG["CONFIDENCE_THRESHOLD"]:
            label = "Uncertain"
            conf = float(max(probs))
        elif with_mask_prob > without_mask_prob:
            label = "WithMask"
            conf = with_mask_prob
        else:
            label = "WithoutMask"
            conf = without_mask_prob

    return {
        "label": label,
        "confidence": round(conf, 3),
        "risk": CONFIG["RISK"].get(label, "LOW"),
        "color": CONFIG["COLORS"].get(label, "#888888"),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
    }


@app.route("/")
def index():
    model_loaded = MODEL is not None
    return render_template("index.html",
                           model_loaded=model_loaded,
                           classes=CONFIG["CLASSES"])


@app.route("/api/detect", methods=["POST"])
def api_detect():
    global SESSION

    try:
        data = request.get_json()
        image_b64 = data.get("image", "")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        frame = decode_image(image_b64)
        faces = detect_faces(frame)

        detections = []
        unmasked_cnt = 0

        for bbox in faces:
            result = classify_face(frame, bbox)
            detections.append(result)
            if result["label"] == "WithoutMask":
                unmasked_cnt += 1
                SESSION["without_mask"] += 1
            elif result["label"] == "WithMask":
                SESSION["with_mask"] += 1
            elif result["label"] == "MaskWornIncorrectly":
                SESSION["incorrect"] += 1

        SESSION["frames"] += 1
        SESSION["total_faces"] += len(faces)
        alert = unmasked_cnt >= 1
        if alert:
            SESSION["alerts"] += 1

        det_total = SESSION["with_mask"] + SESSION["without_mask"] + SESSION["incorrect"]
        compliance = round(SESSION["with_mask"] / det_total * 100, 1) if det_total else 100.0

        return jsonify({
            "detections": detections,
            "alert": alert,
            "stats": {
                "frames": SESSION["frames"],
                "total_faces": SESSION["total_faces"],
                "with_mask": SESSION["with_mask"],
                "without_mask": SESSION["without_mask"],
                "incorrect": SESSION["incorrect"],
                "alerts": SESSION["alerts"],
                "compliance": compliance,
                "uptime_s": round(time.time() - SESSION["start_time"], 1),
            }

            
        })

    except Exception as e:
        log.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global SESSION
    SESSION = {
        "frames": 0, "total_faces": 0, "with_mask": 0,
        "without_mask": 0, "incorrect": 0, "alerts": 0,
        "start_time": time.time(),
    }
    return jsonify({"status": "reset"})


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded": MODEL is not None,
        "classes": CONFIG["CLASSES"],
        "mode": "live" if MODEL else "demo",
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="saved_model/face_mask_model.h5",
                        help="Path to trained model file")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    load_model_once(args.model)

    print("\n" + "="*60)
    print("  FACE MASK DETECTION - WEB INTERFACE")
    print("="*60)
    print(f"  Open in browser:  http://localhost:{args.port}")
    print(f"  Model status  :  {'Loaded' if MODEL else 'DEMO mode (no model)'}")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")

    app.run(host=args.host, port=args.port, debug=False)