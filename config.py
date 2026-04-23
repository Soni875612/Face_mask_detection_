import os

# ─── Base Directory ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset Paths ────────────────────────────────────────
DATASET_PATH     = os.path.join(BASE_DIR, "Dataset", "images")
ANNOTATIONS_PATH = os.path.join(BASE_DIR, "Dataset", "annotations")
WITH_MASK_PATH   = os.path.join(BASE_DIR, "Dataset", "images", "with_mask")
WITHOUT_MASK_PATH= os.path.join(BASE_DIR, "Dataset", "images", "without_mask")

# ─── Model Paths ──────────────────────────────────────────
SAVED_MODEL_DIR  = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH       = os.path.join(SAVED_MODEL_DIR, "face_mask_model.h5")

# ─── Flask Upload Path ────────────────────────────────────
UPLOAD_FOLDER    = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ─── Model Hyperparameters ────────────────────────────────
IMG_SIZE         = 224          # Image resize size (224x224)
BATCH_SIZE       = 32
EPOCHS           = 20
LEARNING_RATE    = 0.0001
VALIDATION_SPLIT = 0.2

# ─── Detection Settings ───────────────────────────────────
CONFIDENCE_THRESHOLD = 0.6      # 60% se upar tabhi result dikhao
MASK_LABEL           = "Mask"
NO_MASK_LABEL        = "No Mask"

# ─── Colors (BGR format for OpenCV) ───────────────────────
GREEN = (0, 255, 0)             # Mask detected
RED   = (0, 0, 255)             # No mask detected