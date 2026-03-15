from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import os, cv2, json, yaml, sys
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision import transforms

# ---------------- FLASK ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- GLOBAL STATS ----------------
total_videos = 0
fake_count = 0
real_count = 0

# ---------------- LOAD CLASSES ----------------
with open("model/classes.json") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k.upper() for k, v in class_to_idx.items()}

print("Loaded classes:", class_to_idx)

# ---------------- LOAD CROSS EFFICIENT VIT ----------------
sys.path.insert(0, "cross-efficient-vit")
from cross_efficient_vit import CrossEfficientViT

# Load architecture config
with open("configs/architecture.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize model
model = CrossEfficientViT(config=config)

# Load trained weights
model.load_state_dict(
    torch.load("model/best_cross_vit.pth", map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

print("Cross Efficient ViT model loaded successfully")

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, max_frames=30):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while cap.isOpened() and len(frames) < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])

        face = frame[y:y+h, x:x+w]

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = transform(Image.fromarray(face))

        frames.append(face)

    cap.release()

    return frames


# ---------------- VIDEO PREDICTION ----------------
def predict_video(frames):

    probs = []

    with torch.no_grad():

        for f in frames:

            f = f.unsqueeze(0).to(DEVICE)

            output = model(f)

            prob = torch.sigmoid(output).item()

            probs.append(prob)

    if len(probs) == 0:
        return "UNCERTAIN", 0.0

    avg_prob = np.mean(probs)

    fake_index = class_to_idx["fake"]
    real_index = class_to_idx["real"]

    # Correct mapping
    if fake_index == 1:
        fake_prob = avg_prob
        real_prob = 1 - avg_prob
    else:
        real_prob = avg_prob
        fake_prob = 1 - avg_prob

    if real_prob >= 0.65:
        return "REAL", round(real_prob * 100, 2)

    elif fake_prob >= 0.60:
        return "FAKE", round(fake_prob * 100, 2)

    else:
        return "UNCERTAIN", round(max(real_prob, fake_prob) * 100, 2)


# ---------------- HOME PAGE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    global total_videos, fake_count, real_count

    result = None
    confidence = None

    if request.method == "POST":

        video = request.files.get("video")

        if video and video.filename:

            filename = secure_filename(video.filename)

            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            video.save(path)

            frames = extract_frames(path)

            result, confidence = predict_video(frames)

            # UPDATE STATS
            total_videos += 1

            if result == "FAKE":
                fake_count += 1

            elif result == "REAL":
                real_count += 1

            os.remove(path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    accuracy = 98.2  # replace with your real test accuracy

    return render_template(
        "dashboard.html",
        total_videos=total_videos,
        fake_count=fake_count,
        real_count=real_count,
        accuracy=accuracy
    )


# ---------------- DETECTION PAGE ----------------
@app.route("/detect", methods=["GET", "POST"])
def detect():

    global total_videos, fake_count, real_count

    result = None
    confidence = None

    if request.method == "POST":

        video = request.files.get("video")

        if video and video.filename:

            filename = secure_filename(video.filename)

            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            video.save(path)

            frames = extract_frames(path)

            result, confidence = predict_video(frames)

            # UPDATE STATS
            total_videos += 1

            if result == "FAKE":
                fake_count += 1

            elif result == "REAL":
                real_count += 1

            os.remove(path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
