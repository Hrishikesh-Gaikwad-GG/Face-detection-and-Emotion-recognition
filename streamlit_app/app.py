import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Face Emotion Recognition", layout="centered")
st.title("🎭Face Emotion Recognition🎭")
st.write("YOLO Face Detection + ResNet Emotion Classification")

device = 'cpu'

# ---------------------------
# Load Models (Cached)
# ---------------------------
@st.cache_resource
def load_models():
    # Load YOLO face detector
    face_detector = YOLO("models/yolo_face_best.pt")

    # Load emotion model
    emotion_net = models.resnet18(weights=None)
    emotion_net.fc = nn.Linear(emotion_net.fc.in_features, 7)
    emotion_net.load_state_dict(
        torch.load("models/resnet18_emotion_best.pth", map_location=device)
    )
    emotion_net.eval().to(device)
    face_detector.to(device)
    return face_detector, emotion_net

face_detector, emotion_net = load_models()

# Emotion labels
emotion_classes = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# ---------------------------
# Preprocessing
# ---------------------------
emotion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# Inference Function
# ---------------------------
def infer(image):
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    results = face_detector(img_rgb, conf=0.4)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = img_rgb[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_pil = Image.fromarray(face)
        face_tensor = emotion_transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = emotion_net(face_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs.max().item()

        label = f"{emotion_classes[pred]} ({conf:.2f})"

        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            img_bgr, label, (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ---------------------------
# UI
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width = 'stretch')

    if st.button("Detect Faces & Emotions"):
        output = infer(image)
        st.image(output, caption="Prediction", width = 'stretch')
