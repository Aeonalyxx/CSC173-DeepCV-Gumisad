import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
ONNX_MODEL_PATH = MODEL_DIR / "mobilenetv2.onnx"
PTH_MODEL_PATH = MODEL_DIR / "mobilenetv2_best.pth"

session = ort.InferenceSession(str(ONNX_MODEL_PATH))

input_name = session.get_inputs()[0].name

def preprocess_image(image):

    image = np.array(image)

    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image

device = torch.device("cpu")

cam_model = models.mobilenet_v2(pretrained=False)

cam_model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(cam_model.last_channel, 2)
)

cam_model.load_state_dict(
    torch.load(PTH_MODEL_PATH, map_location=device)
)

cam_model.eval()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):

        output = self.model(x)

        self.model.zero_grad()

        score = output[0, class_idx]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        cam = cam.detach().numpy()
        cam = np.maximum(cam, 0)

        cam = cam / (cam.max() + 1e-8)

        return cam
    

def overlay_heatmap(image, cam):

    image = np.array(image.resize((224, 224)))

    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + image * 0.6

    return overlay.astype(np.uint8)

from torchvision import transforms

def resize_for_display(image, size=(400, 400)):
    return image.resize(size)

cam_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

target_layer = cam_model.features[-1]


def predict_with_cam(image):

    # ONNX prediction
    input_tensor = preprocess_image(image)

    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]

    probs = np.exp(logits) / np.sum(np.exp(logits))
    class_idx = np.argmax(probs)

    classes = ["Clear Skin", "Skin Imperfection"]

    label = classes[class_idx]
    confidence = float(probs[class_idx])

    # Grad-CAM
    tensor_img = cam_transform(image).unsqueeze(0)

    cam_generator = GradCAM(cam_model, target_layer)
    cam = cam_generator.generate(tensor_img, class_idx)

    heatmap = overlay_heatmap(image, cam)

    return label, confidence, heatmap

st.set_page_config(page_title="Skin Detection", layout="centered")

st.title("👨‍🦲 Skin Imperfection Detection System")
st.write("Upload a skin image to detect imperfections using MobileNetV2 + Grad-CAM.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    label, confidence, heatmap = predict_with_cam(image)

    st.markdown("## 🤖 Grad-CAM Visualization")

    display_size = (400, 400)

    image_disp = resize_for_display(image, display_size)
    heatmap_disp = resize_for_display(Image.fromarray(heatmap), display_size)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Image")
        st.image(image_disp)

    with col2:
        st.markdown("### Heatmap")
        st.image(heatmap_disp)
    
    st.markdown("## Prediction Result")

    if label == "Clear Skin":
        st.success(f"{label} ({confidence*100:.2f}%)")
    else:
        st.error(f"{label} ({confidence*100:.2f}%)")

def show_heatmap_legend():
    
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, ax = plt.subplots(figsize=(6, 1))

    ax.imshow(gradient, aspect='auto', cmap='jet')

    ax.set_title("Grad-CAM Heatmap Guide", fontsize=12)

    ax.set_yticks([])
    ax.set_xticks([
        0, 64, 128, 192, 255
    ])

    ax.set_xticklabels([
        "Low", "Low-Mid", "Mid", "High", "Very High"
    ])

    return fig

with st.expander("🎨 What does the heatmap mean?"):
    st.pyplot(show_heatmap_legend())

    st.markdown("""
    - 🔵 Blue → Model ignores this region  
    - 🟢 Green → Low attention  
    - 🟡 Yellow → Medium attention  
    - 🔴 Red → High attention (possible imperfection area)
    """)