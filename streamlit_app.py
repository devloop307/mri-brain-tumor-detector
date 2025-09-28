# streamlit_app.py

import os, json, io
from pathlib import Path
import numpy as np
from PIL import Image

import streamlit as st
import torch, torch.nn as nn
from torchvision import models, transforms
import cv2

# -------------------- Config página --------------------
st.set_page_config(
    page_title="Detector de tumor cerebral (MRI)",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("<h1 style='text-align:center;color:#4B8BBE;'>🧠 Detección de tumor cerebral (MRI)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Sube una imagen (axial/coronal/sagital) y presiona <b>Predecir</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- Rutas y umbral --------------------
ROOT = Path(__file__).parent
CKPT = ROOT / "resnet18_best.pt"
TH_JSON = ROOT / "best_threshold.json"

threshold = 0.5
if TH_JSON.exists():
    try:
        info = json.load(open(TH_JSON))
        threshold = float(info["thresholds"]["recall_priority"]["threshold"])
    except Exception:
        pass

# -------------------- Modelo y transforms --------------------
@st.cache_resource
def load_model_and_transforms():
    # modelo
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    state = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(state)
    m.eval()

    # transforms como en entrenamiento
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return m, tfm

model, tfm = load_model_and_transforms()

# -------------------- Utilidades Grad-CAM --------------------
def gradcam_yes_overlay(model, x_tensor, orig_pil):
    """
    Devuelve imagen RGB con overlay Grad-CAM y caja (para clase YES).
    """
    target_layer = model.layer4[-1].conv2
    feats = {}

    def fwd_hook(m, inp, out): feats['a'] = out.detach()
    def bwd_hook(m, gin, gout): feats['g'] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    x_req = x_tensor.clone().requires_grad_(True)
    logits = model(x_req)
    logits[0, 1].backward(retain_graph=True)

    h1.remove(); h2.remove()
    A = feats['a'][0]             # CxHxW
    G = feats['g'][0]             # CxHxW
    w = G.mean(dim=(1,2), keepdim=True)
    cam = (w * A).sum(0).clamp(min=0).cpu().numpy()
    cam = cam / (cam.max() + 1e-8)

    w0, h0 = orig_pil.size
    heat = cv2.resize(cam, (w0, h0))
    hm_u8 = (heat * 255).astype(np.uint8)

    rgb = np.array(orig_pil.convert("RGB"))
    overlay = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = (0.35*overlay + 0.65*rgb[...,::-1]).astype(np.uint8)

    # Caja por Otsu
    _, th = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x1,y1,wc,hc = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(overlay, (x1,y1), (x1+wc, y1+hc), (255,255,255), 2)

    # BGR->RGB
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

# -------------------- Subir imagen --------------------
file = st.file_uploader("Sube una imagen (png/jpg/jpeg/tif)", type=["png","jpg","jpeg","tif","tiff"])

if file:
    orig = Image.open(file).convert("L")
    # Streamlit 1.35 usa use_column_width (no existe use_container_width)
    st.image(orig, caption="Imagen cargada", use_column_width=True)

    if st.button("🔍 Predecir", use_container_width=True):
        # Preprocesar y predecir
        x = tfm(orig).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        p_no, p_yes = float(probs[0]), float(probs[1])
        pred_cls = "yes" if p_yes >= threshold else "no"

        st.markdown(f"**Umbral:** `{threshold:.2f}`  |  **prob_yes:** `{p_yes:.3f}`")
        if pred_cls == "yes":
            st.error(f"🧠 **Pred: YES** — posible anomalía (prob_yes={p_yes:.3f})")
            # Grad-CAM
            overlay = gradcam_yes_overlay(model, x, orig)
            st.image(overlay, caption="Regiones de atención (Grad-CAM)", use_column_width=True)
        else:
            st.success(f"✅ **Pred: NO** — sin anomalía (prob_yes={p_yes:.3f})")

        st.caption("Nota: Resultado orientativo; no sustituye criterio médico.")
