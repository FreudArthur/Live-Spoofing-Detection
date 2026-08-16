from pathlib import Path

import streamlit as st
from PIL import Image

from spoof_model import load_checkpoint, predict_pil_image

st.set_page_config(page_title="Live Spoofing Detection", page_icon="🛡️")
st.title("Live Spoofing Detection")
st.write("Chargez une image de visage pour prédire **spoof** ou **non-spoof**.")

model_path = st.text_input("Checkpoint du modèle", value="checkpoints/best.pt")
device = st.selectbox("Device", options=["cpu", "cuda"], index=0)


@st.cache_resource
def load_model(path: str, selected_device: str):
    return load_checkpoint(path, device=selected_device)


uploaded_file = st.file_uploader("Image visage", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Ajoutez une image pour lancer une prédiction.")
elif not Path(model_path).exists():
    st.error(f"Checkpoint introuvable: {model_path}")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    model, class_names = load_model(model_path, device)
    label, score, class_scores = predict_pil_image(model, image, class_names, device=device)

    st.success(f"Prédiction: **{label}** ({score:.2%})")
    st.write("Détail des probabilités")
    st.json({k: round(v, 4) for k, v in class_scores.items()})
