# ChestScan AI - application Streamlit (version simple)
# Prototype pedagogique : ceci n'est PAS un outil de diagnostic medical.
# Le modele est charge UNE SEULE FOIS grace a @st.cache_resource.
import json
import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

MODELE = "chestscan_model.keras"
CONFIG = "config.json"


@st.cache_resource
def charger_modele():
    """Charge le modele et la config une seule fois, puis le cache s occupe du reste."""
    if not os.path.exists(MODELE) or not os.path.exists(CONFIG):
        return None, None
    with open(CONFIG, encoding="utf-8") as f:
        return load_model(MODELE), json.load(f)


st.set_page_config(page_title="ChestScan AI", layout="centered")
st.title("ChestScan AI")
st.caption("Radiographie thoracique : NORMAL ou PNEUMONIA")
st.warning("Prototype pedagogique, non destine au diagnostic medical.")

modele, config = charger_modele()
if modele is None:
    st.error("Modele introuvable : lancez d abord le notebook, qui ecrit chestscan_model.keras et config.json.")
    st.stop()

fichier = st.file_uploader("Chargez une radiographie", type=["jpeg", "jpg", "png", "bmp"])
if fichier is None:
    st.info("En attente d une image.")
    st.stop()

try:
    image = Image.open(fichier).convert("RGB")  # un fichier casse leve une erreur ici
except (UnidentifiedImageError, OSError, ValueError):
    st.error("Fichier illisible ou format non supporte (JPEG, JPG, PNG ou BMP).")
    st.stop()

st.image(image, caption="Image chargee", use_container_width=True)

if st.button("Lancer la prediction", type="primary"):
    # Meme preparation que l entrainement : RGB + 180x180, PAS de division par 255
    # car la couche Rescaling est deja DANS le modele
    taille = tuple(config["image_size"])
    pixels = np.array(image.resize(taille), dtype=np.uint8)
    score = float(modele.predict(np.expand_dims(pixels, axis=0), verbose=0)[0][0])
    seuil = config.get("threshold", 0.5)
    classe = config["classes"][1] if score >= seuil else config["classes"][0]

    st.subheader(f"Classe predite : {classe}")
    st.metric("Score PNEUMONIA", f"{score:.1%}")
    st.progress(min(score, 1.0))
    st.caption(f"Seuil utilise : {seuil:.2f} | score >= seuil => PNEUMONIA, sinon NORMAL.")
