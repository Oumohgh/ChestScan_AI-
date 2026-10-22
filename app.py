# ChestScan AI - application Streamlit
# Prototype pedagogique : ceci n'est PAS un outil de diagnostic medical.
# Le modele est charge UNE SEULE FOIS grace a @st.cache_resource.
#
# Le modele et sa configuration sont ecrits par le notebook Breif2.ipynb :
#   - chestscan_model.keras  : le modele retenu
#   - config.json            : taille d'entree, classes, seuil, pretraitement
# Le pretraitement est lu dans config.json -> l'application suit toujours
# exactement le pipeline utilise a l'entrainement.
import json
import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

MODELE = "chestscan_model.keras"
CONFIG = "config.json"

# Interpolation identique a celle de l'entrainement (bilinear).
RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR


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

taille = tuple(config["image_size"])
classes = config["classes"]                       # ["NORMAL", "PNEUMONIA"]
seuil = float(config.get("threshold", 0.5))
pretraitement = config.get("preprocess", "none")  # "mobilenetv2.preprocess_input" ou "none"

st.caption(
    f"Modele : {config.get('model_name', MODELE)} | entree {taille[0]}x{taille[1]} | "
    f"pretraitement : {pretraitement}"
)

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
    # Meme preparation que l'entrainement : RGB + redimensionnement bilinear.
    pixels = np.array(image.resize(taille, RESAMPLE), dtype=np.float32)   # valeurs dans [0, 255]
    if pretraitement == "mobilenetv2.preprocess_input":
        # MobileNetV2 (Keras 3) n'a PAS de Rescaling interne : c'est l'application
        # qui doit appliquer preprocess_input, exactement comme a l'entrainement.
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        pixels = preprocess_input(pixels)                                  # [0, 255] -> [-1, 1]
    # sinon : la couche Rescaling(1/255) est DANS le modele (CNN from scratch).
    score = float(modele.predict(np.expand_dims(pixels, axis=0), verbose=0)[0][0])
    classe = classes[1] if score >= seuil else classes[0]

    st.subheader(f"Classe predite : {classe}")
    st.metric("Score PNEUMONIA", f"{score:.1%}")
    st.progress(min(score, 1.0))
    st.caption(f"Seuil utilise : {seuil:.2f} | score >= seuil => {classes[1]}, sinon {classes[0]}.")
