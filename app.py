# ChestScan AI - application Streamlit
# Prototype pedagogique : ceci n'est PAS un outil de diagnostic medical.
# L'application recharge un modele deja entraine (chestscan_model.keras).
# Elle ne reentraine JAMAIS le modele : c'est la regle du deploiement.

import json
import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Chemins relatifs au dossier du projet : pas de chemin en dur
MODEL_PATH = "chestscan_model.keras"
CONFIG_PATH = "config.json"

# Formats acceptes par le sujet (JPEG, JPG, PNG, BMP)
ACCEPTED_TYPES = ["jpeg", "jpg", "png", "bmp"]


# On charge le modele UNE SEULE FOIS.
# Sans cache, chaque clic sur un bouton rechargerait le modele (lent pour rien).
@st.cache_resource
def load_pipeline():
    """Retourne (modele, config) ou (None, None) si un fichier manque."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        return None, None
    modele = load_model(MODEL_PATH)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    return modele, config


def preprocess_image(image, config):
    """Prepare l'image COMME a l'entrainement : RGB, 180x180, valeurs [0,1].

    Si le pretraitement differe de l'entrainement, les predictions sont fausses.
    C'est la cause d'erreur n 1 des deploiements.
    """
    taille = tuple(config["image_size"])
    # convert("RGB") : une radiographie peut etre lue en 1 canal, le modele attend 3
    image = image.convert("RGB").resize(taille)
    tableau = img_to_array(image) / 255.0
    # axis=0 : le modele attend un lot (batch) meme pour UNE seule image
    return np.expand_dims(tableau, axis=0)


def predict_xray(image, modele, config):
    """Retourne (classe predite, score PNEUMONIA, seuil utilise)."""
    entree = preprocess_image(image, config)
    score = float(modele.predict(entree, verbose=0)[0][0])
    seuil = float(config.get("threshold", 0.5))
    # La classe 1 est PNEUMONIA (voir config["classes"])
    classe = config["classes"][1] if score >= seuil else config["classes"][0]
    return classe, score, seuil


# ---------------------------------------------------------------- interface
st.set_page_config(page_title="ChestScan AI", page_icon="🩻", layout="centered")

st.title("ChestScan AI")
st.caption("Classification de radiographies thoraciques : NORMAL ou PNEUMONIA")

# Avertissement obligatoire : outil pedagogique, pas medical
st.warning("Prototype pedagogique, non destine au diagnostic medical.")

modele, config = load_pipeline()

if modele is None:
    # Cas courant : le notebook n'a pas encore ete execute, donc pas de .keras
    st.error(
        "Modele introuvable. Lancez d'abord le notebook (entrainement + sauvegarde) "
        "pour generer chestscan_model.keras et config.json."
    )
    st.stop()

# Panneau lateral : on montre la configuration chargee (transparence)
with st.sidebar:
    st.header("Configuration du modele")
    st.write("Taille d'image :", tuple(config["image_size"]))
    st.write("Classes :", config["classes"])
    st.write("Seuil de decision :", config.get("threshold", 0.5))
    st.caption("Source des donnees : " + config.get("dataset", "inconnue"))

fichier = st.file_uploader(
    "Chargez une radiographie", type=ACCEPTED_TYPES, help="Formats : JPEG, JPG, PNG, BMP"
)

if fichier is None:
    st.info("En attente d'une image.")
    st.stop()

# On ouvre l'image DANS UN TRY : un fichier corrompu ou un mauvais format
# ne doit pas faire planter l'application.
try:
    image = Image.open(fichier)
    image.load()  # force la lecture reelle du fichier (leve une erreur si casse)
except (UnidentifiedImageError, OSError, ValueError):
    st.error("Fichier illisible ou format non supporte. Essayez un fichier JPEG, JPG, PNG ou BMP.")
    st.stop()

col_image, col_result = st.columns(2)

with col_image:
    st.image(image, caption="Image chargee", use_container_width=True)

with col_result:
    if st.button("Lancer la prediction", type="primary", use_container_width=True):
        try:
            classe, score, seuil = predict_xray(image, modele, config)
        except Exception as erreur:
            # Derniere protection : on affiche l'erreur au lieu de planter
            st.error(f"La prediction a echoue : {erreur}")
            st.stop()

        couleur = "red" if classe == "PNEUMONIA" else "green"
        st.subheader(f":{couleur}[{classe}]")
        st.metric("Score PNEUMONIA", f"{score:.1%}")
        st.progress(min(score, 1.0))
        st.caption(
            f"Seuil = {seuil:.2f} : score >= seuil => PNEUMONIA, sinon NORMAL. "
            "Le score est une probabilite, pas une certitude."
        )

st.divider()
st.caption("Projet ChestScan AI - CNN TensorFlow/Keras - usage pedagogique uniquement.")
