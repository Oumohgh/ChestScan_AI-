# ChestScan AI — Classification de radiographies thoraciques (NORMAL / PNEUMONIA)

Projet de fin d'année : classification binaire d'images radiographiques thoraciques
issues du dataset Kaggle **Chest X-Ray Images (Pneumonia)**.

> **Avertissement** — Prototype pédagogique. Cet outil n'est **pas** un dispositif
> médical et ne doit pas être utilisé pour un diagnostic.

---

## 1. Objectif

Construire, entraîner et comparer plusieurs architectures de classification d'images
sur le même jeu de données, puis livrer le meilleur modèle accompanied d'une
application Streamlit de démonstration.

| Modèle | Résolution | Description |
|---|---|---|
| **MobileNetV2 — Transfer Learning** | 224×224 | Backbone ImageNet gelé + tête `GlobalAveragePooling2D → Dropout → Dense(1)` |
| **MobileNetV2 — Fine-tuning** | 224×224 | Les 20 derniers blocs du backbone sont entraînés, BatchNorm **gelés**, LR `1e-5` |
| **CNN from scratch** | 180×180 | `Conv2D(32) → Conv2D(64) → GlobalAveragePooling2D → Dense(64) → Dense(1)` |

---

## 2. Données

* Source : `kagglehub` → `paultimothymooney/chest-xray-pneumonia` (`chest_xray/chest_xray/`).

| Dossier | NORMAL | PNEUMONIA | Total | Usage |
|---|---|---|---|---|
| `train/` | 1 341 | 3 875 | 5 216 | 85 % entraînement / 15 % validation |
| `test/` | 234 | 390 | 624 | **une seule fois**, pour la comparaison finale |
| `val/` | 8 | 8 | 16 | trop petit → non utilisé |

* **Encodage des labels** : `NORMAL = 0`, `PNEUMONIA = 1` (`class_names` explicite).
* **Validation** : `validation_split=0.15`, `shuffle=True`, `seed=42` → découpage
  **identique et reproductible** pour les trois modèles.
* **Le dossier `test/` n'est jamais utilisé pour l'entraînement ni pour le choix du
  modèle** (early stopping, hyper-paramètres) : uniquement pour la mesure finale.
* Les identifiants de patients de `train/` et `test/` sont disjoints.

---

## 3. Exécution du notebook

Le notebook **`Breif2.ipynb`** est autonome et s'exécute de bout en bout.

### Sur Google Colab (recommandé)

1. Ouvrir `Breif2.ipynb` dans Colab, puis **Runtime → Run all**.
2. Les cellules 0 à 7 téléchargent le dataset et construisent les `tf.data.Dataset`.
3. Les cellules 22, 25 et 33 entraînent les trois modèles (GPU V100 : ~10 min).
4. Les dernières cellules produisent les métriques, le tableau comparatif et les livrables.

### En local

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Breif2.ipynb
```

> Le notebook télécharge le dataset via `kagglehub` (connexion Internet requise la
> première fois).

---

## 4. Ce que produit le notebook

| Fichier | Contenu |
|---|---|
| `chestscan_model.keras` | le modèle retenu (copié depuis `cnn_scratch.keras`, `mobilenetv2_transfer.keras` ou `mobilenetv2_finetuned.keras`) |
| `config.json` | taille d'entrée, classes, seuil, prétraitement attendu par l'application |
| `resultats_comparaison.csv` | tableau comparatif des trois modèles |
| `mobilenetv2_transfer.keras`, `mobilenetv2_finetuned.keras`, `cnn_scratch.keras` | les trois modèles entraînés |

**Règle de sélection du modèle final** : on maximise d'abord le **recall sur
PNEUMONIA** (rater le moins de pneumonies possible), puis on départage à égalité par
le **ROC-AUC**, puis par le **F1**. Les métriques et les matrices de confusion des
trois modèles sont produites par une **fonction d'évaluation unique**
(`evaluate_model`), donc directement comparables.

---

## 5. Application Streamlit

```bash
streamlit run app.py
```

L'application lit `chestscan_model.keras` + `config.json`, applique **exactement** le
prétraitement utilisé à l'entraînement (déclaré dans `config.json` :

* `mobilenetv2.preprocess_input` → `x / 127.5 − 1` (MobileNetV2 **n'a pas** de
  `Rescaling` interne en Keras 3) ;
* `none` → la couche `Rescaling(1/255)` est **dans** le modèle (CNN from scratch).

---

## 6. Structure du dépôt

```
.
├── Breif2.ipynb          # notebook principal : données → 3 modèles → comparaison → livrables
├── Breif1.ipynb          # notebook de référence (CNN « from scratch »)
├── app.py                # application Streamlit
├── config.json           # écrit par le notebook
├── requirements.txt
└── data/chest_xray/      # copie locale du dataset (train / val / test)
```

---

## 7. Correctifs appliqués par rapport à la version initiale

| Problème | Correction |
|---|---|
| `preprocess_input` défini mais **jamais appliqué** (`.map()` manquant) → MobileNetV2 entraîné sur des pixels en 0‑255 | `.map(prep)` sur `train_ds` / `val_ds` / `test_ds` |
| `plt.plot(h.h["loss"])` → `AttributeError` (`History` n'a pas d'attribut `.h`) | `h.history["loss"]` |
| Cellule vide conservant un affichage périmé | source restaurée (`train_ds`), sortie purgée |
| Entraînement et métriques sans chronométrage | `time.time()` + `training_duration` |
| Fine-tuning **sans** `EarlyStopping` et avec un `EarlyStopping` réutilisé | nouvel objet `early_stop_ft`, `restore_best_weights=True` |
| Fine-tuning : BatchNorm **non gelés** (dérive des statistiques en Keras 3) | gel explicite des 7 `BatchNormalization` |
| Fine-tuning compilé avec `metrics=["accuracy"]` seulement | `metrics=make_metrics()` (mêmes métriques que les autres modèles) |
| ROC-AUC calculé sur les prédictions 0/1 dans le notebook de référence | calcul sur les **probabilités** |
| Entraînement et prédiction pour un seul modèle | trois modèles entraînés + fonction d'évaluation commune + tableau comparatif |
| Aucun livrable (modèle, configuration, application cohérente) | `.keras` + `config.json` + `predict_xray()` + `app.py` aligné |
| Application codée en dur pour un CNN 180×180 avec `Rescaling` interne | lecture de `config.json` (taille **et** prétraitement) |

---

## 8. Limites

* Le jeu de données est déséquilibré (≈ 61 % PNEUMONIA) : les `class_weights` sont
  appliqués, mais l'**accuracy** reste un indicateur trompeur — le **recall
  PNEUMONIA** et le **nombre de faux négatifs** sont les métriques de référence.
* Les données proviennent d'un dataset public unique : pas de validation externe.
* Aucun test clinique, aucune calibration de probabilité, aucune validation par
  radiologue.
