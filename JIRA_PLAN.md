# Plan Jira - ChestScan AI

Livrable demande : Epics et tickets, prets a recopier dans Jira.

Sujet : CNN (TensorFlow/Keras) qui classe des radiographies en NORMAL / PNEUMONIA, plus un prototype Streamlit.
Travail individuel, 5 jours. Statuts : `A faire` / `En cours` / `Termine`.

---

## Epic 1 : Donnees

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-1 | Recuperer et explorer le dataset | Installer kagglehub, telecharger `paultimothymooney/chest-xray-pneumonia`, lister les dossiers train/val/test, compter les images par classe et tracer la distribution. | Termine |
| CHEST-2 | Charger les images avec PIL | Boucle for + PIL : RGB, 180x180, tableaux NumPy uint8, try/except pour exclure et compter les images corrompues, labels depuis les noms de dossiers. | Termine |
| CHEST-3 | Split train / validation / test sans fuite | Deux appels a train_test_split avec stratify : d abord le test independant (20 %), ensuite la validation. Seed fixee a 42. | Termine |
| CHEST-4 | Traiter le desequilibre des classes | Graphique de distribution, constat du ratio (~2,9 PNEUMONIA pour 1 NORMAL), calcul des poids de classes avec la formule total / (2 x nombre). | Termine |

## Epic 2 : Modele

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-5 | Construire le CNN simple | Sequential : Rescaling + RandomFlip/Rotation/Zoom dans le modele, 3 blocs Conv2D + MaxPooling2D, Flatten, Dense(128), Dropout, sortie 1 neurone sigmoide. | Termine |
| CHEST-6 | Entrainer avec callbacks | model.fit avec class_weight, EarlyStopping + ReduceLROnPlateau, duree mesuree, courbes loss et accuracy train vs validation. | Termine |

## Epic 3 : Evaluation

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-7 | Evaluer sur le test independant | Accuracy, precision, recall, F1, ROC-AUC, classification_report et matrice de confusion. | Termine |
| CHEST-8 | Analyser les erreurs | Compter et afficher les faux positifs et faux negatifs, 6 predictions correctes et 6 incorrectes avec leurs scores, comparaison train / validation / test. | Termine |
| CHEST-9 | Etudier le seuil de decision | Courbe ROC, metriques pour les seuils 0.3 a 0.7, justification du seuil (rappel : un faux negatif est plus grave qu un faux positif). | Termine |

## Epic 4 : Deploiement

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-10 | Sauvegarder, recharger, predire | chestscan_model.keras + config.json, rechargement sans reentrainer avec verification des memes predictions, fonction predict_xray testee sur plusieurs images. | Termine |
| CHEST-11 | Application Streamlit | Upload JPEG/JPG/PNG/BMP, affichage, bouton, classe + score PNEUMONIA, try/except sur fichier invalide, avertissement pedagogique, modele charge une seule fois. | Termine |

## Epic 5 : Documentation

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-12 | README et depot GitHub propre | README complet (description, source des donnees, pretraitement, architecture, resultats reels, structure, installation, lancement, limites), requirements.txt, .gitignore sans le dataset. | Termine |
