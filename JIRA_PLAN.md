# Plan Jira - ChestScan AI

Livrable demande : Epics et tickets, prets a recopier dans Jira.

Sujet : CNN (TensorFlow/Keras) pour classer des radiographies thoraciques en NORMAL / PNEUMONIA,
puis prototype Streamlit. Travail individuel, 5 jours.

Statuts utilises : `A faire` / `En cours` / `Termine`.

---

## Epic 1 : Donnees

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-1 | Telecharger et explorer le dataset | Installer kagglehub, telecharger `paultimothymooney/chest-xray-pneumonia`, lister les dossiers train/val/test et compter les images par classe. | Termine |
| CHEST-2 | Filtrer les images corrompues | Verifier chaque fichier avec PIL, lister et exclure les images illisibles, compter les exclusions. | Termine |
| CHEST-3 | Split train / validation / test sans fuite | Garder le test Kaggle comme test independant, fusionner train+val (val ne contient que 16 images), split stratifie 80/20 pour la validation. Seed fixee. | Termine |
| CHEST-4 | Pretraitement et gestion du desequilibre | Redimension 180x180, conversion RGB, normalisation [0,1] appliquee une seule fois, graphique de distribution, class_weight pour le desequilibre. | Termine |

## Epic 2 : Modele

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-5 | Construire le CNN | Sequence Conv2D + MaxPooling2D + Flatten + Dense + Dropout, sortie 1 neurone sigmoide, compile Adam + binary_crossentropy. | Termine |
| CHEST-6 | Entrainement avec callbacks | model.fit(), augmentation sur le train seulement, EarlyStopping + ReduceLROnPlateau, duree mesuree. | Termine |
| CHEST-7 | Analyse des courbes d'entrainement | Courbes loss et accuracy train/validation, detection du surapprentissage / sous-apprentissage, commentaires dans le notebook. | Termine |

## Epic 3 : Evaluation

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-8 | Evaluer sur le test independant | Accuracy, precision, recall, F1, ROC-AUC, matrice de confusion, classification_report. | Termine |
| CHEST-9 | Analyse des erreurs | Compter et afficher les faux positifs et faux negatifs, montrer des exemples corrects et incorrects, expliquer pourquoi un faux negatif est grave en medical. | Termine |
| CHEST-10 | Etude du seuil de decision | Comparer les metriques pour plusieurs seuils (0.3 / 0.5 / 0.7) et justifier le seuil choisi. | Termine |

## Epic 4 : Deploiement

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-11 | Sauvegarde et rechargement | Sauvegarder le modele en .keras + config.json (taille, classes, seuil), recharger sans reentrainer et verifier les memes predictions. | Termine |
| CHEST-12 | Application Streamlit | Upload image (JPEG/JPG/PNG/BMP), affichage, bouton de prediction, classe + score PNEUMONIA, gestion des erreurs, avertissement pedagogique. | Termine |

## Epic 5 : Documentation

| Cle | Titre | Description courte | Statut |
|-----|-------|--------------------|--------|
| CHEST-13 | README complet | Description, source du dataset, recuperation des donnees, pretraitement, architecture, resultats reels, structure, installation, lancement, limites. | Termine |
| CHEST-14 | Depot GitHub propre | .gitignore sans le dataset, requirements.txt, code commente en francais, arborescence claire. | Termine |
