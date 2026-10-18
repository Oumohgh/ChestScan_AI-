# Preparation a l'oral - ChestScan AI

## 1. Resume des choix techniques (a dire en 10 lignes)

1. **Dataset** : Kaggle `chest-xray-pneumonia`, 5856 radiographies (1341 NORMAL / 3875 PNEUMONIA en train), enfants de 1 a 5 ans, un seul hopital.
2. **Test independant** : le dossier `test` du dataset (624 images) n'est jamais touche avant l'evaluation finale. Le dossier `val` (16 images) est inutilisable : fusionne au train, puis split stratifie 80/20 pour la validation.
3. **Pretraitement unique** : RGB, 180x180, division par 255 **une seule fois**, applique partout (notebook, `predict_xray`, `app.py`).
4. **Desequilibre** : `class_weight="balanced"` + augmentation sur le train seul + metriques par classe (recall/precision) plutot que l accuracy seule.
5. **Modele** : CNN 4 blocs Conv2D(32-64-128-128) + MaxPooling2D, Flatten, Dense(512, relu), Dropout(0.5), sortie 1 neurone **sigmoide** (binaire).
6. **Compile** : `adam` + `binary_crossentropy` + metriques accuracy/precision/recall/AUC.
7. **Entrainement** : epochs max 20, batch 32, **EarlyStopping** (patience 5, restore_best_weights) + **ReduceLROnPlateau** (patience 3), duree mesuree.
8. **Evaluation** : test independant uniquement - accuracy, precision, recall, F1, ROC-AUC, matrice de confusion, exemples d'erreurs, comparaison train/val/test.
9. **Arbitrage medical** : un faux negatif (malade rate) est plus grave qu'un faux positif => on regarde le recall et le seuil (0.5 par defaut, etude 0.3 a 0.7).
10. **Deploiement** : `chestscan_model.keras` + `config.json` (taille, classes, seuil), rechargement sans reentrainement, app Streamlit avec avertissement pedagogique.

## 2. Les 10 questions les plus probables

**Q1. Pourquoi 180x180 et RGB alors que ce sont des radios en niveaux de gris ?**
C'est la taille demandee par le sujet. RGB (3 canaux) permet d'utiliser un modele standard de vision par ordinateur ; le meme canal est simplement repete, le cout est faible et le pretraitement devient identique a n'importe quelle image.

**Q2. Quelle est la difference entre precision et recall, et laquelle compte le plus ici ?**
Precision = parmi les predits malades, combien le sont vraiment. Recall = parmi les vrais malades, combien ont ete detects. En depistage, le **recall** prime : un faux negatif laisse un malade sans traitement. On accepte donc plus de faux positifs.

**Q3. Qu'est-ce qu'une fuite de donnees (data leakage) et comment l'avez-vous evitee ?**
C'est quand une information du test "fuit" dans l'entrainement, ce qui fausse les resultats. Ici : split fait avant normalisation, test jamais utilise pour choisir un epoch ou un seuil, et augmentation appliquee au train seulement.

**Q4. Pourquoi le dossier `val` du dataset n'est-il pas utilise tel quel ?**
Il ne contient que 16 images : 8 par classe. Avec une val aussi petite, la courbe de validation est un bruit ininterpretable (on voyait val_accuracy osciller entre 50 % et 68 %). On l'a fusionne au train et on a fait un split stratifie 80/20 (~1000 images de validation).

**Q5. Qu'est-ce que le surapprentissage et comment le detectez-vous ?**
Quand le modele memorise le train au lieu d'apprendre des regles generales : la perte d'entrainement baisse alors que la perte de validation remonte. On le voit sur les courbes, et on le limite avec le Dropout, l'augmentation, l'EarlyStopping et le class_weight.

**Q6. Pourquoi une sigmoide et non un softmax ?**
Pour 2 classes opposees, un seul neurone + sigmoide donne directement la probabilite de PNEUMONIA, avec `binary_crossentropy`. Un softmax a 2 sortes ferait la meme chose mais avec plus de parametres.

**Q7. Le score 0.9 veut-il dire 90 % de chances d'etre malade ?**
Non. C'est la confiance du modele, pas une probabilite medicale calibree. Le score est utile pour ranger les images et pour choisir un seuil, pas pour affirmer un chiffre au patient.

**Q8. Pourquoi ne pas avoir utilise du transfer learning (VGG, EfficientNet) ?**
Parce que le sujet demande un CNN developpe de zero et explicable en 30 minutes. C'est la piste d'amelioration numero 1 : quelques lignes de plus, un gain de performance, mais un modele moins lisible a defendre.

**Q9. Que se passe-t-il si on change le seuil de 0.5 a 0.3 ?**
Plus de malades detectes (recall augmente), donc moins de faux negatifs, mais plus de faux positifs (fausses alarmes). Le tableau des seuils du notebook montre exactement ce compromis.

**Q10. Comment l'application reutilise-t-elle le modele sans le reentrainement ?**
Le modele est sauvegarde en `.keras`, la config (taille, ordre des classes, seuil) dans `config.json`. Streamlit charge les deux une seule fois (`@st.cache_resource`) et applique le meme pretraitement que l'entrainement. Aucun `fit()` dans `app.py`.

## 3. Trois scenarios de mise en situation (modification de code exacte)

### Scenario A - "Baissez le seuil pour rater moins de malades"
Objectif : passer le seuil a 0.3 et voir l'effet sur recall / faux negatifs.

Dans le notebook (cellule d'evaluation), remplacer :
```python
SEUIL = 0.5
```
par :
```python
SEUIL = 0.3  # on accepte plus de faux positifs pour rater moins de malades
```
Puis **relancer la cellule d'evaluation et la matrice de confusion** (pas besoin de reentrainer : le modele est deja sauvegarde). Dans `config.json`, changer aussi `"threshold": 0.3` pour que l'application utilise le nouveau seuil. A dire : "le modele n'a pas change, seule la decision a bouge - c'est un arbitrage medical".

### Scenario B - "Ajouter une couche de Dropout pour limiter le surapprentissage"
Dans le modele, ajouter un Dropout apres la couche dense :
```python
model = Sequential([
    ...
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(512, activation="relu"),   # <- nouvelle couche dense
    Dropout(0.3),                    # <- nouveau Dropout (30 % des neurones eteints)
    Dense(1, activation="sigmoid"),
])
```
Puis re-executer la construction, la compilation et l'entrainement (il faut reentrainer). A dire : "le Dropout force le reseau a ne pas dependre d'un seul neurone : c'est de la regularisation, ca reduit l ecart train/validation".

### Scenario C - "Expliquer un faux negatif concret"
Ouvrir la cellule d'exemples d'erreurs, section "Faux negatifs". Procedure orale :
1. montrer l'image et le score (ex. 0.42 : le modele hésite, il n'a pas "cru" a la pneumonie) ;
2. rappeler qu'avec le seuil a 0.5, 0.42 devient NORMAL : le malade est rate ;
3. proposer l'action concrete : baisser le seuil a 0.3 (voir scenario A) ou renvoyer tout score entre 0.3 et 0.5 vers un second examen (relecture par un radiologue) ;
4. conclure : c'est exactement pour ca qu'on regarde le recall et pas seulement l accuracy.

## 4. Les 5 concepts a maitriser absolument

1. **Fuite de donnees (data leakage)** : le test doit etre neuf. Dites les 3 regles appliquees : split avant normalisation, test non utilise pour decider, augmentation au train seul.
2. **Precision vs recall vs F1, et pourquoi le recall en medical** : pouvoir donner les vrais chiffres du test et compter les faux negatifs.
3. **Surapprentissage (overfitting)** : lire les courbes, nommer les armes utilisees (Dropout, augmentation, EarlyStopping, class_weight, reduction du learning rate).
4. **Pretraitement identique partout** : si l'application divise par 255 deux fois ou oublie la conversion RGB, le modele recoit des valeurs inconnues et les predictions deviennent du hasard silencieux.
5. **Seuil de decision** : la sigmoide sort une probabilite, la decision (NORMAL/PNEUMONIA) depend du seuil. On peut changer le seuil sans reentrainer, et c'est la que se joue l'equilibre faux positifs / faux negatifs.
