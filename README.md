# scrollorder — test 2D de deux idées pour le déroulage

Prototype pour mesurer, sur des coupes 2D d'un rouleau, si deux idées valent
la peine d'être portées en 3D :

- **H1 — surfaces ordonnées.** Entre deux spires connues séparées de K tours, il
  passe exactement K−1 feuilles, dans un ordre fixe. On les place toutes
  ensemble par coupe de graphe (méthode des couches de la rétine, Li, Wu, Chen
  et Sonka, 2006), avec une séparation minimale imposée. Les fusions et les
  croisements sont impossibles par construction.
- **H2 — décalage entier par code-barres.** À travers une zone illisible, les
  feuilles gardent leur ordre : la seule inconnue est un entier, le décalage.
  On l'estime avec le profil des espacements entre feuilles, comme on recale
  des cernes d'arbres en dendrochronologie.

Les deux idées sont un raisonnement, pas un résultat. Ce paquet sert à les
tester, avec un critère d'arrêt fixé d'avance.

## Ce que contient le paquet

| Fichier | Rôle |
|---|---|
| `scrollorder/ordered_cut.py` | Coupe de graphe : K−1 surfaces ordonnées, optimum global exact |
| `scrollorder/barcode.py` | Décalage entier entre deux piles de feuilles, avec indice de confiance |
| `scrollorder/polar.py` | Vue polaire autour de l'axe, échantillonnage entre deux spires, crêtes |
| `scrollorder/io_real.py` | Lecture des surfaces TIFXYZ et de l'ombilic (formats du dépôt `villa`) |
| `scrollorder/experiments.py` | Les deux mesures du protocole |
| `scrollorder/cli.py` | Commandes `stack`, `cut`, `barcode` |
| `scrollorder/phantom.py` | Coupe synthétique à réponse connue, **pour les tests uniquement** |
| `tests/` | 63 tests |

## Installation et vérification

```sh
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy tifffile matplotlib pytest
python -m pytest -q          # 63 tests, environ 15 s
```

Tout tourne sur CPU. Aucun GPU n'est nécessaire.

## État de la vérification

- Les 63 tests passent (Python 3.13, NumPy 2.5, SciPy 1.18).
- La coupe de graphe est comparée à une recherche exhaustive sur de petits cas :
  elle trouve le même coût optimal.
- Les lecteurs TIFXYZ et ombilic sont testés sur des fichiers synthétiques
  écrits selon la spécification (`lasagna/tifxyz_format.md`,
  `spiral-fitting/umbilicus.py`). **Ils n'ont jamais lu un vrai jeu de données.**
- L'extraction d'une coupe depuis un OME-Zarr (étape 2) n'est pas dans le
  paquet et n'a pas été testée.

Sur le fantôme (30 spires, bruit, trous, secteur comprimé ; 3 graines pour la
coupe, 2 pour le code-barres) :

| Mesure | Résultat |
|---|---|
| Coupe ordonnée, K = 2 à 20 | 99,5 à 99,9 % d'échantillons sur la bonne spire, 0 fusion |
| Division égale (référence) | 74 à 100 % selon la graine |
| Accrochage local (référence) | 71 à 99 %, jusqu'à 10 % de fusions |
| Code-barres, bande de 2° à 90° | 97,5 à 100 % de décalages corrects |
| Rayon le plus proche (référence) | 92 à 98 % à 2°, 20 % ou moins dès 15° |

Ces chiffres prouvent que le code fait ce qu'il doit sur un cas construit pour
lui. Ils ne disent rien des vrais rouleaux : le fantôme a des crêtes nettes et
un code-barres qui se conserve par construction.

## Protocole sur données réelles

### Étape 0 — Vérifier que ce n'est pas déjà fait

Avant d'y passer du temps, poser trois questions sur le Discord (un service de
recherche dans l'historique existe, primé en octobre 2026) :

1. Une segmentation multi-surfaces ordonnée par coupe de graphe a-t-elle été
   essayée entre spires connues ?
2. Le profil des espacements entre feuilles a-t-il servi à recaler des piles
   de part et d'autre d'une zone abîmée ? Le projet `winding-ruler` a un atlas
   du pas des spires sur 36 rouleaux, à regarder d'abord.
3. Quelle est la meilleure vérité terrain « une surface par spire » publiée ?

### Étape 1 — Choisir la vérité terrain

Le prototype attend **un dossier TIFXYZ par spire**. Deux sources possibles, à
confirmer dans le Data Browser :

- les surfaces reconstruites de PHerc. 1667 (31 spires vérifiées à la main) ;
- les sorties `wNNN` de l'ajusteur de spirale sur PHerc. Paris 4, à ne retenir
  que là où elles s'accordent avec les patchs vérifiés du jeu `spiral-input`.

Les patchs vérifiés seuls ne conviennent pas en l'état : il faudrait d'abord
les regrouper par numéro de spire (`abs_winding.json`). Ce n'est pas codé.

### Étape 2 — Extraire les coupes de probabilité de surface

Une coupe 2D de la prédiction de surface à une hauteur z, en `.npy` ou `.tif`.
Exemple non testé, chemins et niveau à adapter :

```python
import numpy as np, zarr
vol = zarr.open("…/surface-prediction.zarr/2", mode="r")   # axes (z, y, x)
np.save("prob_z5000.npy", np.asarray(vol[5000]))
```

Prendre 50 hauteurs espacées. En réserver 10 pour régler les paramètres et 40
pour mesurer. Ne jamais régler sur les 40.

### Étape 3 — Construire la pile de spires et la contrôler à l'œil

```sh
python -m scrollorder stack --tifxyz chemin/vers/les/spires \
    --z 5000 --umbilicus umbilicus.json --n-theta 1440 \
    --prob prob_z5000.npy --figure controle.png --out stack_z5000.npy
```

- `--z`, l'ombilic et les surfaces doivent être dans le repère de la coupe.
  Si la coupe est à un niveau de pyramide plus grossier, utiliser `--xyz-scale`
  et `--umbilicus-scale` (0,5 pour un niveau d'écart). D'après une correction
  récente du dépôt, le jeu `spiral-input` de Paris 4 est au niveau 2 (9,6 µm).
- Ouvrir `controle.png` : les courbes doivent suivre les crêtes, et la commande
  doit afficher 0 violation d'ordre. Sinon, ne pas aller plus loin.

### Étape 4 — Lancer les deux mesures

```sh
python -m scrollorder cut --prob prob_z5000.npy --stack stack_z5000.npy \
    --umbilicus umbilicus.json --z 5000 --k 2,5,10,20,30 --out cut_z5000.csv

python -m scrollorder barcode --stack stack_z5000.npy \
    --bands-deg 2,5,15,45,90 --trials 300 --out barcode_z5000.csv
```

`cut` cache les K−1 feuilles entre deux spires connues et compare trois
méthodes : division égale (`interp`), accrochage local (`snap`), coupe ordonnée
(`cut`). `barcode` cache une bande angulaire et un nombre inconnu de feuilles
aux bords, puis compare le rayon le plus proche (`nearest`) au code-barres.

Colonnes utiles : `accuracy` (part des échantillons sur la bonne spire),
`accuracy_dense` (idem sur les 10 % d'échantillons les plus serrés), `merged`
(feuilles voisines superposées), `confident_share` et `confident_accuracy`
(part des essais où le code-barres se dit sûr, et sa justesse dans ce cas).

### Étape 5 — Critères de décision, fixés avant de regarder les résultats

**H1 continue en 3D si**, sur les 40 coupes de mesure et pour K ≥ 10 :
`cut` dépasse `interp` d'au moins 10 points sur `accuracy_dense`, avec
`merged` à 0. Sinon, on arrête H1 et on passe au détecteur de sauts de feuille.

**H2 continue si**, pour une bande d'au moins 15° : `confident_accuracy` ≥ 99 %
avec `confident_share` ≥ 50 %. **H2 est abandonnée si**, à 5°, le code-barres
fait moins bien que le rayon le plus proche ou se dit sûr moins d'une fois sur
dix : les espacements sont alors trop réguliers ou trop instables.

La courbe de justesse en fonction de K dit combien de spires de référence il
faut. Elle ne suffit pas à tenir les 8 heures du Grand Prize : à 25 heures
d'annotation par spire, même une référence toutes les 30 spires coûte trop
cher. Les références devront venir d'ailleurs, par exemple des fibres
numérotées par le solveur de flot de l'équipe.

## Paramètres

| Paramètre | Défaut | Sens |
|---|---|---|
| `--pos-per-gap` | 8 | Positions testées par espacement moyen entre feuilles |
| `--alpha` | 0,35 | Séparation minimale, en part de l'espacement moyen |
| `--prior-weight` | 0,05 | Rappel vers la division égale là où l'image ne montre rien |
| `--theta-stride` | 1 | Un rayon sur N, pour aller plus vite |
| `--max-drop` | 3 | Feuilles retirées au hasard à chaque bord de la pile |

Durée de la coupe avec SciPy : environ 15 s pour 330 000 nœuds, 3 minutes pour
1,4 million (cas le plus défavorable). Pour K = 20 ou 30, commencer avec
`--theta-stride 4`.

## Limites connues

- **2D seulement.** Aucune contrainte de continuité selon z.
- **Détection parfaite pour H2.** Le code-barres est mesuré sur les rayons de
  la vérité terrain. C'est un plafond : si le plafond est mauvais, inutile de
  continuer. Avec des crêtes détectées, il faudra gérer les feuilles manquantes
  (alignement de séquences).
- **Rayon fonction de l'angle.** Une feuille qui se replie sur elle-même casse
  la vue polaire. Le portage 3D devra passer par le repère de l'ajusteur de
  spirale.
- **Déchirures.** Si le nombre de feuilles diffère de part et d'autre d'une
  zone, le décalage entier n'a plus de sens.
- **La couture à l'angle 0** n'est pas traversée : les mesures restent à
  l'intérieur d'un tour.
- **Une surface par spire** est supposée. Des surfaces qui couvrent plus d'un
  tour fausseraient la pile ; la figure de contrôle le montre.

## Règles du dépôt `villa` à garder en tête

Le `CONTRIBUTING.md` accepte les assistants IA, à trois conditions : la
contribution vient d'un humain qui travaille sur les données des rouleaux, son
utilité est mesurée sur de vraies données (les exemples synthétiques ne
comptent pas), et le commentaire est écrit par l'humain. Le fantôme de ce
paquet ne peut donc servir que de test du code.
