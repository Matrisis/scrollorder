# Campagne de tests pour le déroulage : masse, ordre et coupe ordonnée

- **Date :** 15 novembre 2026
- **Statut :** conception, en attente d'approbation par Matthieu
- **Projet :** paquet `scrollorder` (coupe ordonnée, code-barres, lecture des surfaces ; 63 tests automatiques ; validé sur une coupe synthétique seulement, jamais sur de vraies données)

## 1. Pourquoi cette campagne

Le déroulage des rouleaux bute sur les zones pressées, où les feuilles se touchent. Les mesures actuelles s'appuient surtout sur le creux visible entre deux feuilles. Quand ce creux disparaît, elles deviennent peu fiables au même endroit.

Notre idée de départ est de traiter le déroulage comme un problème d'ordre. Les feuilles ne se croisent jamais, donc on peut les placer toutes ensemble avec un algorithme exact, celui qui sert à séparer les couches de la rétine. Il lui manquait une information : combien de feuilles passent entre deux repères. La piste du 15 novembre est de compter les feuilles par la quantité de papyrus, qui ne dépend pas des creux.

La campagne doit dire, sur de vraies données et avec des seuils fixés d'avance, lesquelles de ces idées tiennent. Si elles tiennent, elles répondent à deux demandes publiques de l'équipe du Vesuvius Challenge (message du 14 novembre 2026 sur le Discord) : une autre façon d'estimer la distance entre spires, et une façon de produire des surfaces à partir de fibres numérotées.

## 2. La logique d'ensemble

Quatre briques, chacune avec son rôle :

- **L'ordre** dit ce qui est interdit : deux feuilles ne se croisent pas et ne fusionnent pas. C'est une contrainte dure.
- **La coupe ordonnée** place les feuilles. C'est l'algorithme de la rétine : une coupe de graphe qui trouve la meilleure position de toutes les feuilles à la fois, en respectant l'ordre.
- **La masse** dit combien de feuilles il y a, même quand on ne les distingue plus.
- **Le vide** dit où passent les frontières quand il existe, et sert à recaler la masse.

Dijkstra et les certificats d'ordre sont deux autres usages de la même logique : numéroter des points en payant un péage de masse, et prouver qu'un tracé a sauté d'une feuille à l'autre.

## 3. Buts, non-buts, à éviter

**Buts**

1. Trancher chaque hypothèse sur de vraies données, avec des seuils écrits avant de regarder les résultats.
2. Comparer chaque méthode à une référence simple et, quand elle existe, à l'estimation de l'équipe (la densité de feuilles apprise par `lasagna`).
3. Rendre chaque test rejouable : une commande, un tableau de résultats, une figure.

**Un résultat est dit probant** s'il passe son seuil sur les coupes de mesure et s'il fait mieux que l'estimation de l'équipe dans les zones pressées. S'il passe son seuil sans faire mieux, il est vrai mais pas nouveau, et on le dit ainsi. Pour les tests où l'équipe n'a pas d'estimation comparable (T2, T4, T5, T6), probant veut dire : passer le seuil et battre toutes les références simples.

**Non-buts**

- Dérouler un rouleau entier ou travailler en 3D complet. Les tests portent sur des coupes 2D et de petits blocs.
- La détection d'encre.
- Entraîner un réseau de neurones ou utiliser une carte graphique.
- Reproduire le pipeline de l'équipe.

**À éviter**

- Un résultat qui ne tient que sur la coupe synthétique. Elle sert à tester le code, pas les idées.
- Régler un paramètre sur les coupes de mesure.
- Annoncer un succès sans référence de comparaison ou sans contrôle négatif.
- Livrer un outil de mesure sans méthode : les prix mensuels le découragent.

## 4. Contraintes

- **Exécution en natif sur le Mac de Matthieu** (processeur Apple, pas de carte NVIDIA), par Claude Code en local. La session distante et son pont vers le Mac n'atteignent pas les serveurs de données (vérifié le 15 novembre).
- **Le solveur de fibres de l'équipe (`vc_wgraph`) exige une carte NVIDIA.** Le test T4d le simule.
- **Lecture des données par morceaux** depuis le dépôt public, avec un cache local hors du dossier du code. Budget disque indicatif : 60 Go, dont 20 Go au plus pour T5. Le Mac a 462 Go libres : on demande à Matthieu avant de dépasser.
- **Volumes publics en 8 bits**, d'après le preprint de l'équipe (arXiv 2606.29085). La fenêtre de conversion n'est pas documentée, et nous n'avons pas pu lire nous-mêmes le type des volumes. C'est le premier risque, traité par T0.
- **Licence des données :** CC BY-NC 4.0, avec citation.
- **Règles du dépôt `villa` :** utilité mesurée sur de vraies données, commentaire écrit par un humain. Licence permissive exigée pour concourir aux prix.
- **Le catalogue de données lu date de juin 2026** (`scrollprize.org/static/data_browser/index.json`). Les chemins de la section 6 sont à revérifier sur le serveur.
- **Le code nouveau est testé d'abord sur des cas synthétiques** dont on connaît la réponse, comme le paquet actuel. Les vraies données servent aux mesures, pas aux tests du code.

## 5. Les formes envisagées

- **A. Un arbre de décision (retenu).** D'abord le contrôle des niveaux de gris, puis deux tests en parallèle : la règle de masse et la coupe ordonnée avec repères. Les autres tests ne démarrent que si celui dont ils dépendent passe. Avantage : chaque journée de travail sert, même en cas d'échec. Inconvénient : plus lent en temps écoulé.
- **B. Tout en parallèle.** Plus rapide avec plusieurs agents. Écarté : si les niveaux de gris sont inexploitables, la moitié du travail est perdue, et il devient difficile de garder les seuils honnêtes.
- **C. Reproduire d'abord le pipeline de l'équipe**, puis mesurer ce que nos idées ajoutent. C'est la forme la plus proche d'une contribution finale. Écartée pour l'instant : elle demande une carte NVIDIA et plusieurs semaines.

Matthieu a approuvé la forme A le 15 novembre, en demandant que la coupe ordonnée et la logique combinatoire restent au centre. La coupe avec repères est donc dans la première vague.

## 6. Données

Dépôt public : `s3://vesuvius-challenge-open-data/`, aussi accessible en HTTPS sur `vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com`. Les volumes sont en OME-Zarr à plusieurs niveaux de résolution, lisibles par morceaux. Les surfaces tracées sont en `tifxyz` (trois images x, y, z) ou en OBJ.

**PHerc. 0139, banc d'essai principal**

- Volume à 2,4 µm : `PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/`
- 37 spires consécutives tracées une par une, de `w023` à `w059`, dans `PHerc0139/segments/` (dossiers du type `<id>-w059_2025022312/`). Elles sont repérées sur le volume ci-dessus. La présence et le format des maillages dans ces dossiers restent à vérifier (T0).
- Prédictions de l'équipe sur ce même volume, dans `PHerc0139/representations/predictions/` : `lasagna` au niveau 2 (`lasagna/20260102150214-lasagna-20260419180421-L2/`) et deux modèles de fibres au niveau 1.
- Second scan à 9,4 µm (`20250728140407-9.362um-1.2m-113keV-masked.zarr`), avec deux autres reconstructions du même scan (`20251107132835-…-pag0-masked.zarr` et `20251107135911-…-pag50-masked.zarr`).
- Axe du rouleau : aucun fichier connu. À chercher sur le serveur, sinon à estimer à partir des spires tracées.

**PHerc. 1667, réplication**

- Volume à 2,4 µm : `PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/`
- 14 spires consécutives (`w028` à `w041`) et 5 isolées, d'après le catalogue. Prédiction `lasagna` au niveau 2.

**PHerc. Paris 4, certificats d'ordre (T5)**

- Jeu `spiral-input` (`dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/`, 49,6 Go au total) : environ 27 400 patchs vérifiés et 203 900 non vérifiés (maillages `tifxyz`), `relative_windings.json`, `abs_winding.json`, `same_windings.json`, `umbilicus.json`. Coordonnées en voxels de 9,6 µm. Description : `scrollprize.org/static/data/datasets/spiral-input-PHercParis4-README.md` dans le dépôt `villa`.

**Résolution de travail.** Niveau 2 du volume à 2,4 µm (voxels d'environ 9,6 µm), sauf mention contraire. À cette résolution, une feuille fait de l'ordre de 10 à 15 voxels d'épaisseur.

**Choix des coupes.** Sur chaque rouleau, 50 coupes horizontales dans la plage de hauteur couverte par toutes les spires, espacées d'au moins 0,2 mm.

- 10 coupes de réglage, prises dans le cinquième inférieur de la plage.
- 40 coupes de mesure, prises dans le reste, à plus de 0,5 mm des coupes de réglage.
- La liste est tirée une fois, après T0, avec une graine fixe, et enregistrée dans le dépôt.
- Si la plage est trop courte pour 50 coupes, on en prend moins, en gardant une coupe de réglage pour quatre de mesure.
- Si une coupe complète coûte trop cher à télécharger, on travaille par secteurs angulaires ou on regroupe les coupes par couches de morceaux. Ce choix se fait après T0 et avant le tirage.

## 7. Règles communes

**Vocabulaire et géométrie**

- **Vers l'extérieur :** en s'éloignant de l'axe du rouleau.
- **Courbe :** l'intersection d'une spire tracée avec une coupe. La **courbe suivante** est celle qui vient juste à l'extérieur. Le sens de la numérotation des dossiers (`w023`, `w024`…) est à vérifier en T0.
- **Hypothèse du recto.** Les surfaces tracées suivent la face de la feuille tournée vers l'axe. Le papyrus d'une feuille est donc juste à l'extérieur de sa courbe, et le vide juste à l'intérieur. À vérifier en T0. La règle de masse (T1) n'en dépend pas : entre deux courbes consécutives il y a une feuille entière, où que la surface passe dans la feuille.
- **Normale `n` :** la direction perpendiculaire aux feuilles, en 3D. Elle vient des surfaces tracées quand on les connaît (T1, T4), et du scan sinon (T2, T3, T6).
- **Niveau de l'air `a` :** une carte lisse, par coupe, construite à partir du gris au fond des vides larges d'au moins 4 voxels. Dans un paquet sans vide, on prend la valeur interpolée depuis les vides voisins, jamais un minimum local.
- **Densité `ρ` :** le gris moins `a`. On ne coupe pas les valeurs négatives, sauf mention contraire.
- **Masse traversée** le long d'un chemin : la somme de `ρ × |n·dx|`, où `dx` est le pas du chemin. Le facteur `|n·dx|` corrige les traversées en biais : une feuille inclinée par rapport à la coupe, ou coupée de travers, paraît plus épaisse qu'elle ne l'est.
- **Profil :** le gris le long d'une droite qui part d'une courbe, perpendiculaire à elle dans le plan de la coupe, vers l'extérieur. Pas d'un demi-voxel, interpolation linéaire.
- **Arc :** un tronçon de 0,5 mm de courbe. La valeur d'un arc est la médiane de ses profils. Les arcs où la feuille est trop couchée (`|n·dx|` inférieur à la moitié du pas) sont écartés et comptés.
- **Espacement :** la distance entre une courbe et la suivante, corrigée de l'inclinaison. Pour un trajet qui traverse `m` feuilles, on prend l'espacement moyen du trajet.
- **Zone propre :** dans une coupe, les arcs dont l'espacement dépasse la médiane de cette coupe.
- **Décile serré :** les 10 % de trajets les plus resserrés, pris sur l'ensemble des coupes de mesure d'un rouleau et calculés à part pour chaque `m` (ou chaque `K`). C'est notre définition de la zone pressée.
- **Masse d'une feuille `σ` :** la masse traversée d'une courbe à la suivante.
- **Référence `σ_ref` :** la médiane de `σ` sur les arcs de la zone propre de la coupe, raccords exclus. Elle ne se sert donc jamais de la zone pressée.
- **Raccord :** l'endroit où deux feuilles de fabrication sont collées bout à bout. Elles se chevauchent, et la spire y compte deux épaisseurs de papyrus au lieu d'une. La masse compte donc des **couches** de papyrus, pas des spires. La part de chaque spire couverte par des raccords n'est pas connue : T1 la mesure.
- **Raccord reconnu :** un tronçon d'au moins 5 mm d'une seule spire où la masse vaut entre 1,6 et 2,4 `σ_ref`, alors que les deux spires voisines pèsent normalement au même endroit, et qu'on retrouve au même endroit de la feuille (à 3 mm près) dans les deux coupes les plus proches. Cette règle est fixée ici, avant toute mesure. Si l'équipe a publié des raccords annotés pour le rouleau, on les prend à la place.
- **Couches attendues** le long d'un trajet : le nombre de spires traversées, plus le nombre de raccords reconnus traversés. Les compteurs de T1, T2, T4c et T6 sont notés contre ce nombre.

**Statistiques**

- L'unité est l'arc. Les intervalles de confiance à 95 % se calculent en rééchantillonnant les coupes entières, parce que les arcs voisins se ressemblent.
- La décision porte sur la valeur mesurée. L'intervalle est rapporté à côté.
- « Faire mieux que » veut dire : l'intervalle de la différence, calculée arc par arc, ne contient pas zéro.

**Règles**

- Chaque test a un **critère décisif**, avec un seuil de réussite et un seuil d'arrêt. Les autres mesures servent à comprendre le résultat : elles ont une valeur attendue, pas un seuil.
- Les seuils de ce document sont gelés. Tout changement s'écrit et se date avant de regarder les coupes de mesure.
- Entre le seuil d'arrêt et le seuil de réussite, c'est la zone grise : un seul raffinement est permis, celui que le test annonce, puis on tranche.
- Quand un test n'a qu'un seuil, ne pas l'atteindre vaut arrêt.
- Chaque test a un contrôle négatif, qui doit échouer.
- Chaque test produit un tableau CSV, au moins une figure, et un verdict de cinq lignes écrit par Matthieu.
- Un test qui passe sur PHerc. 0139 est rejoué sur PHerc. 1667 sans retoucher les paramètres, dans la limite de ses 14 spires consécutives (pas de longue distance).

## 8. Les tests

### En bref

- **T0, l'instrument.** Les niveaux de gris des scans publics permettent-ils de peser le papyrus ?
- **T1, la règle de masse.** La quantité de papyrus entre deux spires donne-t-elle le nombre de feuilles ? *Intégrale du gris.*
- **T2, l'escalier.** La masse entre deux vides vaut-elle toujours un nombre entier de feuilles ? *Recalage d'entiers, comme en GPS.*
- **T3, la distance en feuilles.** Peut-on numéroter une région à partir d'une seule spire ? *Dijkstra avec péage de masse, comparé à Poisson.*
- **T4, la coupe ordonnée.** Placer toutes les feuilles ensemble fait-il mieux que les placer une à une ? *Coupe de graphe de la rétine.*
- **T5, les certificats d'ordre.** L'empilement des patchs prouve-t-il qu'un tracé a sauté de feuille ? *Tarjan, plus longs chemins.*
- **T6, les fibres.** Le changement de direction des fibres compte-t-il les feuilles sans espace d'air ? *Tenseur de structure.*
- **T7, le code-barres.** Une feuille a-t-elle une signature reconnaissable ? *Alignement de séquences.*
- **T8, le budget humain.** Combien de clics pour certifier une coupe ?

### T0. L'instrument : les niveaux de gris sont-ils exploitables ?

- **En clair.** Avant de peser le papyrus, on vérifie la balance.
- **Ce qu'on fait.**
  1. Ouvrir le volume, les 37 surfaces et la prédiction `lasagna`. Noter le type des données, les niveaux, la taille des morceaux, l'ordre des axes, l'échelle des coordonnées des surfaces, les canaux de `lasagna`, et si l'équipe publie des raccords annotés pour ce rouleau.
  2. Superposer les courbes tracées à la coupe (commande `scrollorder stack`, figure de contrôle). Vérifier le sens de la numérotation et l'hypothèse du recto.
  3. Histogramme des gris entre les courbes : part des voxels à 0 et à 255, y compris dans le décile serré.
  4. Carte du niveau de l'air sur une coupe.
  5. Comparer un niveau de la pyramide à la moyenne par blocs de 2 du niveau plus fin, pour savoir si le sous-échantillonnage est une moyenne.
  6. Sur PHerc. 0139 à 9,4 µm, comparer les moyennes de gris par blocs de 0,5 mm entre les reconstructions `pag0` et `pag50`. Les deux fenêtres de conversion peuvent différer : on cherche une relation en ligne droite, pas une égalité.
- **Critère décisif.**
  - Les courbes suivent les feuilles à l'œil.
  - Moins de 1 % des voxels à 255 entre les courbes, décile serré compris.
- **Mesures pour comprendre.**
  - Hypothèse du recto : pour au moins 90 % des spires, le gris moyen des 3 voxels à l'extérieur de la courbe dépasse celui des 3 voxels à l'intérieur. Sinon, on mesure le décalage moyen entre la courbe et le bord du papyrus, et on l'applique dans T2, T3 et T4a.
  - Moyennes `pag0` et `pag50` alignées sur une droite, avec un R² d'au moins 0,99. Si ce point échoue seul, on continue vers T1 avec cette réserve : c'est un indice, pas une preuve.
  - Si les vides valent exactement 0, le masque de l'équipe les a effacés. Alors `a = 0`, et on vérifie sur une figure que le masque ne mord pas sur le papyrus.
  - Si la pyramide n'est pas une moyenne (écart de plus de 2 %), on refait nous-mêmes la moyenne à partir du niveau 0 pour les mesures de masse.
- **Si ça passe.** Les mesures de masse ont un sens sur ces volumes.
- **Si les courbes ne suivent pas les feuilles.** C'est un problème de repère (échelle, axes), pas un échec de l'idée. On le corrige avant d'aller plus loin.
- **Si les maillages de PHerc. 0139 sont introuvables.** On les cherche sur `dl.ash2txt.org`. À défaut, PHerc. 1667 devient le banc d'essai principal. En dernier recours, on prend les patchs vérifiés de Paris 4 avec `abs_winding.json`.
- **Si le papyrus sature à 255.** On ne peut pas conclure sur la masse avec les volumes publics. L'idée n'est pas réfutée : on demande à l'équipe des volumes 16 bits. Les tests T4a, T4d et T5 continuent, ils n'utilisent pas la masse.
- **Effort.** 1,5 jour.

### T1. La règle de masse : la quantité de papyrus compte-t-elle les feuilles ?

- **En clair.** On pèse ce qu'il y a entre deux spires connues, on divise par le poids d'une feuille, et on regarde si on retrouve le bon nombre de feuilles.
- **Algorithme.** Intégrale du gris le long de la normale. Aucun apprentissage.
  - `M` : la masse traversée d'une courbe à la `m`-ième courbe suivante, pour `m` de 1 à 10. La normale vient des deux courbes de départ et d'arrivée, pas des courbes cachées entre elles.
  - Carte des raccords : la masse de chaque spire le long de la feuille, coupe par coupe. On y applique la règle du raccord reconnu.
  - Compte estimé : `M / σ_ref`, arrondi à l'entier. Il est exact s'il vaut le nombre de couches attendues.
  - Les trajets qui passent par un trou d'une surface tracée sont écartés et comptés.
  - Variante rapportée à côté : la masse totale de la bande comprise entre les deux courbes, divisée par leur longueur moyenne. Elle se passe de normales dans le plan.
- **Références de comparaison**, sur les mêmes trajets :
  - la distance divisée par l'espacement médian de la zone propre ;
  - le nombre de crêtes du gris lissé ;
  - l'intégrale de la densité de feuilles de `lasagna` (canal `grad_mag`), brute puis étalonnée comme la masse.
- **Critère décisif : le compte.** Part de comptes exacts dans le décile serré. Réussite à 90 % pour chaque `m` de 1 à 4. Arrêt sous 70 % pour `m = 2`.
- **Probant** si, en plus, la masse fait mieux que `lasagna` dans le décile serré, sur l'ensemble des trajets de `m` = 2 à 4. `lasagna` est noté contre les spires seules, qui sont sa cible.
- **Mesures pour comprendre.**
  - **Stabilité.** Dispersion de `σ` par arc en zone propre (écart-type divisé par la moyenne). Attendu : sous 15 %.
  - **Conservation.** Rapport entre la médiane de `σ` dans le décile serré et en zone propre, raccords exclus. Attendu : entre 0,95 et 1,05.
  - **Compte brut.** La même part de comptes exacts, notée contre les spires seules. C'est ce qu'on obtiendrait sans carte des raccords.
  - **Raccords.** Part de chaque spire qu'ils couvrent, part des trajets du décile serré qui en croisent un. Alerte s'ils sont plus fréquents dans le décile serré qu'en zone propre : la règle excuserait alors de vraies erreurs.
  - **Erreur en fonction de `m`.** Biais et dispersion de l'écart entre `M / σ_ref` et les couches attendues.
  - **Résolution.** Même mesure aux niveaux 1, 2 et 3. Attendu : la part de comptes exacts ne perd pas plus de 5 points du niveau 1 au niveau 3.
- **D'où viennent ces chiffres.** Avec 15 % de dispersion par feuille et aucun biais, le compte de 4 feuilles est juste 9 fois sur 10. Un biais de 10 % sur le poids d'une feuille le fait tomber sous 70 %, même avec une dispersion de 10 %. Si le compte passe alors que la stabilité ou la conservation sortent de l'attendu, on le signale comme une alerte.
- **Contrôle négatif.** Remplacer le gris par une constante : on retombe sur la distance, qui doit échouer dans le décile serré.
- **Si ça passe.** La somme des gris compte les couches de papyrus là où les creux ont disparu, sur ce rouleau et ce scan. On obtient au passage une carte des raccords entre feuilles.
- **Ce que ça ne prouve pas.** Que ça marche sur un autre rouleau, ni que ça améliore un déroulage complet. Ni qu'on sait passer des couches aux spires sans surfaces tracées : la carte des raccords s'appuie ici sur elles. Ni que ça tient sur de longues distances : avec 15 % de dispersion, le compte de 10 feuilles n'est juste que 7 fois sur 10. C'est le rôle du recalage de T2.
- **Si ça échoue.** Avec ces volumes et cet étalonnage, la masse n'est pas assez précise. On abandonne T2, T3, T4b, T4c et T8.
- **Raffinement permis en zone grise** (un seul, choisi d'après l'erreur en fonction de `m`) :
  - si l'erreur grandit avec `m`, moyenner sur un pavé de 0,5 mm de côté (le long de l'arc et en hauteur) au lieu d'un arc ;
  - si l'erreur est la même pour tous les `m`, recaler les deux extrémités sur le bord du papyrus.
- **Pièges connus.**
  - Les surfaces tracées n'existent que là où un humain a pu tracer. Les pires zones peuvent manquer.
  - `lasagna` a peut-être été entraîné sur ces mêmes spires. La comparaison lui serait alors favorable.
  - Les raccords. Si chaque spire en porte sur 5 % de sa longueur, un trajet de 4 feuilles en croise un près d'une fois sur cinq. Sans la règle du raccord reconnu, le test échouerait pour une raison étrangère à l'idée.
  - Les grains minéraux et les sédiments pèsent plus lourd que le papyrus. La médiane par arc limite leur effet.
- **Effort.** 2,5 jours.

### T2. L'escalier : la masse est-elle quantifiée ? (idée « masse contre vide »)

- **En clair.** On cumule la masse en traversant les feuilles. La courbe est un escalier : elle monte dans le papyrus et reste plate dans le vide. Si l'idée est juste, entre deux vides la masse vaut toujours un nombre entier de feuilles : 1, 2, 3. Un paquet qui pèse 1 est une feuille isolée, confirmée deux fois : par les vides qui la bordent et par son poids.
- **Algorithme.**
  - Masse cumulée le long d'un trajet qui traverse au moins 10 spires, en unités de `σ_ref`. La normale vient du scan.
  - Vide : au moins 2 voxels de suite où la densité est sous un seuil réglé sur les coupes de réglage.
  - **Paquet :** ce qu'il y a entre deux vides consécutifs. On mesure sa masse.
  - Recalage d'entier : à chaque vide, si le cumul est à moins de 0,3 d'un entier, on le ramène à cet entier. Sinon on signale l'endroit. C'est le principe de la fixation d'ambiguïtés entières en GPS.
- **Critère décisif**, en deux parties :
  - **Paquets.** Pour les paquets qui contiennent au moins 2 courbes tracées, la masse arrondie donne le nombre de couches attendues. Réussite à 90 %, arrêt sous 70 %.
  - **Longue distance.** Compte exact des couches attendues entre deux courbes distantes de 20 à 30, avec recalage. Réussite à 90 %. Sans recalage, on attend entre 50 et 70 %.
- **Mesures pour comprendre.**
  - Sans vérité terrain : parmi les paquets de plus de 1,5 feuille, part de ceux dont la masse est à moins de 0,25 d'un entier. Attendu : 70 % ou plus. Le hasard donne 50 %.
  - La masse totale le long de rayons entiers, de l'axe au bord, rapportée angle par angle.
- **Contrôles négatifs.**
  - Remplacer la masse par la distance : le critère sur les paquets doit échouer.
  - Multiplier `σ_ref` par 1,25 : pour les paquets de 2 et 3 feuilles, la part proche d'un entier doit tomber sous 50 %.
- **Si ça passe.** La masse et le vide disent la même chose, et chaque vide remet l'erreur à zéro. L'histogramme de la masse des paquets, avec des pics sur 1, 2, 3, se vérifie à l'œil. Il se trace aussi sur un rouleau sans aucune spire tracée.
- **Ce que ça ne prouve pas.** Rien sur la cohérence d'un trajet à l'autre, ni sur le tour complet du rouleau.
- **Si ça échoue alors que T1 passe.** La masse compte sur de courtes distances mais dérive. On la garde pour les petits paquets seulement.
- **Piège connu.** Une feuille peut se dédoubler de l'intérieur. Le vide qui s'ouvre la coupe en deux moitiés, et le cumul tombe sur un demi. Dans ce cas on ne recale pas, on signale l'endroit.
- **Effort.** 1 jour. Dépend de T1.

### T3. La distance en feuilles : Dijkstra avec péage de masse (idée « relier des points connus »)

- **En clair.** Avancer le long d'une feuille est gratuit, traverser du papyrus coûte sa masse. Le chemin le moins cher entre deux points coûte alors le nombre de feuilles qui les séparent. À partir d'une seule spire connue, on numérote toute une région.
- **Algorithme.**
  - Plus court chemin de Dijkstra (1959) sur la grille des pixels, dans un secteur de 45° de la coupe. Deux secteurs par coupe : le plus serré et un secteur propre.
  - Coût d'un pas : la masse traversée par ce pas, divisée par `σ_ref`, plus un très petit péage par voxel parcouru. Ici la densité négative est ramenée à zéro, parce que Dijkstra n'accepte pas de coût négatif.
  - Direction des feuilles : tenseur de structure du gris lissé, ou directions de `lasagna`.
  - Départ : tous les pixels d'une courbe connue, au milieu du secteur.
- **Ce qu'on s'attend à voir.** Dijkstra prend le chemin le moins cher, donc il passe par les trous et les fissures et sous-compte. La parade est de lisser la densité le long des feuilles sur 0,5 mm.
- **Trois estimateurs comparés**, sur la même donnée :
  - l'intégrale en ligne droite de T1 ;
  - Dijkstra avec lissage (l'estimateur principal) et sans ;
  - une résolution de Poisson aux moindres carrés, qui fait la moyenne sur tous les chemins au lieu de prendre le moins cher.
- **Critère décisif.** Part des pixels des courbes tracées qui reçoivent le bon numéro, jusqu'à 10 spires du départ et sans aller au-delà du premier raccord reconnu du secteur. Réussite à 90 % sur le secteur et à 80 % dans le décile serré. Arrêt sous 70 % sur le secteur. Le seuil s'applique à Dijkstra avec lissage. Les deux autres sont rapportés à côté, et si Poisson fait mieux, on le dit.
- **D'où vient ce chiffre.** Avec 15 % de dispersion par feuille, la ligne droite donne environ 85 %. Le seuil de 90 % mesure donc ce qu'apporte la mise en commun le long des feuilles.
- **Référence.** La même propagation faite avec la densité de `lasagna` à la place de la masse.
- **Contrôle négatif.** Densité constante : les numéros doivent être faux dans le décile serré.
- **Si ça passe.** On peut propager une numérotation à travers une zone pressée à partir d'une seule spire, sans repère supplémentaire.
- **Ce que ça ne prouve pas.** Le tour complet du rouleau (la condition « un tour ajoute une spire » n'est pas testée ici), ni la 3D.
- **Effort.** 2 jours. Dépend de T1.

### T4. La coupe ordonnée : l'algorithme de la rétine

- **En clair.** Entre deux repères, on sait combien de feuilles passent. On cherche la position de toutes ces feuilles à la fois. On interdit les croisements et les feuilles trop proches. La coupe de graphe donne la meilleure solution de toutes, pas une approximation.
- **Algorithme.** Recherche optimale de surfaces multiples par coupe minimale (Li, Wu, Chen et Sonka, 2006), déjà codée dans `scrollorder/ordered_cut.py` et vérifiée contre une recherche exhaustive sur de petits cas.

**T4a. Avec repères connus** (première vague, code existant : commande `scrollorder cut`)

- Repères : une courbe et la `K`-ième suivante, pour `K` = 2, 5, 10, 20 et 30. On cache les `K−1` courbes entre elles.
- Indice : le bord du vide, c'est-à-dire l'endroit où le gris monte en allant vers l'extérieur (de l'air au papyrus). C'est là que passe le recto. On ne garde que le sommet de chaque montée. Variante rapportée à côté : le même test avec un indice tiré de `lasagna`.
- Trois méthodes comparées : division à parts égales entre les repères, accrochage de chaque feuille à l'indice le plus proche, coupe ordonnée.
- Mesure : part des points placés sur la bonne spire (plus près de leur vraie courbe que de toute autre), et part des feuilles fusionnées.
- Critère décisif, à `K = 10` et dans le décile serré : la coupe dépasse la division égale d'au moins 10 points et l'accrochage d'au moins 5 points, sans aucune fusion. Arrêt si la coupe est à moins de 2 points de la division égale. On prend toutes les paires de repères distantes de 10 spires, décalées de 5 en 5.
- `K` = 20 et 30 sont rapportés à côté : ils disent jusqu'où on peut écarter les repères, mais 37 spires n'en donnent qu'une paire indépendante chacun.
- Le README disait « `K` de 10 ou plus » avec le seul premier seuil. On précise `K = 10` et on ajoute la comparaison à l'accrochage.
- Contrôle négatif : une image sans indice. La coupe doit alors redonner exactement la division égale.
- Si ça passe : placer les feuilles ensemble, sous contrainte d'ordre, fait mieux que les placer une à une.
- Ce que ça ne prouve pas : qu'on sait se passer de repères tracés à la main.
- Effort : 1 jour.

**T4b. En coordonnée de masse** (dépend de T1)

- On remplace la distance par la masse cumulée. La séparation entre deux feuilles vaut entre 0,6 et 1,5 `σ_ref`. Le rappel se fait vers la division à masses égales. Aux raccords reconnus, la borne haute est levée et les points ne sont pas notés.
- Méthode ajoutée à la comparaison : la division à masses égales seule, sans coupe.
- Critère décisif, à `K = 10` et dans le décile serré : la coupe en masse fait au moins aussi bien que la coupe de T4a. Mesure pour comprendre : la division à masses égales dépasse-t-elle la division à distances égales ?
- Ce que ça ne prouve pas : dans un paquet de densité uniforme, masses égales et distances égales reviennent au même. Le gain attendu vient des paquets inégaux.
- Effort : 1,5 jour.

**T4c. Sans compter à la main** (dépend de T1 et T2)

- **Étape 1.** Entre deux courbes éloignées, le nombre de feuilles vient de la masse, plus de la vérité terrain. Si le nombre est faux, tout le placement compte comme faux. Critère décisif : au plus 5 points de moins que T4b, à `K = 10`. Effort : 1 jour.
- **Étape 2, la spirale entière.** Plus aucune courbe connue. Sur l'angle déroulé, la spirale est une seule surface qui garde ses distances avec elle-même un tour plus loin. Le problème reste une seule coupe.
  - Ce qui est fixé. On teste d'abord sur la coupe synthétique, puis sur une bande de 12 spires tracées, élargie d'un pas et demi de chaque côté. Le nombre de tours vient de la masse. Il faut des coupes entières, pas des secteurs : 5 de réglage et 10 de mesure. Critère décisif : au plus 5 points de moins que la coupe avec repères sur la même bande, dans le décile serré.
  - Ce qui reste à concevoir. Le traitement des bords de la bande et de la couture entre deux tours. Une courte note de conception (une demi-journée) sera soumise à Matthieu après la vague 2, parce qu'elle dépend de ce que T0 à T2 auront montré.
  - Effort : 3 à 4 jours après la note.

**T4d. Avec des repères clairsemés, comme des fibres numérotées** (dépend de T4a)

- But : répondre à la demande de l'équipe, qui sait numéroter des fibres mais pas encore en tirer des surfaces. Ici aucune courbe entière n'est connue.
- Montage : on prend 11 spires consécutives. Sur chacune, on ne révèle que de petits arcs de 1 mm avec leur numéro de spire, comme des fibres horizontales. Ils couvrent 50, 20, 10, 5 ou 2 % de la longueur de chaque spire. On fausse d'une spire le numéro de 5 % d'entre eux. La bande de recherche est construite à partir de ces seuls repères, avec une marge d'un pas et demi de chaque côté.
- La coupe place les 11 spires à la fois. Elle utilise le bord du vide comme en T4a. Les repères attirent la feuille sans l'obliger.
- Critère décisif : avec 10 % de couverture et 5 % de numéros faux, 95 % des points sur la bonne spire, sur l'ensemble des coupes de mesure.
- Mesures pour comprendre :
  - la courbe de justesse en fonction de la couverture, et le chiffre du décile serré ;
  - la part des numéros faux corrigés par la contrainte d'ordre (attendu : 80 %) ;
  - la version facile, où les deux spires extrêmes sont connues en entier ;
  - la version où les repères sont des points isolés, comme des fibres verticales, en même nombre.
- Référence : interpoler chaque spire entre ses propres repères.
- Si ça passe : produire des surfaces denses à partir de fibres numérotées est faisable, et on sait quelle densité de fibres il faut.
- Ce que ça ne prouve pas : que ça marche avec les vraies sorties du solveur de l'équipe. Ce sera un test à part, avec une carte NVIDIA ou des sorties publiées.
- Effort : 2 à 3 jours.

### T5. Les certificats d'ordre (maillages seuls, sans scan)

- **En clair.** Si le patch A est sous le patch B à un endroit, il ne peut pas être au-dessus ailleurs. Une contradiction prouve qu'un des deux a sauté de feuille.
- **Algorithme.**
  - Chaque patch reçoit un numéro de spire entier, inconnu. On tient compte du passage par l'angle zéro autour de l'axe : un patch qui le traverse change de numéro d'un côté à l'autre.
  - On lance des rayons le long des normales, vers l'extérieur, sur 2 à 3 spires de distance. Quand un rayon rencontre un second patch à la distance `d`, ce patch est au moins une spire plus loin (borne basse), et au plus `d / p_min` spires plus loin, arrondi à l'entier inférieur (borne haute). `p_min` est le plus petit pas possible entre deux spires.
  - Deux patchs superposés à moins d'une petite tolérance sont sur la même spire.
  - Contradictions : un cycle dans les bornes basses, trouvé par composantes fortement connexes (Tarjan, 1972).
  - Intervalle entre deux patchs : la plus forte borne basse et la plus faible borne haute le long des chaînes de patchs qui les relient. Ce sont des plus longs et plus courts chemins, comme dans la méthode du chemin critique et l'algorithme de Bellman-Ford.
- **Entrées.** La géométrie seule : les patchs vérifiés et l'axe. Les trois fichiers d'annotations ne servent qu'à noter. Seule exception : dans la fenêtre de réglage, ils fixent `p_min` (le 1er centile du pas entre spires voisines) et la tolérance.
- **Données.** Paris 4, sur une fenêtre d'environ 1 000 coupes, là où les patchs sont les plus denses. Une fenêtre voisine de 200 coupes sert au réglage. Première étape : lire les trois fichiers d'annotations et noter leurs conventions. On ne note que les paires annotées dont les deux points tombent sur des patchs vérifiés de la fenêtre.
- **Critère décisif**, en quatre parties :
  - **Justesse.** Part des paires annotées dans `relative_windings.json` que nos intervalles contredisent : réussite sous 2 %, arrêt au-dessus de 5 %.
  - **Fausses alertes.** Moins de 2 % des patchs vérifiés pris dans une contradiction.
  - **Utilité.** Au moins 20 % des paires annotées reçoivent un intervalle d'une spire au plus.
  - **Détection.** Au moins 80 % de chimères retrouvées. Une chimère est un patch fabriqué en soudant en douceur deux patchs vérifiés de spires voisines.
- **Référence.** L'intervalle tiré de la seule distance radiale entre les deux points, divisée par le pas le plus grand puis par le plus petit. Probant si nos intervalles sont plus souvent étroits que ceux-là, à justesse égale.
- **Piège connu.** Là où la feuille se replie sur elle-même, « plus loin vers l'extérieur » ne veut plus dire « spire suivante ». Ces endroits donneront des fausses alertes.
- **Si ça passe.** L'ordre géométrique est une source d'information sûre : une feuille non tracée élargit un intervalle, elle ne le fausse pas. On peut alors trier les 204 000 patchs non vérifiés.
- **Ce que ça ne prouve pas.** Que le déroulage final s'améliore.
- **Si ça échoue.** Les patchs s'empilent trop rarement, ou les normales sont trop bruitées. On abandonne cette famille.
- **Effort.** 3 à 4 jours. Indépendant de tout le reste.

### T6. La polarité des fibres et le vote entre compteurs

- **En clair.** Chaque feuille a des fibres horizontales au recto et verticales au verso. En allant vers l'extérieur, passer du vertical à l'horizontal marque le début d'une nouvelle feuille, même sans espace d'air.
- **Algorithme.** Tenseur de structure 3D sur 40 blocs à 2,4 ou 4,8 µm (20 en zone serrée, 20 en zone propre), ou les prédictions de fibres de l'équipe, qui distinguent en principe les deux directions (à vérifier). Comptage des passages vertical-horizontal le long de la normale, avec deux seuils pour éviter les rebonds.
- **Ce qui existe déjà.** Un membre de la communauté se sert de ce changement de direction pour vérifier une surface, et le solveur de l'équipe apparie une fibre horizontale à la verticale qui la double. S'en servir comme compteur est, à notre connaissance, nouveau.
- **Préalable.** En zone propre, chaque feuille montre bien deux couches, horizontale puis verticale, sur au moins 80 % des profils. Sinon on arrête.
- **Critère décisif.** Compte exact des couches attendues dans le décile serré : réussite à 90 % pour chaque `m` de 1 à 4, arrêt sous 70 % pour `m = 2`.
- **Mesure pour comprendre : le vote.** Quand la masse et les fibres donnent le même nombre, on attend qu'il soit juste dans 99 % des cas, sur au moins la moitié des trajets.
- **Si ça passe.** Deux compteurs indépendants qui s'accordent donnent un compte certifié. Leurs désaccords désignent les endroits à faire vérifier par un humain.
- **Ce que ça ne prouve pas.** Rien à 9 µm, où chaque couche de fibres ne fait que quelques voxels.
- **Effort.** 2 à 3 jours.

### T7. Le code-barres, revisité avec la masse

- **En clair.** Chaque feuille a-t-elle une signature assez stable pour être reconnue de l'autre côté d'une zone illisible ?
- **Algorithme.** Notre estimateur de décalage (`scrollorder/barcode.py`), avec comme série la masse par feuille au lieu de l'espacement. Alignement de séquences avec trous (Needleman et Wunsch, 1970) pour les feuilles manquantes.
- **Préalable, gratuit avec T1.** La masse varie-t-elle d'une feuille à l'autre, et cette variation se conserve-t-elle le long de la feuille ? Sans cela, on s'arrête là.
- **Critère décisif.** Celui du README : juste à 99 % quand l'estimateur se dit sûr, sur au moins la moitié des essais, pour une bande cachée de 15°.
- **Attente.** Faible. Un essai voisin, sur les fibres, a été abandonné par un membre de la communauté faute de variation.
- **Effort.** Une demi-journée.

### T8. Le budget humain

- **En clair.** Combien de clics faut-il à un humain pour certifier une coupe, s'il ne clique que là où la méthode hésite ?
- **Algorithme.** On part de la numérotation de T3, propagée depuis une spire connue. Un clic donne le vrai numéro d'un point, tiré de la vérité terrain. Ce point devient un nouveau départ, et on recalcule. Deux façons de choisir le clic suivant :
  - guidée : au milieu de la zone où deux départs donnent des numéros différents ; à défaut, là où la masse et le nombre de crêtes ne s'accordent pas, ou là où le cumul est le plus loin d'un entier ;
  - au hasard, sur les courbes tracées.
- **Critère décisif.** Sur les pixels des courbes tracées des secteurs de T3 : atteindre 99 % de bons numéros avec au plus la moitié des clics du tirage au hasard.
- **Variante rapportée** si T4d passe : les clics servent de repères à la coupe.
- **Ce que ça ne prouve pas.** Le temps réel d'un annotateur. La conversion en heures repose sur une hypothèse de 4 secondes par clic.
- **Effort.** 2 jours. Dépend de T1 et T3.

## 9. Ordre et calendrier

```
T0 instrument
 ├─ échec ─▶ demander des volumes 16 bits ; T4a, T4d et T5 continuent
 └─ ok
     ├─▶ T1 règle de masse
     │     ├─ arrêt ─▶ abandon de T2, T3, T4b, T4c, T8
     │     └─ réussite ─▶ T2 escalier, T3 Dijkstra, T4b coupe en masse
     │                     ├─▶ T4c sans compter à la main (si T2 passe)
     │                     └─▶ T8 budget humain (si T3 passe)
     └─▶ T4a coupe avec repères ─▶ T4d repères clairsemés
T5 certificats d'ordre : à tout moment
T6 fibres, T7 code-barres : après T1
```

- **Vague 1, environ 5 jours :** T0, T1, T4a. Premier point de décision.
- **Vague 2, environ 5 jours :** T2, T3, T4b si T1 passe. T5 (3 à 4 jours) se glisse quand on veut.
- **Vague 3, 6 à 8 jours :** T4c étape 1, T4d, T6, T7.
- **Vague 4, 6 à 7 jours :** T4c étape 2 après sa note de conception, T8. Puis réplication sur PHerc. 1667.

Si tout passe, le total est de 24 à 28 jours de travail, hors réplication. Ce sont des estimations. Après la vague 1, quel que soit le résultat, Matthieu présente les chiffres à la communauté et demande si l'idée a déjà été essayée.

## 10. Ce que la campagne peut et ne peut pas prouver

**Elle peut prouver**

- que la somme des gris compte les couches de papyrus sur ces scans, ou qu'elle ne le fait pas ;
- que placer les feuilles ensemble sous contrainte d'ordre fait mieux que les méthodes locales ;
- que l'ordre entre patchs est une information sûre ;
- quelle combinaison marche sur des coupes de deux rouleaux.

**Elle ne peut pas prouver**

- qu'un rouleau entier se déroule en 8 heures humaines ;
- que les idées tiennent sur les rouleaux du Grand Prize, scannés à 8 ou 9 µm et sans vérité terrain (T1 en donne seulement un indice, par le test de résolution) ;
- qu'on sait passer du nombre de couches au nombre de spires sans surfaces tracées : il faut une carte des raccords, et ici elle vient des surfaces tracées ;
- que les idées sont nouvelles : nos recherches sur le Discord ne trouvent que des mots exacts ;
- que le solveur de fibres de l'équipe en tire profit, tant qu'on n'a pas ses sorties.

## 11. Décisions prises et choix laissés au constructeur

**Décisions de Matthieu**

- Se concentrer sur le déroulage, pas sur l'encre.
- Garder au centre la logique combinatoire et l'algorithme de la rétine.
- Mettre au programme ses deux idées du 15 novembre : relier des points connus par Dijkstra (T3) et opposer la masse et le vide (T2).
- Pousser les idées avant de les soumettre à la communauté.
- Forme en arbre de décision, exécutée en local sur le Mac (approuvée le 15 novembre).

**Choix de Claude, à corriger si besoin**

- PHerc. 0139 comme banc d'essai principal, PHerc. 1667 en réplication, Paris 4 pour T5.
- Niveau 2 comme résolution de travail.
- Tous les seuils chiffrés de ce document, le critère décisif de chaque test, et la définition de « probant ».
- La définition de la zone pressée par le décile serré.
- La règle qui reconnaît un raccord entre feuilles, et le fait de noter les compteurs en couches plutôt qu'en spires.
- T4a jugé à `K = 10`, avec la comparaison à l'accrochage en plus du seuil du README.
- T4d jugé sans aucune courbe entière connue.
- T5 sur la géométrie seule, les annotations étant réservées à la notation.
- T8 appliqué à la numérotation de T3.
- La spirale entière (T4c, étape 2) attend une courte note de conception après la vague 2.
- Le budget disque de 60 Go, indicatif.
- Présenter les chiffres à la communauté après la vague 1.
- Initialiser un dépôt git local dans le dossier du projet au premier jour, sans rien publier. L'historique datera le gel des seuils avant les mesures.

**Laissé au constructeur**

- La bibliothèque de lecture des volumes, le cache, l'environnement Python.
- L'organisation du code, en prolongeant le paquet `scrollorder` (une commande par test, résultats dans un dossier par test et par rouleau).
- Le choix entre profils le long des normales et rayons depuis l'axe, selon ce que montre la figure de contrôle.
- La façon de calculer la normale à partir du scan (tenseur de structure ou `lasagna`).
- Les paramètres réglés sur les 10 coupes de réglage (seuil de vide, lissage, hauteur minimale des crêtes, petit péage de T3, poids des repères de T4d).
- Les optimisations de vitesse et de mémoire.

## 12. Références

Citées de mémoire, non relues pour ce document, sauf mention.

- Coupe ordonnée : Li, Wu, Chen, Sonka, « Optimal surface segmentation in volumetric images », IEEE TPAMI, 2006.
- Plus court chemin : Dijkstra, 1959. Chemins minimaux en imagerie : Cohen et Kimmel, 1997.
- Tenseur de structure et lissage le long des structures : Weickert, 1999.
- Intégration aux moindres carrés : Ghiglia et Romero, 1994.
- Fixation d'ambiguïtés entières : Teunissen, 1995.
- Composantes fortement connexes : Tarjan, 1972. Contraintes de différence et plus courts chemins : Bellman, 1958 ; Ford, 1956.
- Alignement de séquences : Needleman et Wunsch, 1970.
- Reconstruction des scans : Paganin et al., 2002 ; preprint de l'équipe arXiv 2606.29085 (lu par un agent le 15 novembre).
- Densité de feuilles de l'équipe : `lasagna/lasagna_3d.md`, dépôt `villa` (lu le 15 novembre).
- Solveur de fibres de l'équipe : `vesuvius/docs/wgraph.md` et `wgraph-quickstart.md`, dépôt `villa` (lus le 15 novembre).
- Anatomie d'un rouleau, raccords et délaminations : `SCROLL_ANATOMY.md`, dépôt `villa` (lu le 15 novembre).
- Notes préparatoires : `research_notes/Méthodes déroulage/` (quatre rapports d'agents, relevé du Discord, relevé des données).
