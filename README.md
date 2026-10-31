# Prédiction de prix de vente des logements : Ridge, Lasso et Elastic Net

Projet R du cours d'Analyse Prédictive (Master 2 Data Science & Systèmes d'Information).
Méthode : régression Ridge / Lasso. Données : **Ames Housing**, 2 930 ventes réelles de
logements à Ames (Iowa) entre 2006 et 2010, prix en dollars US.

## Données

- **Source d'origine** : Dean De Cock, *Ames, Iowa: Alternative to the Boston Housing Data as an
  End of Semester Regression Project*, Journal of Statistics Education, 19(3), 2011.
  Article : <https://jse.amstat.org/v19n3/decock.pdf>
- **Fichier utilisé** : `housing/data/AmesHousing.csv`, 2 930 lignes, 82 colonnes
  (79 variables explicatives, la cible `saleprice`, et deux identifiants `order` et `pid`).
- **Lien de téléchargement et licence du fichier** : [URL EXACTE À COMPLÉTER],
  licence : [LICENCE AFFICHÉE SUR CETTE PAGE À COMPLÉTER]. Fichier téléchargé le [DATE].
  Les noms de colonnes sont en minuscules avec « _ » (ex. `gr_liv_area`), ce qui diffère du
  fichier Excel original de De Cock (`Gr Liv Area`) : ce nettoyage vient de la source ci-dessus.
- **Dictionnaire des 82 variables** : `housing/docs/dictionnaire_variables.csv`.
- **Cible** : `saleprice` (quantitative). Voir le rapport pour les valeurs manquantes
  et les cinq ventes atypiques (surface de plus de 4 000 pi²).

## Contenu (dossier `housing/`)

| Fichier | Rôle |
|---|---|
| `housing.Rproj` | Ouvre le projet dans RStudio (fixe le dossier de travail) |
| `main.R` | Lance toute l'analyse, dans l'ordre |
| `R/01_packages.R` | Charge les packages, fixe la graine |
| `R/functions.R` | Fonctions partagées avec l'application : `prepare_ames()` (avec les options `train_max_area` et `test_max_area` pour écarter les ventes atypiques), `make_new_x()`, `metrics_reg()`, `fit_models()` |
| `R/02_exploration.R` | Graphiques, paires de variables corrélées, asymétrie du prix |
| `R/03_preprocessing.R` | Prétraitement et découpage 80 % / 20 % |
| `R/04_ridge.R`, `R/05_lasso.R`, `R/05b_elasticnet.R` | Modèles pénalisés, validation croisée à 10 blocs |
| `R/06_comparison.R` | Comparaison MCO, Ridge, Lasso, Elastic Net |
| `R/07_prediction.R` | Prédiction pour de nouvelles maisons |
| `R/08_variantes.R` | Prix brut ou log(prix), avec ou sans les 5 ventes atypiques dans l'apprentissage ; toutes les variantes sont évaluées sur le même jeu de test |
| `R/09_robustesse.R` | Évaluation sur 10 découpages aléatoires |
| `R/10_showcase.R` | **Où Ridge et Lasso font la différence** : courbe d'apprentissage et ajout de 200 colonnes de bruit |
| `report/rapport_ridge_lasso.Rmd` | **Rapport** (points 4.1 à 4.6 du guide), compilé en HTML |
| `app/app.R` | **Application Shiny** (4 onglets) |
| `docs/` | Dictionnaire des variables |
| `results/` | Graphiques et tableaux produits par `main.R` |

L'analyse, le rapport et l'application utilisent le même prétraitement (`R/functions.R`).

## Installation (une fois, dans la console R)

```r
install.packages(c("shiny", "bslib", "plotly", "DT", "glmnet", "caret",
                   "tidyverse", "corrplot", "Metrics", "scales",
                   "rmarkdown", "knitr", "gridExtra"))
```

## Reproduire les résultats

1. Ouvrir `housing/housing.Rproj` dans RStudio.
2. Lancer `source("main.R")`. Les graphiques et tableaux arrivent dans `results/`
   (compter plusieurs minutes, surtout pour `R/09_robustesse.R` et `R/10_showcase.R`).
3. Compiler le rapport : ouvrir `report/rapport_ridge_lasso.Rmd` et cliquer sur **Knit**
   (ou `rmarkdown::render("report/rapport_ridge_lasso.Rmd")`). Le rapport relit les fichiers
   `results/variantes.csv`, `results/robustesse_decoupages.csv` et `results/10_*.csv` : lancer `main.R` d'abord.
   Les chiffres cités dans le texte du rapport sont calculés à la compilation.
4. Lancer l'application : `shiny::runApp("app")` depuis `housing/`, ou **Run App** dans `app/app.R`.

La graine est fixée à 123 (`set.seed(123)`) ; elle se règle aussi dans l'application.

## Résultats principaux (un découpage, graine 123, prix brut)

| Modèle | RMSE ($) | R² | Variables retenues |
|---|---|---|---|
| MCO | 23 455 | 0,912 | 286 |
| Ridge | 24 161 | 0,907 | 294 |
| Lasso | 23 041 | 0,915 | 122 |
| Elastic Net | 23 058 | 0,915 | 125 |

Ce tableau montre qu'**avec toutes les données, les modèles pénalisés sont proches des MCO** :
l'écart est de l'ordre de quelques centaines de dollars, comparable à la variation d'un découpage à l'autre.
Le Lasso obtient la meilleure erreur avec 122 colonnes non nulles au lieu de 286 pour les MCO.

Les autres résultats se trouvent dans `results/` et dans le rapport :

- **Dix découpages aléatoires** (log(prix), sans les cinq ventes atypiques) : `results/robustesse_decoupages.csv`
  (rapport, section 7.2) ;
- **Variantes** (prix brut ou log, avec ou sans les cinq ventes atypiques dans l'apprentissage) :
  `results/variantes.csv`. Toutes les variantes sont évaluées sur le **même jeu de test**, sans les cinq
  ventes atypiques : leurs RMSE ne sont donc pas comparables à ceux du tableau ci-dessus, dont le jeu de
  test les contient.

### Où Ridge et Lasso font la différence (`R/10_showcase.R`)

Sur l'échantillon complet (2 346 lignes d'apprentissage pour 294 variables non constantes), les MCO s'en
sortent encore bien. Deux expériences placent volontairement les modèles dans des situations plus difficiles,
avec le même jeu de test et 10 tirages chacune :

**A. Courbe d'apprentissage** : on réduit l'échantillon d'apprentissage (150, 300, 500, 1 000 lignes, puis
toutes). Avec moins de lignes que de variables, les MCO ne sont plus définis et leur erreur explose, alors que
Ridge et Lasso restent utilisables. Résultats : `results/10_courbe_apprentissage.csv` et `.png`.

**B. 200 colonnes de bruit pur** (nombres aléatoires sans lien avec le prix), RMSE médian sur le jeu de test :

| Lignes d'apprentissage | Modèle | Sans bruit ($) | Avec bruit ($) | Hausse |
|---|---|---|---|---|
| 500 | MCO | 36 719 | 65 702 | +78,9 % |
| 500 | Ridge | 27 454 | 42 707 | +55,6 % |
| 500 | Lasso | 25 980 | 26 128 | +0,6 % |
| 1 000 | MCO | 27 824 | 31 680 | +13,9 % |
| 1 000 | Ridge | 26 496 | 27 674 | +4,4 % |
| 1 000 | Lasso | 25 089 | 25 219 | +0,5 % |
| 2 346 | MCO | 23 455 | 24 541 | +4,6 % |
| 2 346 | Ridge | 24 227 | 25 242 | +4,2 % |
| 2 346 | Lasso | 23 085 | 23 283 | +0,9 % |

Sur les 200 colonnes de bruit, le Lasso en retient en médiane entre 20 et 25 avec `lambda.min`
(65 à 100 variables réelles) et **aucune** avec `lambda.1se` (21 à 27 variables réelles)
(`results/10_bruit_selection.csv`).

Lecture : la pénalisation ne rend pas le modèle meilleur quand les données sont abondantes. Elle le
rend **utilisable et stable** quand les données sont limitées par rapport au nombre de variables, et le Lasso
écarte presque toutes les variables inutiles. Précautions : le bruit est artificiel, le jeu de test est
unique, et les erreurs correspondent à `lambda.min` alors que « 0 colonne de bruit retenue » correspond à
`lambda.1se`. Les valeurs peuvent varier légèrement selon la version de R et de `glmnet`.

## Application Shiny

Quatre onglets, qui correspondent à ce que demande le guide (section 5.2) :

- **Données** : chargement optionnel d'un fichier CSV (voir ci-dessous), description du jeu de données et aperçu du tableau ;
- **Modèle** : choix de l'algorithme (Ridge, Lasso, Elastic Net), de α, de λ (λ min, λ 1se ou valeur manuelle)
  et du nombre de blocs de validation croisée ; RMSE, MAE, R², nombre de variables retenues ;
  courbe de validation croisée, chemins des coefficients, prix réels et prédits ;
- **Variables** : coefficients les plus influents et tableau des coefficients non nuls ;
- **Prédiction** : formulaire de saisie d'une maison et estimation de son prix.

Le découpage est fixe (80 % apprentissage, 20 % test, graine 123), comme dans `main.R`, donc les
résultats de l'application se retrouvent à l'identique dans les scripts. Par défaut, l'application travaille sur le jeu
Ames Housing du dossier `data/`. L'onglet **Données** permet de charger un autre fichier CSV, à condition qu'il
ait la structure d'Ames Housing (colonnes `saleprice`, `gr_liv_area`, `year_built`, `overall_qual`,
`neighborhood` au minimum, noms en minuscules) ; un fichier invalide est refusé avec un message. Après un
chargement, cliquer sur le bouton d'entraînement de l'onglet Modèle. L'onglet Prédiction reste spécifique aux
variables d'Ames. La comparaison avec les MCO, les variantes, la robustesse sur dix découpages et la
vitrine (courbe d'apprentissage, variables de bruit) se trouvent dans les scripts `R/06` à `R/10` et dans le rapport.
