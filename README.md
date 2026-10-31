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
- **Lien de téléchargement et licence du fichier** : à renseigner par le binôme (adresse
  de la page Kaggle ou OpenML d'où le fichier a été récupéré, et licence indiquée sur cette page).
- **Dictionnaire des 82 variables** : `housing/docs/dictionnaire_variables.csv`.
- **Cible** : `saleprice` (quantitative). Voir le rapport pour les valeurs manquantes
  et les cinq ventes atypiques (surface de plus de 4 000 pi²).

## Contenu (dossier `housing/`)

| Fichier | Rôle |
|---|---|
| `housing.Rproj` | Ouvre le projet dans RStudio (fixe le dossier de travail) |
| `main.R` | Lance toute l'analyse, dans l'ordre |
| `R/01_packages.R` | Charge les packages, fixe la graine |
| `R/functions.R` | Fonctions partagées avec l'application : `prepare_ames()`, `make_new_x()`, `metrics_reg()`, `fit_models()` |
| `R/02_exploration.R` | Graphiques, paires de variables corrélées, asymétrie du prix |
| `R/03_preprocessing.R` | Prétraitement et découpage 80 % / 20 % |
| `R/04_ridge.R`, `R/05_lasso.R`, `R/05b_elasticnet.R` | Modèles pénalisés, validation croisée à 10 blocs |
| `R/06_comparison.R` | Comparaison MCO, Ridge, Lasso, Elastic Net |
| `R/07_prediction.R` | Prédiction pour de nouvelles maisons |
| `R/08_variantes.R` | Prix brut ou log(prix), avec ou sans les 5 ventes atypiques |
| `R/09_robustesse.R` | Évaluation sur 10 découpages aléatoires |
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
   (compter quelques minutes, surtout pour `R/09_robustesse.R`).
3. Compiler le rapport : ouvrir `report/rapport_ridge_lasso.Rmd` et cliquer sur **Knit**
   (ou `rmarkdown::render("report/rapport_ridge_lasso.Rmd")`). Le rapport relit les fichiers
   `results/variantes.csv` et `results/robustesse_decoupages.csv` : lancer `main.R` d'abord.
4. Lancer l'application : `shiny::runApp("app")` depuis `housing/`, ou **Run App** dans `app/app.R`.

La graine est fixée à 123 (`set.seed(123)`) ; elle se règle aussi dans l'application.

## Résultats principaux (un découpage, graine 123, prix brut)

| Modèle | RMSE ($) | R² | Variables retenues |
|---|---|---|---|
| MCO | 23 455 | 0,912 | 286 |
| Ridge | 24 161 | 0,907 | 294 |
| Lasso | 23 041 | 0,915 | 122 |
| Elastic Net | 23 058 | 0,915 | 125 |

Sur dix découpages (log(prix), sans les cinq ventes atypiques), les modèles pénalisés
restent entre 17 900 $ et 20 900 $ de RMSE, alors que les MCO montent jusqu'à 36 600 $
sur certains découpages. Le détail est dans `results/robustesse_decoupages.csv`
et dans le rapport.

## Application Shiny

Quatre onglets, qui correspondent à ce que demande le guide (section 5.2) :

- **Données** : description du jeu de données et aperçu du tableau ;
- **Modèle** : choix de l'algorithme (Ridge, Lasso, Elastic Net), de α, de λ (λ min, λ 1se ou valeur manuelle)
  et du nombre de blocs de validation croisée ; RMSE, MAE, R², nombre de variables retenues ;
  courbe de validation croisée, chemins des coefficients, prix réels et prédits ;
- **Variables** : coefficients les plus influents et tableau des coefficients non nuls ;
- **Prédiction** : formulaire de saisie d'une maison et estimation de son prix.

Le découpage est fixe (80 % apprentissage, 20 % test, graine 123), comme dans `main.R`, donc les
résultats de l'application se retrouvent à l'identique dans les scripts. L'application travaille sur le jeu
Ames Housing du dossier `data/` ; elle ne propose pas le chargement d'un autre fichier. La comparaison
avec les MCO, les variantes (log(prix), ventes atypiques) et la robustesse sur dix découpages se trouvent
dans les scripts `R/06` à `R/09` et dans le rapport.
