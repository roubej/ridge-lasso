# ============================================================
# PROJET ANALYSE PREDICTIVE
# Régression Ridge et Lasso
# Dataset : Ames Housing
# ============================================================

# Nettoyage de l'environnement
rm(list = ls())

# Pas de notation scientifique sur les axes (-600000 au lieu de -6e+05)
options(scipen = 999)

# Création du dossier résultats si nécessaire
if(!dir.exists("results")) {
  dir.create("results")
}

# ------------------------------------------------------------
# Packages
# ------------------------------------------------------------

source("R/01_packages.R")

# ------------------------------------------------------------
# Chargement des données
# ------------------------------------------------------------

data <- read.csv(
  "data/AmesHousing.csv",
  stringsAsFactors = FALSE
)

cat("\n========================================\n")
cat("      AMES HOUSING - RIDGE / LASSO\n")
cat("========================================\n")

cat("\nDimensions :", dim(data), "\n")

# ------------------------------------------------------------
# Exploration
# ------------------------------------------------------------

source("R/02_exploration.R")

# ------------------------------------------------------------
# Prétraitement
# ------------------------------------------------------------

source("R/03_preprocessing.R")

# ------------------------------------------------------------
# Ridge
# ------------------------------------------------------------

source("R/04_ridge.R")

# ------------------------------------------------------------
# Lasso
# ------------------------------------------------------------

source("R/05_lasso.R")

# ------------------------------------------------------------
# Elastic Net
# ------------------------------------------------------------

source("R/05b_elasticnet.R")

# ------------------------------------------------------------
# Comparaison (MCO, Ridge, Lasso, Elastic Net)
# ------------------------------------------------------------

source("R/06_comparison.R")

# ------------------------------------------------------------
# Exemple de prédiction
# ------------------------------------------------------------

source("R/07_prediction.R")

# ------------------------------------------------------------
# Variantes (log, ventes atypiques) et robustesse (10 découpages)
# ------------------------------------------------------------

source("R/08_variantes.R")
source("R/09_robustesse.R")

# ------------------------------------------------------------
# Vitrine : où Ridge et Lasso font la différence
# (courbe d'apprentissage, variables de bruit ; quelques minutes)
# ------------------------------------------------------------

source("R/10_showcase.R")

cat("\n========================================\n")
cat("          ANALYSE TERMINEE\n")
cat("========================================\n")

cat(
  "\nLes graphiques sont disponibles dans le dossier results/.\n"
)
