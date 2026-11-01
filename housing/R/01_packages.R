# ============================================================
# Packages utilisés dans le projet
# Régression Ridge et Lasso - Ames Housing
# ============================================================

packages <- c(
  "tidyverse",
  "glmnet",
  "caret",
  "corrplot",
  "Metrics",
  "scales",
  "dplyr"
)

# Installation des packages manquants
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]

if(length(new_packages) > 0) {
  install.packages(new_packages)
}

# Chargement
library(tidyverse)
library(glmnet)
library(caret)
library(corrplot)
library(Metrics)
library(scales)
library(dplyr)


# Pour avoir les mêmes résultats à chaque exécution
set.seed(123)
