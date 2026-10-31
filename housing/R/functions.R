# ============================================================
# functions.R - Fonctions partagées (analyse ET application Shiny)
# ============================================================
# Ce fichier est appelé par :
#   - R/03_preprocessing.R  (analyse complète lancée par main.R)
#   - app/app.R             (application Shiny)
# Ainsi, les deux utilisent EXACTEMENT le même prétraitement.
# Nécessite le package caret (createDataPartition, dummyVars).
# ============================================================

# ------------------------------------------------------------
# prepare_ames() : prétraitement complet
#   data : data.frame brut (AmesHousing.csv)
#   p    : proportion du jeu d'apprentissage (ex. 0.80)
#   seed : graine aléatoire (reproductibilité)
# Retourne une liste avec les matrices x_train / x_test, les cibles
# y_train / y_test et les objets utiles pour prédire de nouvelles maisons.
# ------------------------------------------------------------
prepare_ames <- function(data, p = 0.80, seed = 123) {

  # 1. Suppression des identifiants
  d <- data[, !(names(data) %in% c("order", "pid")), drop = FALSE]

  # 2. Chaînes de caractères -> facteurs
  for (col in names(d)) {
    if (is.character(d[[col]])) d[[col]] <- as.factor(d[[col]])
  }
  # ms_subclass (type de logement) et mo_sold (mois de vente) sont des codes
  # nominaux : 20, 60, 120... n'ont pas d'ordre numérique. On les traite
  # comme des catégories et non comme des quantités.
  for (col in c("ms_subclass", "mo_sold")) {
    if (col %in% names(d)) d[[col]] <- as.factor(d[[col]])
  }

  # 3. Séparation apprentissage / test
  set.seed(seed)
  idx   <- caret::createDataPartition(d$saleprice, p = p, list = FALSE)
  train <- d[idx, , drop = FALSE]
  test  <- d[-idx, , drop = FALSE]

  # 4. Cible et variables explicatives
  y_train <- train$saleprice
  y_test  <- test$saleprice
  x_train_raw <- train[, names(train) != "saleprice", drop = FALSE]
  x_test_raw  <- test[,  names(test)  != "saleprice", drop = FALSE]

  # 5a. Variables numériques : NA -> médiane calculée sur le TRAIN
  num_cols <- names(x_train_raw)[sapply(x_train_raw, is.numeric)]
  medians  <- sapply(num_cols, function(col) {
    m <- median(x_train_raw[[col]], na.rm = TRUE)
    if (is.na(m)) 0 else m
  })

  # 5b. Variables catégorielles : NA -> "Missing", modalité inconnue -> "Other"
  fac_cols    <- names(x_train_raw)[sapply(x_train_raw, is.factor)]
  levels_list <- list()
  for (col in fac_cols) {
    v <- as.character(x_train_raw[[col]])
    v[is.na(v)] <- "Missing"
    levels_list[[col]] <- unique(c(unique(v), "Other"))
  }

  clean <- function(df) {
    # Année du garage : NA = pas de garage (la variable garage_type vaut alors
    # "Missing"), et une valeur aberrante (2207) existe dans les données.
    # On prend l'année de construction de la maison plutôt qu'une médiane.
    if (all(c("garage_yr_blt", "year_built") %in% names(df))) {
      bad <- is.na(df$garage_yr_blt) | df$garage_yr_blt > 2010
      df$garage_yr_blt[bad] <- df$year_built[bad]
    }
    for (col in num_cols) df[[col]][is.na(df[[col]])] <- medians[[col]]
    for (col in fac_cols) {
      v <- as.character(df[[col]])
      v[is.na(v)] <- "Missing"
      v[!(v %in% levels_list[[col]])] <- "Other"
      df[[col]] <- factor(v, levels = levels_list[[col]])
    }
    df
  }
  x_train_raw <- clean(x_train_raw)
  x_test_raw  <- clean(x_test_raw)

  # 6. Encodage des variables catégorielles (variables indicatrices)
  dummy   <- caret::dummyVars(~ ., data = x_train_raw, fullRank = TRUE)
  x_train <- as.matrix(predict(dummy, newdata = x_train_raw))
  x_test  <- as.matrix(predict(dummy, newdata = x_test_raw))

  # Valeur la plus fréquente de chaque variable catégorielle (pour prédire)
  modes <- sapply(fac_cols, function(col) {
    names(which.max(table(x_train_raw[[col]])))
  })

  list(x_train = x_train, x_test = x_test,
       y_train = y_train, y_test = y_test,
       x_train_raw = x_train_raw, x_test_raw = x_test_raw,
       dummy = dummy, medians = medians, modes = modes,
       train_data = train, test_data = test)
}

# ------------------------------------------------------------
# make_new_x() : construit la matrice d'UNE nouvelle maison
#   overrides : liste nommée des caractéristiques saisies
#               (ex. list(overall_qual = 7, gr_liv_area = 1500))
# Toutes les autres variables prennent une valeur "typique"
# (médiane pour les numériques, modalité la plus fréquente sinon).
# Quelques règles de cohérence évitent des maisons impossibles.
# ------------------------------------------------------------
make_new_x <- function(prep, overrides = list()) {
  ref <- prep$x_train_raw[1, , drop = FALSE]
  for (col in names(prep$medians)) ref[[col]] <- prep$medians[[col]]
  for (col in names(prep$modes)) {
    ref[[col]] <- factor(prep$modes[[col]], levels = levels(prep$x_train_raw[[col]]))
  }

  set_val <- function(col, value) {
    if (!(col %in% names(ref))) return(invisible(NULL))
    if (col %in% names(prep$modes)) {
      v <- as.character(value)
      if (!(v %in% levels(ref[[col]]))) v <- "Other"
      ref[[col]] <<- factor(v, levels = levels(ref[[col]]))
    } else {
      ref[[col]] <<- as.numeric(value)
    }
  }
  for (nm in names(overrides)) set_val(nm, overrides[[nm]])

  # --- Règles de cohérence (variables dérivées des saisies) ---
  ov <- overrides
  if (!is.null(ov$year_built)) {
    set_val("year_remod_add", ov$year_built)   # pas de rénovation avant la construction
    set_val("garage_yr_blt",  ov$year_built)   # garage construit avec la maison
  }
  if (!is.null(ov$garage_cars)) set_val("garage_area", 250 * ov$garage_cars)
  if (!is.null(ov$gr_liv_area)) {
    sous_sol <- if (is.null(ov$total_bsmt_sf)) 0 else ov$total_bsmt_sf
    etage1 <- if (sous_sol > 0) min(sous_sol, ov$gr_liv_area) else ov$gr_liv_area
    set_val("X1st_flr_sf", etage1)
    set_val("X2nd_flr_sf", ov$gr_liv_area - etage1)
  }
  if (!is.null(ov$bedroom_abvgr)) set_val("totrms_abvgrd", ov$bedroom_abvgr + 3)

  as.matrix(predict(prep$dummy, newdata = ref))
}

# ------------------------------------------------------------
# Métriques (mêmes formules que dans 04_ridge.R / 05_lasso.R)
# ------------------------------------------------------------
# R2 "classique" : 1 - SCR/SCT. (caret::R2 calcule par défaut le carré de la
# corrélation, ce qui donne un R2 un peu plus flatteur sur un jeu de test.)
r2_trad <- function(pred, obs) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

metrics_reg <- function(pred, obs) {
  c(RMSE = caret::RMSE(pred, obs),
    MAE  = caret::MAE(pred, obs),
    R2   = r2_trad(pred, obs))
}

# ------------------------------------------------------------
# fit_models() : OLS + Ridge + Lasso + Elastic Net sur un jeu prétraité
#   prep    : résultat de prepare_ames()
#   uselog  : TRUE = on modélise log(prix) (les métriques restent en dollars)
#   seed    : graine de la validation croisée
# Retourne un data.frame de métriques (jeu de test) avec lambda.min.
# ------------------------------------------------------------
fit_models <- function(prep, uselog = FALSE, seed = 123, nfolds = 10) {
  yt <- if (uselog) log(prep$y_train) else prep$y_train
  back <- function(p) if (uselog) exp(p) else p

  ols <- suppressWarnings(lm(yt ~ ., data = as.data.frame(prep$x_train)))
  po  <- back(suppressWarnings(as.numeric(predict(ols, as.data.frame(prep$x_test)))))
  res <- list(data.frame(Modele = "Régression classique (MCO)", Alpha = NA, Lambda = NA,
                         t(metrics_reg(po, prep$y_test)),
                         Variables = sum(!is.na(coef(ols))[-1])))

  for (spec in list(c("Ridge", 0), c("Lasso", 1), c("Elastic Net", 0.5))) {
    a <- as.numeric(spec[2])
    set.seed(seed)
    cv <- cv.glmnet(prep$x_train, yt, alpha = a, nfolds = nfolds)
    for (rule in c("lambda.min", "lambda.1se")) {
      lam <- cv[[rule]]
      pr  <- back(as.numeric(predict(cv, newx = prep$x_test, s = lam)))
      nm  <- if (rule == "lambda.min") spec[1] else paste0(spec[1], " (1 ET)")
      res[[length(res) + 1]] <- data.frame(
        Modele = nm, Alpha = a, Lambda = lam, t(metrics_reg(pr, prep$y_test)),
        Variables = sum(as.numeric(coef(cv, s = lam))[-1] != 0))
    }
  }
  out <- do.call(rbind, res)
  rownames(out) <- NULL
  out
}
