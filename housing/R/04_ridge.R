# ============================================================
# 04 - Régression Ridge
# alpha = 0
# ============================================================

cat("\n===== REGRESSION RIDGE =====\n")

set.seed(123)

# ------------------------------------------------------------
# Validation croisée pour trouver lambda
# ------------------------------------------------------------

cv_ridge <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 0,
  nfolds = 10,
  standardize = TRUE
)

# Lambda minimisant l'erreur
lambda_ridge_min <- cv_ridge$lambda.min

# Lambda plus régularisé
lambda_ridge_1se <- cv_ridge$lambda.1se

cat(
  "Lambda optimal Ridge (lambda.min) :",
  lambda_ridge_min,
  "\n"
)

cat(
  "Lambda Ridge (lambda.1se) :",
  lambda_ridge_1se,
  "\n"
)

# ------------------------------------------------------------
# Graphique validation croisée
# ------------------------------------------------------------

# RMSE en dollars (plot_cv_rmse est définie dans R/functions.R)
ggplot2::ggsave(
  "results/cv_ridge.png",
  plot_cv_rmse(cv_ridge, "Validation croisée - Ridge"),
  width = 8, height = 5.5, dpi = 120
)

# ------------------------------------------------------------
# Modèle final
# ------------------------------------------------------------

ridge_model <- glmnet(
  x_train,
  y_train,
  alpha = 0,
  lambda = lambda_ridge_min,
  standardize = TRUE
)

# ------------------------------------------------------------
# Prédiction
# ------------------------------------------------------------

pred_ridge <- predict(
  ridge_model,
  s = lambda_ridge_min,
  newx = x_test
)

pred_ridge <- as.numeric(pred_ridge)

# ------------------------------------------------------------
# Métriques
# ------------------------------------------------------------

rmse_ridge <- RMSE(
  pred_ridge,
  y_test
)

mae_ridge <- MAE(
  pred_ridge,
  y_test
)

r2_ridge <- r2_trad(
  pred_ridge,
  y_test
)

cat("\nPerformance Ridge :\n")
cat("RMSE :", round(rmse_ridge, 2), "\n")
cat("MAE  :", round(mae_ridge, 2), "\n")
cat("R²   :", round(r2_ridge, 4), "\n")

# ------------------------------------------------------------
# Coefficients Ridge
# ------------------------------------------------------------

coef_ridge <- coef(
  ridge_model,
  s = lambda_ridge_min
)

coef_ridge_df <- data.frame(
  variable = rownames(coef_ridge),
  coefficient = as.numeric(coef_ridge)
)

coef_ridge_df <- coef_ridge_df %>%
  filter(variable != "(Intercept)") %>%
  arrange(desc(abs(coefficient)))

cat("\nPrincipaux coefficients Ridge :\n")

print(
  head(coef_ridge_df, 15)
)

# ------------------------------------------------------------
# Chemin des coefficients
# ------------------------------------------------------------

ridge_path <- glmnet(
  x_train,
  y_train,
  alpha = 0
)

png(
  "results/ridge_coefficients.png",
  width = 900,
  height = 650
)

plot(
  ridge_path,
  xvar = "lambda",
  label = FALSE
)

title("Chemin des coefficients - Ridge")

dev.off()

# ------------------------------------------------------------
# Variante plus prudente : lambda.1se (modèle plus régularisé)
# On garde lambda.min pour la comparaison principale ; lambda.1se sert de
# point de comparaison : erreur un peu plus grande, modèle plus simple.
# ------------------------------------------------------------

pred_ridge_1se <- as.numeric(
  predict(cv_ridge, newx = x_test, s = "lambda.1se")
)

met_ridge_1se <- metrics_reg(pred_ridge_1se, y_test)

cat("\nVariante lambda.1se (Ridge) :\n")
cat("RMSE :", round(met_ridge_1se["RMSE"], 2),
    "| R² :", round(met_ridge_1se["R2"], 4),
    "| variables non nulles :",
    sum(as.numeric(coef(cv_ridge, s = "lambda.1se"))[-1] != 0), "\n")
