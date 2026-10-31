# ============================================================
# 05 - Régression Lasso
# alpha = 1
# ============================================================

cat("\n===== REGRESSION LASSO =====\n")

set.seed(123)

# ------------------------------------------------------------
# Validation croisée
# ------------------------------------------------------------

cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 10,
  standardize = TRUE
)

lambda_lasso_min <- cv_lasso$lambda.min
lambda_lasso_1se <- cv_lasso$lambda.1se

cat(
  "Lambda optimal Lasso :",
  lambda_lasso_min,
  "\n"
)

cat(
  "Lambda Lasso 1SE :",
  lambda_lasso_1se,
  "\n"
)

# ------------------------------------------------------------
# Graphique CV
# ------------------------------------------------------------

png(
  "results/cv_lasso.png",
  width = 900,
  height = 650
)

plot(cv_lasso)

title("Validation croisée - Lasso")

dev.off()

# ------------------------------------------------------------
# Modèle final
# ------------------------------------------------------------

lasso_model <- glmnet(
  x_train,
  y_train,
  alpha = 1,
  lambda = lambda_lasso_min,
  standardize = TRUE
)

# ------------------------------------------------------------
# Prédiction
# ------------------------------------------------------------

pred_lasso <- predict(
  lasso_model,
  s = lambda_lasso_min,
  newx = x_test
)

pred_lasso <- as.numeric(pred_lasso)

# ------------------------------------------------------------
# Métriques
# ------------------------------------------------------------

rmse_lasso <- RMSE(
  pred_lasso,
  y_test
)

mae_lasso <- MAE(
  pred_lasso,
  y_test
)

r2_lasso <- r2_trad(
  pred_lasso,
  y_test
)

cat("\nPerformance Lasso :\n")
cat("RMSE :", round(rmse_lasso, 2), "\n")
cat("MAE  :", round(mae_lasso, 2), "\n")
cat("R²   :", round(r2_lasso, 4), "\n")

# ------------------------------------------------------------
# Coefficients
# ------------------------------------------------------------

coef_lasso <- coef(
  lasso_model,
  s = lambda_lasso_min
)

coef_lasso_df <- data.frame(
  variable = rownames(coef_lasso),
  coefficient = as.numeric(coef_lasso)
)

coef_lasso_df <- coef_lasso_df %>%
  filter(variable != "(Intercept)")

# Variables réellement sélectionnées
selected_lasso <- coef_lasso_df %>%
  filter(coefficient != 0) %>%
  arrange(desc(abs(coefficient)))

cat(
  "\nNombre de variables sélectionnées par Lasso :",
  nrow(selected_lasso),
  "\n"
)

cat("\nVariables principales :\n")

print(
  head(selected_lasso, 20)
)

# ------------------------------------------------------------
# Chemin des coefficients
# ------------------------------------------------------------

lasso_path <- glmnet(
  x_train,
  y_train,
  alpha = 1
)

png(
  "results/lasso_coefficients.png",
  width = 900,
  height = 650
)

plot(
  lasso_path,
  xvar = "lambda",
  label = FALSE
)

title("Chemin des coefficients - Lasso")

dev.off()

# ------------------------------------------------------------
# Variante plus prudente : lambda.1se (modèle plus régularisé)
# On garde lambda.min pour la comparaison principale ; lambda.1se sert de
# point de comparaison : erreur un peu plus grande, modèle plus simple.
# ------------------------------------------------------------

pred_lasso_1se <- as.numeric(
  predict(cv_lasso, newx = x_test, s = "lambda.1se")
)

met_lasso_1se <- metrics_reg(pred_lasso_1se, y_test)

cat("\nVariante lambda.1se (Lasso) :\n")
cat("RMSE :", round(met_lasso_1se["RMSE"], 2),
    "| R² :", round(met_lasso_1se["R2"], 4),
    "| variables non nulles :",
    sum(as.numeric(coef(cv_lasso, s = "lambda.1se"))[-1] != 0), "\n")
