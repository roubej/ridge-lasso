# ============================================================
# 06 - Comparaison : MCO, Ridge, Lasso, Elastic Net
# ============================================================

cat("\n===== COMPARAISON DES MODELES =====\n")

# ------------------------------------------------------------
# Régression classique (MCO) : modèle de référence
# Sans pénalisation, sur les mêmes variables. Plusieurs colonnes sont
# colinéaires (lm les abandonne), d'où l'avertissement masqué.
# ------------------------------------------------------------

ols_model <- suppressWarnings(
  lm(y_train ~ ., data = as.data.frame(x_train))
)
pred_ols <- suppressWarnings(
  as.numeric(predict(ols_model, as.data.frame(x_test)))
)
met_ols <- metrics_reg(pred_ols, y_test)

n_ols <- sum(!is.na(coef(ols_model))[-1])

# ------------------------------------------------------------
# Tableau des performances (jeu de test, lambda.min)
# ------------------------------------------------------------

comparison <- data.frame(
  Modele = c("MCO", "Ridge", "Lasso", "Elastic Net"),
  Lambda = c(NA, lambda_ridge_min, lambda_lasso_min, lambda_enet_min),
  RMSE   = c(met_ols["RMSE"], rmse_ridge, rmse_lasso, rmse_enet),
  MAE    = c(met_ols["MAE"],  mae_ridge,  mae_lasso,  mae_enet),
  R2     = c(met_ols["R2"],   r2_ridge,   r2_lasso,   r2_enet),
  Variables = c(
    n_ols,
    sum(as.numeric(coef(ridge_model, s = lambda_ridge_min))[-1] != 0),
    nrow(selected_lasso),
    n_enet
  )
) %>%
  mutate(
    Lambda = round(Lambda, 4),
    RMSE = round(RMSE, 2),
    MAE = round(MAE, 2),
    R2 = round(R2, 4)
  )

rownames(comparison) <- NULL
print(comparison)

write.csv(comparison, "results/comparison_models.csv", row.names = FALSE)

# ------------------------------------------------------------
# Comparaison RMSE
# ------------------------------------------------------------

p_rmse <- ggplot(
  comparison,
  aes(x = reorder(Modele, RMSE), y = RMSE, fill = Modele)
) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(RMSE)), vjust = -0.5) +
  labs(
    title = "Comparaison du RMSE (jeu de test)",
    x = "",
    y = "RMSE ($)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p_rmse)

ggsave("results/comparison_rmse.png", p_rmse, width = 7, height = 5)

# ------------------------------------------------------------
# Prix réel vs prédit
# ------------------------------------------------------------

results_test <- data.frame(
  Prix_reel = y_test,
  Ridge = pred_ridge,
  Lasso = pred_lasso
)

make_scatter <- function(col, titre) {
  ggplot(results_test, aes(x = Prix_reel, y = .data[[col]])) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = titre, x = "Prix réel", y = "Prix prédit") +
    theme_minimal()
}

ggsave("results/ridge_real_vs_predicted.png",
       make_scatter("Ridge", "Ridge : prix réels vs prix prédits"),
       width = 7, height = 5)

ggsave("results/lasso_real_vs_predicted.png",
       make_scatter("Lasso", "Lasso : prix réels vs prix prédits"),
       width = 7, height = 5)

# ------------------------------------------------------------
# Lecture du tableau
# ------------------------------------------------------------

best <- comparison$Modele[which.min(comparison$RMSE)]
cat("\nMeilleur RMSE sur ce découpage :", best, "\n")

cat(
  "\nLasso a sélectionné", nrow(selected_lasso),
  "variables sur", ncol(x_train), "après encodage.\n"
)

cat(
  "\nAttention : les écarts de RMSE entre modèles sont petits devant la",
  "variabilité d'un découpage à l'autre.\nVoir R/09_robustesse.R",
  "(10 découpages) avant de conclure.\n"
)
