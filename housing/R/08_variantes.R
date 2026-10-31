# ============================================================
# 08 - Variantes de préparation : log(prix) et ventes atypiques
# ============================================================
# On refait la même comparaison (MCO, Ridge, Lasso, Elastic Net) dans quatre
# configurations. Les métriques sont toujours exprimées en dollars.
#   - cible : prix brut ou log(prix)
#   - données : toutes les ventes, ou sans les 5 maisons de plus de 4 000 pi²
# ============================================================

cat("\n===== VARIANTES : LOG(PRIX) ET VENTES ATYPIQUES =====\n")

variantes <- list(
  list(nom = "Prix brut, toutes les ventes",      uselog = FALSE, exclure = FALSE),
  list(nom = "log(prix), toutes les ventes",      uselog = TRUE,  exclure = FALSE),
  list(nom = "Prix brut, sans atypiques",         uselog = FALSE, exclure = TRUE),
  list(nom = "log(prix), sans atypiques",         uselog = TRUE,  exclure = TRUE)
)

res_variantes <- do.call(rbind, lapply(variantes, function(v) {
  d <- if (v$exclure) data[data$gr_liv_area <= 4000, ] else data
  p <- prepare_ames(d, p = 0.80, seed = 123)
  r <- fit_models(p, uselog = v$uselog, seed = 123)
  cbind(Variante = v$nom, r)
}))

res_variantes$Lambda <- signif(res_variantes$Lambda, 4)
res_variantes$RMSE <- round(res_variantes$RMSE, 1)
res_variantes$MAE <- round(res_variantes$MAE, 1)
res_variantes$R2 <- round(res_variantes$R2, 4)

print(res_variantes[res_variantes$Modele %in%
        c("Régression classique (MCO)", "Ridge", "Lasso", "Elastic Net"),
      c("Variante", "Modele", "RMSE", "MAE", "R2", "Variables")],
      row.names = FALSE)

write.csv(res_variantes, "results/variantes.csv", row.names = FALSE)
