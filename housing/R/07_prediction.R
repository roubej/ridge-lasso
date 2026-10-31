# ============================================================
# 07 - Prédiction sur de nouvelles observations
# ============================================================
# make_new_x() (R/functions.R) construit la ligne de variables d'une maison
# décrite par quelques caractéristiques ; toutes les autres variables
# prennent une valeur typique (médiane ou modalité la plus fréquente).
# C'est la même fonction que celle du formulaire de l'application Shiny.
# ============================================================

cat("\n===== PREDICTION SUR DE NOUVELLES MAISONS =====\n")

nouvelles_maisons <- list(
  "Maison moyenne (1995, 140 m2)" = list(
    overall_qual = 6, gr_liv_area = 1500, year_built = 1995,
    total_bsmt_sf = 1000, garage_cars = 2, full_bath = 2, bedroom_abvgr = 3
  ),
  "Petite maison ancienne (1950, 85 m2)" = list(
    overall_qual = 5, gr_liv_area = 950, year_built = 1950,
    total_bsmt_sf = 700, garage_cars = 1, full_bath = 1, bedroom_abvgr = 2
  ),
  "Grande maison récente (2005, 250 m2)" = list(
    overall_qual = 8, gr_liv_area = 2700, year_built = 2005,
    total_bsmt_sf = 1400, garage_cars = 3, full_bath = 3, bedroom_abvgr = 4
  )
)

predictions_new <- do.call(rbind, lapply(names(nouvelles_maisons), function(nm) {
  x_new <- make_new_x(prep, nouvelles_maisons[[nm]])
  data.frame(
    Maison = nm,
    Ridge = round(as.numeric(predict(ridge_model, newx = x_new, s = lambda_ridge_min))),
    Lasso = round(as.numeric(predict(lasso_model, newx = x_new, s = lambda_lasso_min)))
  )
}))

print(predictions_new, row.names = FALSE)
write.csv(predictions_new, "results/predictions_nouvelles_maisons.csv", row.names = FALSE)

# Contrôle : une maison du jeu de test, dont on connaît le vrai prix
new_house <- x_test[1, , drop = FALSE]
cat(
  "\nContrôle sur une maison du jeu de test : prix réel",
  round(y_test[1]), "$, Ridge",
  round(as.numeric(predict(ridge_model, newx = new_house, s = lambda_ridge_min))),
  "$, Lasso",
  round(as.numeric(predict(lasso_model, newx = new_house, s = lambda_lasso_min))), "$\n"
)
