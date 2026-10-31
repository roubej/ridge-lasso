# ============================================================
# 09 - Robustesse : 10 découpages train/test différents
# ============================================================
# Avec un seul découpage, une différence de 500 $ de RMSE entre deux modèles
# ne veut presque rien dire. On répète donc l'évaluation sur 10 graines
# et on regarde la moyenne et l'écart-type.
# Configuration : log(prix), sans les 5 maisons atypiques.
# ============================================================

cat("\n===== ROBUSTESSE (10 DECOUPAGES) =====\n")

d_clean <- data[data$gr_liv_area <= 4000, ]
modeles_garde <- c("Régression classique (MCO)", "Ridge", "Lasso", "Elastic Net")

res_rob <- do.call(rbind, lapply(1:10, function(sd_) {
  p <- prepare_ames(d_clean, p = 0.80, seed = sd_)
  r <- fit_models(p, uselog = TRUE, seed = sd_)
  r <- r[r$Modele %in% modeles_garde, ]
  cbind(Decoupage = sd_, r)
}))

write.csv(res_rob, "results/robustesse_decoupages.csv", row.names = FALSE)

resume_rob <- res_rob %>%
  group_by(Modele) %>%
  summarise(
    RMSE_moyen = round(mean(RMSE)),
    RMSE_ecart_type = round(sd(RMSE)),
    R2_moyen = round(mean(R2), 4),
    Variables_moyennes = round(mean(Variables)),
    .groups = "drop"
  ) %>%
  arrange(RMSE_moyen)

print(as.data.frame(resume_rob), row.names = FALSE)
write.csv(resume_rob, "results/robustesse_resume.csv", row.names = FALSE)

# Pour chaque découpage, combien de fois chaque modèle est-il le meilleur ?
gagnants <- res_rob %>%
  group_by(Decoupage) %>%
  slice_min(RMSE, n = 1) %>%
  ungroup() %>%
  count(Modele, name = "fois_meilleur")
cat("\nNombre de découpages où le modèle a le plus petit RMSE :\n")
print(as.data.frame(gagnants), row.names = FALSE)

p_rob <- ggplot(res_rob, aes(x = reorder(Modele, RMSE), y = RMSE, fill = Modele)) +
  geom_boxplot(alpha = 0.8) +
  labs(
    title = "RMSE sur 10 découpages (log(prix), sans atypiques)",
    x = "", y = "RMSE ($)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("results/robustesse_rmse.png", p_rob, width = 8, height = 5)
