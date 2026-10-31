# ============================================================
# 10 - VITRINE : OÙ RIDGE ET LASSO FONT VRAIMENT LA DIFFÉRENCE
# ============================================================
# Idée : sur Ames avec toutes les données (n = 2 344 pour 294 variables),
# les MCO s'en sortent déjà bien. Ridge et Lasso deviennent indispensables
# quand le modèle est « trop grand » pour les données. Deux expériences :
#
#   A. Courbe d'apprentissage : on réduit la taille de l'échantillon
#      d'apprentissage (150, 300, 500, 1000, tout) et on compare MCO, Ridge, Lasso
#      sur le MÊME jeu de test.
#   B. Variables inutiles : on ajoute 200 colonnes de bruit pur (sans aucun lien
#      avec le prix) et on regarde si les modèles s'en laissent distraire, pour
#      3 tailles d'apprentissage et 10 tirages de bruit. On compte aussi combien
#      de colonnes de bruit le Lasso retient à tort.
#
# Lancement : depuis le dossier housing/ (source("R/10_showcase.R")),
# ou via main.R. Sorties : results/10_*.csv et results/10_*.png
# ============================================================

for (pkg in c("glmnet", "ggplot2")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop("Package manquant : ", pkg, " -> install.packages(\"", pkg, "\")", call. = FALSE)
}
library(glmnet)
library(ggplot2)

if (!exists("prepare_ames")) source("R/functions.R")
# Attention : "data" existe toujours en R (c'est une fonction de base), donc on teste
# que c'est bien un data.frame et pas seulement que le nom existe.
if (!is.data.frame(get0("data", inherits = TRUE))) {
  data <- read.csv("data/AmesHousing.csv", stringsAsFactors = FALSE)
}
if (!dir.exists("results")) dir.create("results")

cat("\n===== 10. VITRINE : OU RIDGE ET LASSO FONT LA DIFFERENCE =====\n")

# Mêmes données et même découpage (graine 123) que le reste du projet
prep10 <- prepare_ames(data, p = 0.80, seed = 123)
x_tr <- prep10$x_train;  y_tr <- prep10$y_train
x_te <- prep10$x_test;   y_te <- prep10$y_test
# Colonnes réellement utilisables : on écarte celles qui sont constantes (ex. la modalité
# "Other" de chaque variable catégorielle, toujours à 0 dans l'apprentissage).
p_eff <- sum(apply(x_tr, 2, sd) > 0)
cat("Apprentissage :", nrow(x_tr), "lignes x", p_eff, "variables non constantes",
    "(", ncol(x_tr), "colonnes au total ) | test :", nrow(x_te), "lignes\n")

rmse <- function(a, b) sqrt(mean((a - b)^2))

# ------------------------------------------------------------
# fit3() : ajuste MCO, Ridge et Lasso, renvoie les RMSE sur le test
#   - MCO  : lm() sur toutes les colonnes. Quand elles sont trop nombreuses
#            ou colinéaires, lm() met NA aux coefficients impossibles à estimer
#            (comportement habituel de R) : c'est précisément la limite des MCO.
#   - Ridge / Lasso : lambda choisi par validation croisée 10 plis (lambda.min)
# ------------------------------------------------------------
fit3 <- function(xtr, ytr, xte, yte, seed = 1) {
  set.seed(seed)
  dtr <- as.data.frame(xtr);  names(dtr) <- paste0("v", seq_len(ncol(xtr)))
  dte <- as.data.frame(xte);  names(dte) <- names(dtr)
  ols   <- suppressWarnings(lm(ytr ~ ., data = dtr))
  p_ols <- suppressWarnings(predict(ols, newdata = dte))
  rid <- cv.glmnet(xtr, ytr, alpha = 0, nfolds = 10)
  las <- cv.glmnet(xtr, ytr, alpha = 1, nfolds = 10)
  list(rmse = c(MCO   = rmse(yte, p_ols),
                Ridge = rmse(yte, as.numeric(predict(rid, xte, s = "lambda.min"))),
                Lasso = rmse(yte, as.numeric(predict(las, xte, s = "lambda.min")))),
       las = las)
}

# ============================================================
# EXPERIENCE A : courbe d'apprentissage
# ============================================================
cat("\n--- A. Courbe d'apprentissage (10 répétitions par taille) ---\n")
tailles <- c(150, 300, 500, 1000, nrow(x_tr))
reps    <- 10

res_a <- do.call(rbind, lapply(tailles, function(n) {
  do.call(rbind, lapply(seq_len(reps), function(r) {
    set.seed(1000 + r)
    i <- if (n >= nrow(x_tr)) seq_len(nrow(x_tr)) else sample(nrow(x_tr), n)
    f <- fit3(x_tr[i, , drop = FALSE], y_tr[i], x_te, y_te, seed = r)
    data.frame(n = n, rep = r, Modele = names(f$rmse), RMSE = unname(f$rmse))
  }))
}))
write.csv(res_a, "results/10_courbe_apprentissage.csv", row.names = FALSE)

med_a <- aggregate(RMSE ~ n + Modele, data = res_a, FUN = median)
tab_a <- reshape(med_a, idvar = "n", timevar = "Modele", direction = "wide")
names(tab_a) <- sub("RMSE\\.", "", names(tab_a))
cat("RMSE test (médiane des répétitions) :\n")
print(round(tab_a, 0), row.names = FALSE)

g_a <- ggplot(res_a, aes(x = n, y = RMSE, colour = Modele, fill = Modele)) +
  stat_summary(fun.min = min, fun.max = max, geom = "ribbon", alpha = 0.15, colour = NA) +
  stat_summary(fun = median, geom = "line", linewidth = 1) +
  stat_summary(fun = median, geom = "point", size = 2) +
  scale_x_log10(breaks = tailles) +
  scale_y_log10(labels = function(v) format(v, big.mark = " ", scientific = FALSE)) +
  labs(title = "Erreur de prédiction selon la taille de l'échantillon d'apprentissage",
       subtitle = "Médiane et étendue (min-max) sur 10 tirages ; échelle logarithmique ; même jeu de test",
       x = "Nombre de logements dans l'apprentissage", y = "RMSE sur le test ($)") +
  theme_minimal(base_size = 13)
ggsave("results/10_courbe_apprentissage.png", g_a, width = 8.5, height = 5, dpi = 150)

# ============================================================
# EXPERIENCE B : 200 colonnes de bruit pur, selon la taille d'apprentissage
# ============================================================
# Même sous-échantillons que l'expérience A (mêmes graines 1000 + r) : la
# comparaison « sans bruit / avec bruit » est donc appariée, sur les mêmes lignes.
# Durée : quelques minutes (30 ajustements). Réduire reps_b pour aller plus vite.
cat("\n--- B. Ajout de 200 variables de bruit (10 tirages par taille) ---\n")
k        <- 200
tailles_b <- c(500, 1000, nrow(x_tr))
reps_b   <- 10
nom_bruit <- paste0("bruit", seq_len(k))

compte <- function(cvfit, s) {
  b <- as.matrix(coef(cvfit, s = s))[-1, 1]
  est_bruit <- grepl("^bruit", names(b))
  c(reelles = sum(b[!est_bruit] != 0), bruit = sum(b[est_bruit] != 0))
}

res_b <- do.call(rbind, lapply(tailles_b, function(n) {
  do.call(rbind, lapply(seq_len(reps_b), function(r) {
    cat("  n =", n, "| tirage", r, "/", reps_b, "\n")
    set.seed(1000 + r)                                  # mêmes lignes que l'expérience A
    i <- if (n >= nrow(x_tr)) seq_len(nrow(x_tr)) else sample(nrow(x_tr), n)
    set.seed(5000 + r)                                  # bruit différent à chaque tirage
    b_tr <- matrix(rnorm(nrow(x_tr) * k), ncol = k, dimnames = list(NULL, nom_bruit))
    b_te <- matrix(rnorm(nrow(x_te) * k), ncol = k, dimnames = list(NULL, nom_bruit))
    f  <- fit3(cbind(x_tr, b_tr)[i, , drop = FALSE], y_tr[i], cbind(x_te, b_te), y_te, seed = r)
    c1 <- compte(f$las, "lambda.min")
    c2 <- compte(f$las, "lambda.1se")
    data.frame(n = n, rep = r, Modele = names(f$rmse), RMSE_bruit = unname(f$rmse),
               bruit_min = unname(c1["bruit"]), reelles_min = unname(c1["reelles"]),
               bruit_1se = unname(c2["bruit"]), reelles_1se = unname(c2["reelles"]))
  }))
}))

# Référence « sans bruit » : résultats de l'expérience A sur les mêmes lignes
base <- res_a[res_a$n %in% tailles_b, c("n", "rep", "Modele", "RMSE")]
res_b <- merge(res_b, base, by = c("n", "rep", "Modele"))
write.csv(res_b, "results/10_bruit_detail.csv", row.names = FALSE)

resume_b <- aggregate(cbind(RMSE, RMSE_bruit) ~ n + Modele, data = res_b, FUN = median)
names(resume_b)[3:4] <- c("RMSE_sans_bruit", "RMSE_avec_bruit")
resume_b$Hausse_pct <- round(100 * (resume_b$RMSE_avec_bruit / resume_b$RMSE_sans_bruit - 1), 1)
resume_b <- resume_b[order(resume_b$n, resume_b$Modele), ]
cat("\nRMSE test, médiane des", reps_b, "tirages (p =", p_eff, "variables réelles + 200 de bruit) :\n")
print(transform(resume_b, RMSE_sans_bruit = round(RMSE_sans_bruit),
                RMSE_avec_bruit = round(RMSE_avec_bruit)), row.names = FALSE)
write.csv(resume_b, "results/10_bruit_rmse.csv", row.names = FALSE)

# Sélection du Lasso : combien de colonnes de bruit retenues (sur 200) ?
sel <- unique(res_b[, c("n", "rep", "bruit_min", "reelles_min", "bruit_1se", "reelles_1se")])
sel_med <- aggregate(cbind(bruit_min, reelles_min, bruit_1se, reelles_1se) ~ n, data = sel, FUN = median)
cat("\nLasso : variables retenues (médiane des tirages ; bruit = sur", k, "colonnes inutiles) :\n")
print(sel_med, row.names = FALSE)
write.csv(sel_med, "results/10_bruit_selection.csv", row.names = FALSE)

g_b <- ggplot(res_b, aes(x = n, y = RMSE_bruit, colour = Modele, fill = Modele)) +
  stat_summary(fun.min = min, fun.max = max, geom = "ribbon", alpha = 0.15, colour = NA) +
  stat_summary(fun = median, geom = "line", linewidth = 1) +
  stat_summary(fun = median, geom = "point", size = 2) +
  scale_x_log10(breaks = tailles_b) +
  scale_y_log10(labels = function(v) format(v, big.mark = " ", scientific = FALSE)) +
  labs(title = "Effet de 200 variables de bruit selon la taille d'apprentissage",
       subtitle = "Médiane et étendue (min-max) sur 10 tirages de bruit ; échelle logarithmique ; même jeu de test",
       x = "Nombre de logements dans l'apprentissage", y = "RMSE sur le test ($)") +
  theme_minimal(base_size = 13)
ggsave("results/10_bruit_selon_taille.png", g_b, width = 8.5, height = 5, dpi = 150)

cat("\n===== Fin du script 10 : voir results/10_*.csv et results/10_*.png =====\n")