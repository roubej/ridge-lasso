# ============================================================
# APPLICATION SHINY - Ridge / Lasso / Elastic Net
# Dataset : Ames Housing (prix de vente en dollars US)
# ============================================================
# Lancement : ouvrir ce fichier dans RStudio puis cliquer sur "Run App",
#             ou dans la console : shiny::runApp("app")  (depuis housing/)
#
# Packages nécessaires (à installer UNE fois, dans la console) :
#   install.packages(c("shiny", "bslib", "plotly", "DT", "glmnet", "caret"))
#
# Organisation du fichier :
#   1. Packages et données        (exécuté une seule fois au démarrage)
#   2. Fonctions d'affichage      (thème des graphiques, formats français)
#   3. UI                         (l'interface : 7 onglets)
#   4. SERVER                     (le calcul : modèle, graphiques, prédiction)
# ============================================================

# ------------------------------------------------------------
# 1. PACKAGES ET DONNEES
# ------------------------------------------------------------
needed  <- c("shiny", "bslib", "plotly", "DT", "glmnet", "caret")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Packages manquants : ", paste(missing, collapse = ", "),
       ".\nInstallez-les avec : install.packages(c(",
       paste0('"', missing, '"', collapse = ", "), "))", call. = FALSE)
}
library(shiny)
library(bslib)
library(plotly)
library(DT)
library(glmnet)
library(caret)

# Chemin du projet : fonctionne avec "Run App" (dossier app/) ou depuis housing/
root <- if (file.exists("../data/AmesHousing.csv")) ".." else
        if (file.exists("data/AmesHousing.csv"))    "."  else
        stop("Fichier data/AmesHousing.csv introuvable. Ouvrez le projet housing.Rproj.", call. = FALSE)

source(file.path(root, "R", "functions.R"))   # prepare_ames(), make_new_x(), metrics_reg()

data <- read.csv(file.path(root, "data", "AmesHousing.csv"), stringsAsFactors = FALSE)

neighborhoods <- sort(unique(data$neighborhood))

# Noms des variables en français (ceux qui manquent gardent leur nom d'origine)
labels_fr <- c(
  saleprice = "Prix de vente ($)", overall_qual = "Qualité générale (1-10)",
  overall_cond = "État général (1-10)", gr_liv_area = "Surface habitable (pi²)",
  year_built = "Année de construction", year_remod_add = "Année de rénovation",
  total_bsmt_sf = "Surface du sous-sol (pi²)", garage_cars = "Places de garage",
  garage_area = "Surface du garage (pi²)", full_bath = "Salles de bain complètes",
  half_bath = "Demi-salles de bain", bedroom_abvgr = "Chambres",
  lot_area = "Surface du terrain (pi²)", lot_frontage = "Façade du terrain (pi)",
  X1st_flr_sf = "Surface du 1er niveau (pi²)", X2nd_flr_sf = "Surface du 2e niveau (pi²)",
  totrms_abvgrd = "Pièces (hors sous-sol)", fireplaces = "Cheminées",
  mas_vnr_area = "Surface de placage (pi²)", wood_deck_sf = "Terrasse en bois (pi²)",
  open_porch_sf = "Porche ouvert (pi²)", neighborhood = "Quartier",
  house_style = "Style de maison", bldg_type = "Type de bâtiment",
  central_air = "Climatisation centrale", kitchen_qual = "Qualité de la cuisine",
  ms_zoning = "Zonage", sale_condition = "Condition de vente",
  garage_yr_blt = "Année du garage", bsmtfin_sf_1 = "Sous-sol aménagé (pi²)", bsmt_unf_sf = "Sous-sol non fini (pi²)",
  enclosed_porch = "Porche fermé (pi²)", bsmt_qual = "Qualité du sous-sol", exter_qual = "Qualité extérieure",
  roof_matl = "Matériau du toit", misc_feature = "Équipement divers", condition_2 = "Condition (2)",
  bsmt_exposure = "Accès extérieur du sous-sol", pool_qc = "Qualité de la piscine", ms_subclass = "Type de logement"
)
# Nom lisible : libellé français, sinon le nom d'origine sans les « _ »
lab <- function(v) ifelse(v %in% names(labels_fr), unname(labels_fr[v]), gsub("_", " ", v))

# Codes de qualité -> français
qual_fr <- c(Ex = "Excellent", Gd = "Bon", TA = "Moyen", Fa = "Passable", Po = "Mauvais", Missing = "Absent", None = "Aucun")

# Colonnes issues du codage 0/1 (ex. "bsmt_qual.Ex") -> "Qualité du sous-sol : Excellent"
pretty_var <- function(v) {
  vapply(v, function(x) {
    parts <- strsplit(x, ".", fixed = TRUE)[[1]]
    if (length(parts) >= 2) {
      lv <- paste(parts[-1], collapse = ".")
      if (lv %in% names(qual_fr)) lv <- qual_fr[[lv]]
      paste0(lab(parts[1]), " : ", lv)
    } else lab(x)
  }, character(1), USE.NAMES = FALSE)
}


# Textes du tableau DT en français
dt_lang <- list(search = "Rechercher :", lengthMenu = "Afficher _MENU_ lignes",
                info = "Lignes _START_ à _END_ sur _TOTAL_", zeroRecords = "Aucun résultat",
                thousands = "\u00a0", decimal = ",",
                paginate = list(previous = "Précédent", `next` = "Suivant"))

# ------------------------------------------------------------
# 2. FONCTIONS D'AFFICHAGE
# ------------------------------------------------------------
# Style des tableaux DT : bootstrap5 si la version de DT le permet, sinon bootstrap4
dt_style <- tryCatch({
  ok <- eval(formals(DT::datatable)$style)
  if ("bootstrap5" %in% ok) "bootstrap5" else "bootstrap4"
}, error = function(e) "default")

fmt_k <- function(x) paste0(formatC(round(x / 1000), format = "d", big.mark = "\u00a0"), "\u00a0k$")
fmt_usd <- function(x) paste0(formatC(round(x), format = "d", big.mark = "\u00a0"), "\u00a0$")
fmt_num <- function(x, d = 3) format(round(x, d), nsmall = d, decimal.mark = ",")

# Couleurs des graphiques selon le mode clair / sombre
pal <- function(mode) {
  if (identical(mode, "dark")) {
    list(fg = "#E6EAF0", grid = "rgba(255,255,255,0.12)", main = "#2BB3C4", accent = "#F2B450", muted = "#9AA3B2",
         band = "rgba(43,179,196,0.25)", bad = "#FF7F6B", mid = "#2b3038")
  } else {
    list(fg = "#1d2a3a", grid = "rgba(0,0,0,0.10)", main = "#0F7B8A", accent = "#D9822B", muted = "#7A8499",
         band = "rgba(15,123,138,0.20)", bad = "#C0392B", mid = "#f5f5f5")
  }
}

# Applique le même style (fond transparent, couleurs, polices) à tous les graphiques
style_plot <- function(p, mode, title = NULL, xtitle = NULL, ytitle = NULL, ...) {
  cl <- pal(mode)
  p %>%
    layout(
      title = list(text = title, x = 0.02, font = list(size = 15)),
      xaxis = list(title = xtitle, gridcolor = cl$grid, zerolinecolor = cl$grid),
      yaxis = list(title = ytitle, gridcolor = cl$grid, zerolinecolor = cl$grid),
      paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
      font = list(color = cl$fg), margin = list(t = 55),
      legend = list(orientation = "h", y = -0.25), ...
    ) %>%
    config(displaylogo = FALSE, locale = "fr")
}

# Ligne verticale + étiquette (pour λ, moyenne, prix estimé...)
vline <- function(x, color, dash = "dash") {
  list(type = "line", x0 = x, x1 = x, y0 = 0, y1 = 1, yref = "paper",
       line = list(color = color, width = 1.6, dash = dash))
}
vlabel <- function(x, text, color, y = 1.04) {
  list(x = x, y = y, yref = "paper", text = text, showarrow = FALSE,
       font = list(color = color, size = 11))
}

# ------------------------------------------------------------
# 3. UI (interface)
# ------------------------------------------------------------
ui <- page_navbar(
  title = "Prix immobilier · Ridge, Lasso & Elastic Net",
  id = "nav", lang = "fr", fillable = FALSE,
  theme = bs_theme(version = 5, primary = "#0F7B8A", secondary = "#E8A33D",
                   base_font = c("Segoe UI", "system-ui", "-apple-system", "sans-serif")),
  header = tagList(
    tags$style(HTML("
      .card-header { font-weight: 600; }
      .bslib-value-box .value-box-value { font-size: 1.6rem; }
      .small-note { font-size: 0.85rem; opacity: 0.75; }
      .bslib-grid > .card, .bslib-grid > div { min-width: 0; }
      .js-plotly-plot, .plotly.html-widget { max-width: 100%; }
    ")),
    # Les graphiques créés dans un onglet caché sont redimensionnés quand l'onglet s'affiche
    tags$script(HTML("$(document).on('shown.bs.tab', function() { setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 150); });"))
  ),

  # ---------------- Onglet 1 : Données ----------------
  nav_panel("Données", icon = icon("table"),
    layout_column_wrap(width = 1/4, fill = FALSE,
      value_box("Logements", format(nrow(data), big.mark = "\u00a0"), theme = "primary"),
      value_box("Variables", ncol(data) - 2, theme = "primary"),
      value_box("Prix médian", fmt_usd(median(data$saleprice)), theme = "primary"),
      value_box("Cellules manquantes",
                paste0(fmt_num(100 * mean(is.na(data)), 1), " %"), theme = "primary")
    ),
    layout_column_wrap(width = 1/2,
      card(card_header("À propos du jeu de données"),
        p("Ames Housing : ventes ", strong("réelles"), " de maisons à Ames (Iowa, États-Unis)
          entre 2006 et 2010, d'après le registre du service d'évaluation de la ville.
          Compilé par Dean De Cock (2011, ", em("Journal of Statistics Education"), ")."),
        tags$ul(
          tags$li(strong("Cible : "), "saleprice, prix de vente en dollars US."),
          tags$li(strong("Explicatives : "), "environ 80 variables (surfaces, qualité, année,
                  quartier, garage, sous-sol...), numériques et catégorielles."),
          tags$li(strong("Valeurs atypiques : "), "5 maisons de plus de 4 000 pi², dont 3 ventes
                  « Partial » (maisons non terminées) vendues très en dessous de leur valeur."),
          tags$li(strong("Pourquoi ce jeu : "), "beaucoup de variables corrélées pour un nombre
                  limité de logements, donc situation idéale pour Ridge, Lasso et Elastic Net.")
        ),
        p(class = "small-note", "1 pi² (pied carré) = 0,093 m².")),
      card(card_header("Variables avec des valeurs manquantes"),
           plotlyOutput("plot_na", height = "320px"), full_screen = TRUE)
    ),
    card(card_header("Aperçu des données"), DTOutput("tbl_data"), full_screen = TRUE)
  ),

  # ---------------- Onglet 3 : Modèle ----------------
  nav_panel("Modèle", icon = icon("sliders"),
    layout_sidebar(
      sidebar = sidebar(width = 330, title = "Paramètres du modèle",
        selectInput("algo", "Algorithme", c("Ridge", "Lasso", "Elastic Net"), selected = "Lasso"),
        conditionalPanel("input.algo == 'Elastic Net'",
          sliderInput("alpha", "Mélange α (0 = Ridge, 1 = Lasso)",
                      min = 0.05, max = 0.95, value = 0.5, step = 0.05)),
        radioButtons("lam_rule", "Choix de λ",
          choices = c("λ min (meilleure erreur CV)" = "min",
                      "λ 1se (modèle plus simple)" = "1se",
                      "Manuel" = "manuel")),
        conditionalPanel("input.lam_rule == 'manuel'",
          sliderInput("loglam", "log10(λ)", min = -4, max = 6, value = 3, step = 0.1)),
        sliderInput("folds", "Plis de validation croisée", min = 3, max = 10, value = 10, step = 1),
        p(class = "small-note", "Découpage fixe : 80 % apprentissage, 20 % test (graine 123)."),
        actionButton("train", "Entraîner le modèle", icon = icon("play"),
                     class = "btn-primary w-100")
      ),
      layout_column_wrap(width = 1/4, fill = FALSE,
        value_box("RMSE (test)", textOutput("vb_rmse"), theme = "primary"),
        value_box("MAE (test)", textOutput("vb_mae"), theme = "primary"),
        value_box("R² (test)", textOutput("vb_r2"), theme = "primary"),
        value_box("Variables retenues", textOutput("vb_nvar"), theme = "primary")
      ),
      textOutput("lam_info"),
      navset_card_tab(id = "res",
        nav_panel("Validation croisée", plotlyOutput("plot_cv", height = "420px")),
        nav_panel("Chemins des coefficients", plotlyOutput("plot_path", height = "460px")),
        nav_panel("Réel vs prédit", plotlyOutput("plot_pred", height = "460px"))
      )
    )
  ),

  # ---------------- Onglet 4 : Variables ----------------
  nav_panel("Variables", icon = icon("list-ol"),
    card(card_header("Variables les plus influentes (coefficients standardisés)"),
      sliderInput("topn", "Nombre de variables affichées", min = 5, max = 40, value = 15, step = 1),
      plotlyOutput("plot_coef", height = "520px"),
      p(class = "small-note", "Coefficient standardisé = coefficient × écart-type de la variable :
        permet de comparer des variables d'unités différentes. Les barres turquoise augmentent le
        prix, les rouges le diminuent. Les barres orange (⚠) sont des catégories présentes dans 3 maisons
        ou moins : leur coefficient absorbe surtout une vente atypique, il ne faut pas l'interpréter."),
      full_screen = TRUE),
    card(card_header("Tous les coefficients non nuls"),
      DTOutput("tbl_coef"))
  ),

  # ---------------- Onglet 6 : Prédiction ----------------
  nav_panel("Prédiction", icon = icon("house"),
    layout_sidebar(
      sidebar = sidebar(width = 330, title = "Caractéristiques du logement",
        selectInput("p_neigh", "Quartier", choices = neighborhoods,
                    selected = names(which.max(table(data$neighborhood)))),
        sliderInput("p_qual", "Qualité générale (1-10)", min = 1, max = 10, value = 6, step = 1),
        numericInput("p_area", "Surface habitable (pi²)", value = 1500, min = 300, max = 6000, step = 50),
        textOutput("p_area_m2"),
        numericInput("p_year", "Année de construction", value = 1995, min = 1870, max = 2010, step = 1),
        numericInput("p_bsmt", "Surface du sous-sol (pi²)", value = 1000, min = 0, max = 4000, step = 50),
        sliderInput("p_garage", "Places de garage", min = 0, max = 5, value = 2, step = 1),
        sliderInput("p_bath", "Salles de bain complètes", min = 0, max = 4, value = 2, step = 1),
        sliderInput("p_bed", "Chambres", min = 0, max = 8, value = 3, step = 1),
        actionButton("estimate", "Estimer le prix", icon = icon("calculator"),
                     class = "btn-primary w-100"),
        p(class = "small-note", style = "margin-top:10px",
          "Les autres caractéristiques prennent la valeur typique du jeu d'apprentissage
           (médiane ou modalité la plus fréquente).")
      ),
      uiOutput("p_warning"),
      layout_column_wrap(width = 1/3, fill = FALSE,
        value_box("Prix estimé", textOutput("p_price"), theme = "primary"),
        value_box("Fourchette indicative (± 1 RMSE)", textOutput("p_range"), theme = "secondary"),
        value_box("Prix médian du quartier", textOutput("p_neigh_med"), theme = "secondary")
      ),
      card(card_header("Où se situe ce logement ?"),
           plotlyOutput("plot_where", height = "380px")),
      p(class = "small-note", "Le modèle utilisé est celui entraîné dans l'onglet « Modèle »
        (algorithme, λ et paramètres choisis). L'estimation est indicative.")
    )
  )

)

# ------------------------------------------------------------
# 4. SERVER (calculs et graphiques)
# ------------------------------------------------------------
server <- function(input, output, session) {

  # Réglages fixes (les mêmes que dans main.R) : découpage 80 / 20 et graine 123
  SEED  <- 123
  SPLIT <- 0.80

  # Thème graphique (clair uniquement)
  mode <- reactive("light")

  # ==========================================================
  # A. Onglet Données
  # ==========================================================
  output$tbl_data <- renderDT({
    datatable(data, style = dt_style, class = "table-sm", rownames = FALSE,
      options = list(pageLength = 8, scrollX = TRUE, language = dt_lang))
  })

  output$plot_na <- renderPlotly({
    na <- sort(colSums(is.na(data)), decreasing = TRUE)
    na <- head(na[na > 0], 15)
    cl <- pal(mode())
    plot_ly(x = unname(na), y = factor(names(na), levels = rev(names(na))), type = "bar",
            orientation = "h", marker = list(color = cl$main),
            hovertemplate = "%{y} : %{x} valeurs manquantes<extra></extra>") %>%
      style_plot(mode(), xtitle = "Nombre de valeurs manquantes")
  })

  # ==========================================================
  # C. Entraînement du modèle (déclenché par le bouton, et au démarrage)
  # ==========================================================
  fit <- eventReactive(input$train, {
    withProgress(message = "Entraînement du modèle...", value = 0, {
      incProgress(0.15, detail = "Préparation des données")
      dat   <- data
      prep  <- prepare_ames(dat, p = SPLIT, seed = SEED)
      alpha <- switch(input$algo, "Ridge" = 0, "Lasso" = 1, input$alpha)
      uselog <- FALSE   # prix brut, sans transformation logarithmique
      yt <- if (uselog) log(prep$y_train) else prep$y_train

      incProgress(0.35, detail = "Validation croisée")
      set.seed(SEED)
      cv  <- cv.glmnet(prep$x_train, yt, alpha = alpha, nfolds = input$folds)
      lam <- switch(input$lam_rule, "min" = cv$lambda.min, "1se" = cv$lambda.1se, 10^input$loglam)

      incProgress(0.3, detail = "Ajustement final")
      model <- glmnet(prep$x_train, yt, alpha = alpha, lambda = lam)
      path  <- glmnet(prep$x_train, yt, alpha = alpha)

      pred <- as.numeric(predict(model, newx = prep$x_test))
      if (uselog) pred <- exp(pred)
      met  <- metrics_reg(pred, prep$y_test)

      beta <- as.numeric(coef(model))[-1]
      sds  <- apply(prep$x_train, 2, sd)
      coefs <- data.frame(variable = colnames(prep$x_train), coefficient = beta,
                          coef_std = beta * sds, n_train = as.numeric(colSums(prep$x_train != 0)),
                          stringsAsFactors = FALSE)
      # Colonne 0/1 (catégorie) présente dans 3 maisons ou moins : variable « rare »
      coefs$rare <- grepl(".", coefs$variable, fixed = TRUE) & coefs$n_train <= 3

      list(prep = prep, alpha = alpha, algo = input$algo, uselog = uselog, cv = cv, lambda = lam,
           model = model, path = path, pred = pred, metrics = met, coefs = coefs, sds = sds,
           folds = input$folds, lam_rule = input$lam_rule, seed = SEED,
           data_used = dat, excl = FALSE, split = SPLIT * 100)
    })
  }, ignoreNULL = FALSE)

  output$vb_rmse <- renderText(fmt_usd(fit()$metrics["RMSE"]))
  output$vb_mae  <- renderText(fmt_usd(fit()$metrics["MAE"]))
  output$vb_r2   <- renderText(fmt_num(fit()$metrics["R2"], 3))
  output$vb_nvar <- renderText({
    f <- fit(); paste0(sum(f$coefs$coefficient != 0), " / ", sum(f$sds > 0))
  })
  output$lam_info <- renderText({
    f <- fit()
    fl <- function(x) format(signif(x, 4), decimal.mark = ",", big.mark = "\u00a0")
    paste0("Algorithme : ", f$algo, if (f$algo == "Elastic Net") paste0(" (α = ", f$alpha, ")") else "",
           " · λ choisi = ", fl(f$lambda), " (log10 = ", fmt_num(log10(f$lambda), 2), ")",
           " · λ min = ", fl(f$cv$lambda.min), " · λ 1se = ", fl(f$cv$lambda.1se),
           
           if (f$excl) " · sans les 5 maisons > 4 000 pi²" else "")
  })

  output$plot_cv <- renderPlotly({
    f <- fit(); cl <- pal(mode()); cv <- f$cv
    # On affiche la racine de l'erreur quadratique (RMSE) : lisible, en dollars
    d <- data.frame(l = log10(cv$lambda), m = sqrt(cv$cvm), lo = sqrt(cv$cvlo), hi = sqrt(cv$cvup), nz = cv$nzero)
    mn <- min(d$m)
    lc <- log10(f$lambda); l1 <- log10(cv$lambda.min); l2 <- log10(cv$lambda.1se)
    plot_ly(d, x = ~l) %>%
      add_ribbons(ymin = ~lo, ymax = ~hi, fillcolor = cl$band, line = list(width = 0),
                  name = "± 1 écart-type", hoverinfo = "skip") %>%
      add_lines(y = ~m, name = "Erreur de validation croisée", line = list(color = cl$main, width = 3),
                text = ~nz, hovertemplate = "log10(λ) = %{x:.2f}<br>Erreur = %{y:,.0f}<br>Variables non nulles : %{text}<extra></extra>") %>%
      style_plot(mode(), title = paste0("Choix de λ par validation croisée (", f$folds, " plis)"),
                 xtitle = "log10(λ)  (plus à droite = modèle plus simple)",
                 ytitle = if (f$uselog) "Erreur (échelle log)" else "Erreur de prédiction ($)",
                 shapes = list(vline(l1, cl$accent), vline(l2, cl$bad, "dot"), vline(lc, cl$fg, "solid")),
                 annotations = list(vlabel(l1, "λ min", cl$accent), vlabel(l2, "λ 1se", cl$bad, y = 1.12),
                                    vlabel(lc, "λ choisi", cl$fg, y = 1.20))) %>%
      layout(yaxis = list(range = c(mn * 0.9, mn * 2.2)))
  })

  output$plot_path <- renderPlotly({
    f <- fit(); cl <- pal(mode())
    B <- as.matrix(f$path$beta) * f$sds
    keep <- order(apply(abs(B), 1, max), decreasing = TRUE)[seq_len(min(20, nrow(B)))]
    x <- log10(f$path$lambda)
    p <- plot_ly()
    for (i in keep) {
      p <- add_lines(p, x = x, y = B[i, ], name = rownames(B)[i], line = list(width = 1.8),
                     hovertemplate = paste0(rownames(B)[i], "<br>log10(λ) = %{x:.2f}<br>coef. std = %{y:,.0f}<extra></extra>"))
    }
    p %>% style_plot(mode(), title = "Chemins des coefficients (20 variables les plus fortes)",
                     xtitle = "log10(λ)", ytitle = "Coefficient standardisé",
                     shapes = list(vline(log10(f$lambda), cl$fg, "solid")),
                     annotations = list(vlabel(log10(f$lambda), "λ choisi", cl$fg)),
                     legend = list(orientation = "v", font = list(size = 10)))
  })

  output$plot_pred <- renderPlotly({
    f <- fit(); cl <- pal(mode())
    y <- f$prep$y_test; r <- range(c(y, f$pred))
    plot_ly() %>%
      add_markers(x = y, y = f$pred, name = "Logements du test",
                  marker = list(color = cl$main, opacity = 0.55, size = 7),
                  hovertemplate = "Réel : %{x:,.0f} $<br>Prédit : %{y:,.0f} $<extra></extra>") %>%
      add_lines(x = r, y = r, name = "Prédiction parfaite", line = list(color = cl$accent, dash = "dash", width = 2)) %>%
      style_plot(mode(), title = paste0("Prix réels vs prix prédits (R² = ", fmt_num(f$metrics["R2"], 3), ")"),
                 xtitle = "Prix réel ($)", ytitle = "Prix prédit ($)")
  })

  # ==========================================================
  # D. Onglet Variables
  # ==========================================================
  coefs_nz <- reactive({
    d <- fit()$coefs
    d <- d[d$coefficient != 0, ]
    d[order(abs(d$coef_std), decreasing = TRUE), ]
  })

  output$plot_coef <- renderPlotly({
    cl <- pal(mode())
    top <- head(coefs_nz(), input$topn)
    validate(need(nrow(top) > 0, "Aucune variable retenue avec ce λ : diminuez λ."))
    nm <- pretty_var(top$variable)
    nm <- ifelse(top$rare, paste0(nm, "  ⚠ ", top$n_train, " maison(s)"), nm)
    col <- ifelse(top$rare, cl$accent, ifelse(top$coef_std > 0, cl$main, cl$bad))
    plot_ly(x = top$coef_std, y = factor(nm, levels = rev(unique(nm))),
            type = "bar", orientation = "h", marker = list(color = col),
            hovertemplate = "%{y}<br>Coefficient standardisé : %{x:,.0f}<extra></extra>") %>%
      style_plot(mode(), title = paste0(fit()$algo, " : ", nrow(top), " variables les plus influentes"),
                 xtitle = "Effet sur le prix de +1 écart-type de la variable ($)") %>%
      layout(margin = list(l = 260, t = 55))
  })

  output$tbl_coef <- renderDT({
    d <- coefs_nz()
    out <- data.frame(Variable = pretty_var(d$variable), `Nom technique` = d$variable, Coefficient = d$coefficient,
                      `Coefficient standardisé` = d$coef_std,
                      `Maisons concernées` = ifelse(grepl(".", d$variable, fixed = TRUE), d$n_train, NA),
                      check.names = FALSE)
    datatable(out, style = dt_style, class = "table-sm", rownames = FALSE,
      options = list(pageLength = 10, language = dt_lang)) %>%
      formatRound(c("Coefficient", "Coefficient standardisé"), 2)
  })

  # ==========================================================
  # F. Onglet Prédiction
  # ==========================================================
  output$p_area_m2 <- renderText({
    req(!is.na(input$p_area)); paste0("≈ ", round(input$p_area * 0.092903), " m²")
  })

  output$p_warning <- renderUI({
    msgs <- c()
    chk <- function(v, val, name) {
      if (!is.na(val) && (val < min(data[[v]], na.rm = TRUE) || val > max(data[[v]], na.rm = TRUE)))
        msgs <<- c(msgs, paste0(name, " (", val, ") est hors de la plage observée (",
                                min(data[[v]], na.rm = TRUE), " à ", max(data[[v]], na.rm = TRUE), ")."))
    }
    chk("gr_liv_area", input$p_area, "La surface habitable")
    chk("year_built", input$p_year, "L'année de construction")
    chk("total_bsmt_sf", input$p_bsmt, "La surface du sous-sol")
    if (length(msgs) == 0) return(NULL)
    div(class = "alert alert-warning", HTML(paste0("Attention : ", paste(msgs, collapse = "<br>"),
                                                    "<br>L'estimation sera peu fiable.")))
  })

  # Se recalcule au clic sur « Estimer » ET quand le modèle est ré-entraîné
  pred_res <- reactive({
    input$estimate
    f <- fit()
    isolate({
      req(!is.na(input$p_area), !is.na(input$p_year), !is.na(input$p_bsmt))
      ov <- list(neighborhood = input$p_neigh, overall_qual = input$p_qual,
                 gr_liv_area = input$p_area, year_built = input$p_year,
                 total_bsmt_sf = input$p_bsmt, garage_cars = input$p_garage,
                 full_bath = input$p_bath, bedroom_abvgr = input$p_bed)
      xnew <- make_new_x(f$prep, ov)
      p <- as.numeric(predict(f$model, newx = xnew))
      if (f$uselog) p <- exp(p)
      tr <- f$prep$train_data
      list(price = p, rmse = unname(f$metrics["RMSE"]),
           neigh = median(tr$saleprice[tr$neighborhood == input$p_neigh]))
    })
  })

  output$p_price <- renderText(fmt_usd(pred_res()$price))
  output$p_range <- renderText({
    r <- pred_res(); paste0(fmt_k(max(r$price - r$rmse, 0)), " – ", fmt_k(r$price + r$rmse))
  })
  output$p_neigh_med <- renderText({
    m <- pred_res()$neigh; if (is.na(m)) "n/d" else fmt_usd(m)
  })

  output$plot_where <- renderPlotly({
    r <- pred_res(); cl <- pal(mode())
    plot_ly(x = data$saleprice, type = "histogram", nbinsx = 45,
            marker = list(color = cl$main, opacity = 0.75, line = list(color = "white", width = 0.4)),
            hovertemplate = "Prix : %{x}<br>Logements : %{y}<extra></extra>") %>%
      style_plot(mode(), title = "Prix estimé par rapport à tous les logements du jeu de données",
                 xtitle = "Prix de vente ($)", ytitle = "Nombre de logements",
                 shapes = list(
                   list(type = "rect", x0 = max(r$price - r$rmse, 0), x1 = r$price + r$rmse,
                        y0 = 0, y1 = 1, yref = "paper", fillcolor = cl$band, line = list(width = 0)),
                   vline(r$price, cl$accent, "solid")),
                 annotations = list(vlabel(r$price, "prix estimé", cl$accent)))
  })
}

shinyApp(ui = ui, server = server)
