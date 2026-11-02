# ============================================================
# APPLICATION SHINY - Ridge / Lasso / Elastic Net
# Dataset : Ames Housing (prix de vente en dollars US)
# ============================================================
# Lancement : ouvrir ce fichier dans RStudio puis cliquer sur "Run App",
#             ou dans la console : shiny::runApp("app")  (depuis housing/)
#
# Packages nécessaires (à installer UNE fois, dans la console) :
#   install.packages(c("shiny", "bslib", "plotly", "DT", "glmnet", "caret"))
# ============================================================

library(shiny)
library(bslib)
library(plotly)
library(DT)
library(glmnet)
library(caret)

if (!exists("prepare_ames")) source(file.path("R", "functions.R"))
if (!exists("plot_cv_rmse")) source(file.path("R", "functions.R"))

ames_data <- read.csv(file.path("data", "AmesHousing.csv"), stringsAsFactors = FALSE)

ui <- page_navbar(
  title = "Ames Housing Predictor",
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#1f77b4",
    secondary = "#2c3e50",
    success = "#2ecc71",
    warning = "#f39c12"
  ),

  nav_panel(
    title = "Accueil",
    div(
      class = "container-fluid",
      style = "padding: 24px;",
      h2("Prédiction du prix de vente des logements"),
      p("Cette application compare les modèles Ridge, Lasso, Elastic Net et la régression classique pour prédire le prix d'un logement à partir du jeu de données Ames Housing."),
      br(),

      layout_columns(
        card(
          full_screen = TRUE,
          class = "border-0 shadow-sm",
          card_header("Données"),
          card_body(
            tags$b("Observations : "), nrow(ames_data), br(),
            tags$b("Variables explicatives : "), ncol(ames_data) - 2, br(),
            tags$b("Cible : "), "saleprice"
          )
        ),
        card(
          full_screen = TRUE,
          class = "border-0 shadow-sm",
          card_header("Objectif"),
          card_body(
            "Estimer le prix de vente d'une maison à l'aide de méthodes de régression pénalisées, plus robustes quand il y a beaucoup de variables ou du bruit."
          )
        ),
        card(
          full_screen = TRUE,
          class = "border-0 shadow-sm",
          card_header("Méthodes"),
          card_body(
            "Ridge, Lasso, Elastic Net et régression classique (MCO)."
          )
        )
      ),

      br(),
      card(
        class = "border-0 shadow-sm",
        card_header("Aperçu du jeu de données"),
        card_body(
          DTOutput("data_preview", height = "320px")
        )
      )
    )
  ),

  nav_panel(
    title = "Modèle",
    sidebarLayout(
      sidebarPanel(
        width = 3,
        h4("Paramètres du modèle"),
        selectInput(
          inputId = "model_choice",
          label = "Modèle",
          choices = c("Ridge", "Lasso", "Elastic Net", "MCO"),
          selected = "Lasso"
        ),
        sliderInput(
          inputId = "alpha_value",
          label = "Alpha (Elastic Net)",
          min = 0.05,
          max = 0.95,
          value = 0.5,
          step = 0.05
        ),
        sliderInput(
          inputId = "nfolds",
          label = "Nombre de plis (CV)",
          min = 3,
          max = 15,
          value = 10,
          step = 1
        ),
        numericInput(
          inputId = "seed_value",
          label = "Graine aléatoire",
          value = 123,
          min = 1,
          max = 9999,
          step = 1
        ),
        actionButton("train_model", "Entraîner le modèle", class = "btn-primary", width = "100%")
      ),

      mainPanel(
        width = 9,
        fluidRow(
          value_box(
            title = "RMSE",
            value = textOutput("rmse_box", inline = TRUE),
            showcase = icon("chart-line"),
            theme_color = "primary"
          ),
          value_box(
            title = "MAE",
            value = textOutput("mae_box", inline = TRUE),
            showcase = icon("calculator"),
            theme_color = "success"
          ),
          value_box(
            title = "R²",
            value = textOutput("r2_box", inline = TRUE),
            showcase = icon("percent"),
            theme_color = "warning"
          )
        ),
        br(),
        card(
          class = "border-0 shadow-sm",
          card_header("Courbe de validation croisée"),
          card_body(
            plotOutput("cv_plot", height = "350px")
          )
        ),
        br(),
        card(
          class = "border-0 shadow-sm",
          card_header("Résultats du modèle"),
          card_body(
            tableOutput("model_summary")
          )
        )
      )
    )
  ),

  nav_panel(
    title = "Variables",
    card(
      class = "border-0 shadow-sm",
      card_header("Coefficients les plus influents"),
      card_body(
        DTOutput("coef_table", height = "520px")
      )
    )
  ),

  nav_panel(
    title = "Prédiction",
    sidebarLayout(
      sidebarPanel(
        width = 3,
        h4("Description de la maison"),
        numericInput("gr_liv_area", "Superficie habitable (pi²)", value = 1800, min = 400, max = 5000, step = 25),
        numericInput("overall_qual", "Qualité globale", value = 7, min = 1, max = 10, step = 1),
        numericInput("year_built", "Année de construction", value = 2000, min = 1880, max = 2010, step = 1),
        numericInput("garage_cars", "Nombre de voitures garage", value = 2, min = 0, max = 4, step = 1),
        numericInput("total_bsmt_sf", "Surface du sous-sol (pi²)", value = 900, min = 0, max = 4000, step = 50),
        selectInput("neighborhood", "Quartier", choices = unique(ames_data$Neighborhood), selected = "NAmes"),
        actionButton("predict_house", "Estimer le prix", class = "btn-success", width = "100%")
      ),

      mainPanel(
        width = 9,
        card(
          class = "border-0 shadow-sm",
          card_header("Estimation du prix"),
          card_body(
            h3(textOutput("prediction_value")),
            p("Cette estimation est calculée avec le modèle sélectionné et le prétraitement standard du projet."),
            br(),
            plotlyOutput("prediction_plot", height = "350px")
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {

  prep_data <- reactive({
    set.seed(as.integer(input$seed_value))
    prepare_ames(ames_data, p = 0.80, seed = as.integer(input$seed_value))
  })

  model_metrics <- eventReactive(input$train_model, {
    req(prep_data())
    fit_models(
      prep_data(),
      uselog = FALSE,
      seed = as.integer(input$seed_value),
      nfolds = as.integer(input$nfolds)
    )
  }, ignoreNULL = FALSE)

  selected_model_fit <- reactive({
    req(prep_data())
    if (input$model_choice == "MCO") {
      list(model = "MCO")
    } else {
      set.seed(as.integer(input$seed_value))
      alpha_val <- if (input$model_choice == "Ridge") 0 else if (input$model_choice == "Lasso") 1 else input$alpha_value
      cv_fit <- cv.glmnet(
        prep_data()$x_train,
        prep_data()$y_train,
        alpha = alpha_val,
        nfolds = as.integer(input$nfolds),
        family = "gaussian"
      )
      list(model = cv_fit, alpha = alpha_val)
    }
  })

  output$data_preview <- renderDT({
    datatable(
      head(ames_data, 100),
      options = list(pageLength = 10, scrollX = TRUE),
      rownames = FALSE
    )
  })

  output$model_summary <- renderTable({
    metrics <- model_metrics()
    if (is.null(metrics)) return(NULL)
    metrics[, c("Modele", "Alpha", "Lambda", "RMSE", "MAE", "R2", "Variables")]
  }, digits = 0)

  output$cv_plot <- renderPlot({
    req(model_metrics())
    req(prep_data())
    if (input$model_choice == "MCO") {
      return(NULL)
    }

    alpha_val <- if (input$model_choice == "Ridge") 0 else if (input$model_choice == "Lasso") 1 else input$alpha_value
    set.seed(as.integer(input$seed_value))
    cv_fit <- cv.glmnet(
      prep_data()$x_train,
      prep_data()$y_train,
      alpha = alpha_val,
      nfolds = as.integer(input$nfolds)
    )
    plot_cv_rmse(cv_fit, title = paste("Validation croisée -", input$model_choice))
  })

  output$rmse_box <- renderText({
    metrics <- model_metrics()
    if (is.null(metrics)) return("--")
    selected <- metrics[metrics$Modele == input$model_choice | (input$model_choice == "Elastic Net" & grepl("Elastic Net", metrics$Modele)), ]
    if (nrow(selected) == 0) return("--")
    sprintf("%.0f $", selected$RMSE[1])
  })

  output$mae_box <- renderText({
    metrics <- model_metrics()
    if (is.null(metrics)) return("--")
    selected <- metrics[metrics$Modele == input$model_choice | (input$model_choice == "Elastic Net" & grepl("Elastic Net", metrics$Modele)), ]
    if (nrow(selected) == 0) return("--")
    sprintf("%.0f $", selected$MAE[1])
  })

  output$r2_box <- renderText({
    metrics <- model_metrics()
    if (is.null(metrics)) return("--")
    selected <- metrics[metrics$Modele == input$model_choice | (input$model_choice == "Elastic Net" & grepl("Elastic Net", metrics$Modele)), ]
    if (nrow(selected) == 0) return("--")
    sprintf("%.3f", selected$R2[1])
  })

  output$coef_table <- renderDT({
    req(prep_data())
    req(selected_model_fit())
    if (input$model_choice == "MCO") return(datatable(data.frame(Message = "Le modèle MCO n'affiche pas de coefficients pénalisés.")))

    fit <- selected_model_fit()$model
    coefs <- as.matrix(coef(fit, s = "lambda.min"))
    coefs_df <- as.data.frame(coefs)
    names(coefs_df) <- c("Coefficient")
    coefs_df$Variable <- rownames(coefs_df)
    coefs_df <- coefs_df[coefs_df$Variable != "(Intercept)", c("Variable", "Coefficient")]
    coefs_df <- coefs_df[order(abs(coefs_df$Coefficient), decreasing = TRUE), ]
    coefs_df$Coefficient <- round(coefs_df$Coefficient, 4)
    datatable(coefs_df, options = list(pageLength = 20, scrollX = TRUE), rownames = FALSE)
  })

  prediction_values <- eventReactive(input$predict_house, {
    req(prep_data())

    overrides <- list(
      gr_liv_area = input$gr_liv_area,
      overall_qual = input$overall_qual,
      year_built = input$year_built,
      garage_cars = input$garage_cars,
      total_bsmt_sf = input$total_bsmt_sf,
      neighborhood = input$neighborhood
    )

    x_new <- make_new_x(prep_data(), overrides = overrides)
    alpha_val <- if (input$model_choice == "Ridge") 0 else if (input$model_choice == "Lasso") 1 else input$alpha_value

    if (input$model_choice == "MCO") {
      model_ols <- suppressWarnings(lm(prep_data()$y_train ~ ., data = as.data.frame(prep_data()$x_train)))
      pred <- as.numeric(predict(model_ols, as.data.frame(x_new)))
    } else {
      cv_fit <- cv.glmnet(
        prep_data()$x_train,
        prep_data()$y_train,
        alpha = alpha_val,
        nfolds = as.integer(input$nfolds),
        family = "gaussian"
      )
      pred <- as.numeric(predict(cv_fit, newx = x_new, s = "lambda.min"))
    }

    pred
  })

  output$prediction_value <- renderText({
    val <- prediction_values()
    paste0("Prix estimé : ", format(round(val[1], 0), big.mark = " "), " $")
  })

  output$prediction_plot <- renderPlotly({
    p <- plot_ly(
      x = c("Maison saisie", "Prix moyen du jeu"),
      y = c(prediction_values()[1], mean(prep_data()$y_test)),
      type = "bar",
      marker = list(color = c("#1f77b4", "#2ecc71")),
      text = ~paste0(round(y, 0), " $"),
      textposition = "outside"
    ) %>%
      layout(
        title = "Comparaison du prix estimé et du prix moyen du jeu de test",
        xaxis = list(title = ""),
        yaxis = list(title = "Prix ($)")
      )
    p
  })
}

shinyApp(ui, server)
