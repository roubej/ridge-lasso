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
#   3. UI                         (l'interface : 4 onglets)
#   4. SERVER                     (le calcul : modèle, graphiques, prédiction)
# ============================================================
