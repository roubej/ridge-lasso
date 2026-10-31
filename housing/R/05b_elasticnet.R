# ============================================================
# 05b - Elastic Net
# alpha = 0.5 (mélange Ridge / Lasso)
# ============================================================

cat("\n===== ELASTIC NET (alpha = 0.5) =====\n")

set.seed(123)

cv_enet <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 0.5,
  nfolds = 10,
  standardize = TRUE
)

lambda_enet_min <- cv_enet$lambda.min

# Réajustement au lambda retenu, comme pour Ridge et Lasso
enet_model <- glmnet(
  x_train,
  y_train,
  alpha = 0.5,
  lambda = lambda_enet_min,
  standardize = TRUE
)

pred_enet <- as.numeric(predict(enet_model, newx = x_test))

met_enet <- metrics_reg(pred_enet, y_test)
rmse_enet <- met_enet["RMSE"]
mae_enet  <- met_enet["MAE"]
r2_enet   <- met_enet["R2"]

n_enet <- sum(as.numeric(coef(enet_model))[-1] != 0)

cat("Lambda :", lambda_enet_min, "\n")
cat("RMSE :", round(rmse_enet, 2), "\n")
cat("MAE  :", round(mae_enet, 2), "\n")
cat("R²   :", round(r2_enet, 4), "\n")
cat("Variables non nulles :", n_enet, "\n")
