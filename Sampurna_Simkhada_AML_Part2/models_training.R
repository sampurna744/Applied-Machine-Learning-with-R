# Install and load required libraries for data processing, visualization, and modeling
library(mice)         # For missing value imputation
library(ggplot2)      # For data visualization
library(corrplot)     # For correlation plots
library(FactoMineR)   # For PCA analysis
library(psych)        # For descriptive statistics
library(caret)        # For machine learning workflows
library(smotefamily)  # For SMOTE oversampling
library(pROC)         # For ROC curve analysis
library(dplyr)        # For data manipulation
library(recipes)      # For data preprocessing
library(rpart.plot)   # For decision tree visualization

# Load the bank marketing dataset
bank_data <- read.csv("bank marketing campaign.csv", header = TRUE)

# Calculate percentage of missing values per column
missing_values_by_column <- colSums(is.na(bank_data)) / nrow(bank_data) * 100
print(missing_values_by_column)

# Compute average missing percentage across columns
avg_missing <- mean(missing_values_by_column)
print(paste("Average missing percentage:", avg_missing))

# Handle missing values based on average missing percentage
if (avg_missing > 1) {
  # Use MICE imputation if missing percentage > 1%
  imputed_data <- mice(bank_data, m = 5, method = "pmm", seed = 123)
  clean_data <- complete(imputed_data)
  print("Missing values imputed using MICE.")
} else {
  # Remove rows with missing values if missing percentage <= 1%
  clean_data <- na.omit(bank_data)
  print("Rows with missing values removed.")
}

# Verify no missing values remain after cleaning
print(colSums(is.na(clean_data)) / nrow(clean_data) * 100)

# Select numeric columns for analysis
numeric_columns <- which(sapply(clean_data, is.numeric))
numeric_data <- clean_data[, numeric_columns]

# Define function to detect outliers using IQR method
detect_outliers_boxplot <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(which(x < lower_bound | x > upper_bound))
}

# Detect and report outliers for each numeric column
for (col in names(numeric_data)) {
  outliers <- detect_outliers_boxplot(numeric_data[[col]])
  cat("Column:", col, "- Number of outliers:", length(outliers), "\n")
}

# Perform PCA on numeric data for dimensionality reduction
pca_data <- clean_data[, numeric_columns]
pca_result <- PCA(pca_data, scale.unit = TRUE, graph = FALSE)

# Standardize numeric data for consistent scaling
data_scaled <- scale(numeric_data)

# Ensure target variable is a factor
scaled_data <- data.frame(data_scaled, y = clean_data$y)
scaled_data$y <- as.factor(scaled_data$y)

# One-hot encode categorical variables
dummies <- dummyVars(y ~ ., data = scaled_data)
X_encoded <- as.data.frame(predict(dummies, newdata = scaled_data))

# Extract target variable
y_encoded <- scaled_data$y

# Balance dataset using SMOTE to address class imbalance
set.seed(42)
smote_result <- SMOTE(X_encoded, y_encoded, K = 5)

# Extract balanced features and target
X_balanced <- smote_result$data[, -ncol(smote_result$data)]
y_balanced <- as.factor(smote_result$data$class)

# Combine balanced features and target
balanced_data <- cbind(X_balanced, y = y_balanced)

# Split data into training (80%) and testing (20%) sets
set.seed(123)
indxTrain <- createDataPartition(y = balanced_data$y, p = 0.8, list = FALSE)
train <- balanced_data[indxTrain, ]
test <- balanced_data[-indxTrain, ]

# Check class distribution in training and testing sets
cat("Training set class distribution:\n")
print(prop.table(table(train$y)) * 100)

cat("\nTesting set class distribution:\n")
print(prop.table(table(test$y)) * 100)

# --- Decision Tree Model ---
# Define preprocessing recipe for decision tree
dt_recipe <- recipe(y ~ ., data = train) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())  # Remove zero-variance columns

# Train decision tree model with cross-validation
set.seed(123)
dt_model <- train(
  dt_recipe,
  data = train,
  method = "rpart",
  trControl = ctrl,
  metric = "ROC"
)

# Predict on test set
dt_predictions <- predict(dt_model, newdata = test)
dt_probabilities <- predict(dt_model, newdata = test, type = "prob")

# Evaluate decision tree performance
dt_conf_mat <- confusionMatrix(dt_predictions, test$y, positive = "yes")
print(dt_conf_mat)

# Extract and display decision tree metrics
dt_accuracy <- dt_conf_mat$overall["Accuracy"]
dt_kappa <- dt_conf_mat$overall["Kappa"]
dt_sensitivity <- dt_conf_mat$byClass["Sensitivity"]
dt_specificity <- dt_conf_mat$byClass["Specificity"]
dt_ppv <- dt_conf_mat$byClass["Pos Pred Value"]
dt_npv <- dt_conf_mat$byClass["Neg Pred Value"]

cat("\nDecision Tree Model Metrics:\n")
cat("Accuracy:", round(dt_accuracy, 4), "\n")
cat("Kappa:", round(dt_kappa, 4), "\n")
cat("Sensitivity:", round(dt_sensitivity, 4), "\n")
cat("Specificity:", round(dt_specificity, 4), "\n")
cat("PPV:", round(dt_ppv, 4), "\n")
cat("NPV:", round(dt_npv, 4), "\n")

# Plot ROC curve and calculate AUC for decision tree
dt_roc_obj <- roc(response = test$y, predictor = dt_probabilities$yes)
plot(dt_roc_obj, main = "ROC Curve - Decision Tree", col = "#31a354", lwd = 2)
dt_auc_value <- auc(dt_roc_obj)
cat("AUC:", round(dt_auc_value, 4), "\n")

# Visualize decision tree structure
rpart.plot(
  dt_model$finalModel,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree Visualization"
)

# --- Naive Bayes Model ---
# Ensure target is a factor
y_encoded <- factor(y_encoded)

# Combine encoded features and target
train_data <- cbind(X_encoded, y = y_encoded)

# Define cross-validation control for ROC metric
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train Naive Bayes model
set.seed(123)
nb_model <- train(
  x = train_data[, -ncol(train_data)],
  y = train_data$y,
  method = "naive_bayes",
  trControl = ctrl,
  metric = "ROC"
)

# Predict probabilities for Naive Bayes
nb_probabilities <- predict(nb_model, newdata = test, type = "prob")

# Plot ROC curve and calculate AUC for Naive Bayes
nb_roc_obj <- roc(response = test$y, predictor = nb_probabilities$yes)
plot(nb_roc_obj, main = "ROC Curve - Naive Bayes", col = "#ff7f0e", lwd = 2)
nb_auc_value <- auc(nb_roc_obj)
cat("AUC (Naive Bayes):", round(nb_auc_value, 4), "\n")

# Display Naive Bayes model summary
print(nb_model)

# --- Logistic Regression Model ---
# Train logistic regression model with cross-validation
set.seed(123)
log_model <- train(
  y ~ ., 
  data = train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  metric = "ROC"
)

# Predict on test set
predictions <- predict(log_model, newdata = test)
probabilities <- predict(log_model, newdata = test, type = "prob")

# Evaluate logistic regression performance
conf_mat <- confusionMatrix(predictions, test$y, positive = "yes")
print(conf_mat)

# Extract and display logistic regression metrics
accuracy <- conf_mat$overall["Accuracy"]
kappa <- conf_mat$overall["Kappa"]
sensitivity <- conf_mat$byClass["Sensitivity"]
specificity <- conf_mat$byClass["Specificity"]
ppv <- conf_mat$byClass["Pos Pred Value"]
npv <- conf_mat$byClass["Neg Pred Value"]

cat("\nModel Metrics:\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Kappa:", round(kappa, 4), "\n")
cat("Sensitivity:", round(sensitivity, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")
cat("PPV:", round(ppv, 4), "\n")
cat("NPV:", round(npv, 4), "\n")

# Plot ROC curve and calculate AUC for logistic regression
roc_obj <- roc(response = test$y, predictor = probabilities$yes)
plot(roc_obj, main = "ROC Curve - Logistic Regression", col = "#2c7fb8", lwd = 2)
auc_value <- auc(roc_obj)
cat("AUC:", round(auc_value, 4), "\n")

# --- Random Forest Model ---
# Train random forest model with cross-validation
set.seed(123)
rf_model <- train(
  y ~ ., 
  data = train,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5  # Number of hyperparameter combinations to try
)

# Predict on test set
rf_predictions <- predict(rf_model, newdata = test)
rf_probabilities <- predict(rf_model, newdata = test, type = "prob")

# Evaluate random forest performance
rf_conf_mat <- confusionMatrix(rf_predictions, test$y, positive = "yes")
print(rf_conf_mat)

# Extract and display random forest metrics
rf_accuracy <- rf_conf_mat$overall["Accuracy"]
rf_kappa <- rf_conf_mat$overall["Kappa"]
rf_sensitivity <- rf_conf_mat$byClass["Sensitivity"]
rf_specificity <- rf_conf_mat$byClass["Specificity"]
rf_ppv <- rf_conf_mat$byClass["Pos Pred Value"]
rf_npv <- rf_conf_mat$byClass["Neg Pred Value"]

cat("\nRandom Forest Model Metrics:\n")
cat("Accuracy:", round(rf_accuracy, 4), "\n")
cat("Kappa:", round(rf_kappa, 4), "\n")
cat("Sensitivity:", round(rf_sensitivity, 4), "\n")
cat("Specificity:", round(rf_specificity, 4), "\n")
cat("PPV:", round(rf_ppv, 4), "\n")
cat("NPV:", round(rf_npv, 4), "\n")

# Plot ROC curve and calculate AUC for random forest
rf_roc_obj <- roc(response = test$y, predictor = rf_probabilities$yes)
plot(rf_roc_obj, main = "ROC Curve - Random Forest", col = "#d95f02", lwd = 2)
rf_auc_value <- auc(rf_roc_obj)
cat("AUC:", round(rf_auc_value, 4), "\n")

# --- KNN Model ---
# Ensure consistent factor levels for target variable
train$y <- factor(train$y, levels = c("no", "yes"))
test$y <- factor(test$y, levels = c("no", "yes"))

# Train KNN model with cross-validation
knn_model <- train(
  y ~ .,
  data = train,
  method = "knn",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 10  # Tune number of neighbors (k)
)

# Display optimal k value
cat("Best k:", knn_model$bestTune$k, "\n")

# Plot model performance across k values
plot(knn_model)

# Predict on test set
knn_predictions <- predict(knn_model, newdata = test)
knn_probs <- predict(knn_model, newdata = test, type = "prob")

# Evaluate KNN performance
conf_matrix <- confusionMatrix(knn_predictions, test$y, positive = "yes")
print(conf_matrix)

# Plot ROC curve and calculate AUC for KNN
roc_obj <- roc(response = test$y, predictor = knn_probs$yes)
plot(roc_obj, col = "darkblue", lwd = 3, main = "ROC Curve - KNN")
auc_val <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_val, 4)), col = "darkblue", lwd = 3)

# Predict class labels for Naive Bayes
nb_predictions <- predict(nb_model, newdata = test)

# Evaluate Naive Bayes performance
nb_conf_mat <- confusionMatrix(nb_predictions, test$y, positive = "yes")
print(nb_conf_mat)

# --- Model Comparison ---
# Predict probabilities for all models
rf_probabilities <- predict(rf_model, newdata = test, type = "prob")$yes
knn_probabilities <- predict(knn_model, newdata = test, type = "prob")$yes
nb_probabilities <- predict(nb_model, newdata = test, type = "prob")$yes
dt_probabilities <- predict(dt_model, newdata = test, type = "prob")$yes
log_probabilities <- predict(log_model, newdata = test, type = "prob")$yes

# Create ROC objects for each model
rf_roc <- roc(response = test$y, predictor = rf_probabilities)
knn_roc <- roc(response = test$y, predictor = knn_probabilities)
nb_roc <- roc(response = test$y, predictor = nb_probabilities)
dt_roc <- roc(response = test$y, predictor = dt_probabilities)
log_roc <- roc(response = test$y, predictor = log_probabilities)

# Calculate AUC for each model
rf_auc <- auc(rf_roc)
knn_auc <- auc(knn_roc)
nb_auc <- auc(nb_roc)
dt_auc <- auc(dt_roc)
log_auc <- auc(log_roc)

# Plot combined ROC curves for model comparison
plot(rf_roc, col = "#ff7f0e", lwd = 2, main = "ROC Curves for Classification Models")
lines(knn_roc, col = "#2ca02c", lwd = 2)
lines(nb_roc, col = "#9467bd", lwd = 2)
lines(dt_roc, col = "#ff69b4", lwd = 2)
lines(log_roc, col = "#1f77b4", lwd = 2)

# Add legend with AUC values
legend("bottomright", 
       legend = c(
         paste("Random Forest (AUC =", round(rf_auc, 2), ")"),
         paste("KNN (AUC =", round(knn_auc, 2), ")"),
         paste("Naive Bayes (AUC =", round(nb_auc, 2), ")"),
         paste("Decision Tree (AUC =", round(dt_auc, 2), ")"),
         paste("Logistic Regression (AUC =", round(log_auc, 2), ")")
       ),
       col = c("#ff7f0e", "#2ca02c", "#9467bd", "#ff69b4", "#1f77b4"), 
       cex = 0.4,
       lwd = 2)

# Extract variable importance from random forest model
var_imp <- varImp(rf_model, scale = TRUE)

# Convert variable importance to data frame for plotting
var_imp_df <- data.frame(Variable = row.names(var_imp$importance), Importance = var_imp$importance$Overall)

# Sort by importance in descending order
var_imp_df <- var_imp_df[order(-var_imp_df$Importance), ]

# Plot variable importance
ggplot(var_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#ff7f0e") +  # Orange bars matching ROC plot
  coord_flip() +  # Flip coordinates for horizontal bars
  labs(title = "Variable Importance in Random Forest", x = "Variables", y = "Importance") +
  theme_minimal()