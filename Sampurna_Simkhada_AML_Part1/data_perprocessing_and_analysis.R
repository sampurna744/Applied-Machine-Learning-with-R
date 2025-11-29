# Load necessary libraries
library(mice)        # For missing value imputation
library(ggplot2)     # For data visualization
library(corrplot)    # For correlation matrix
library(FactoMineR)  # For PCA analysis
library(psych)       # For descriptive statistics

# Load the dataset
bank_data <- read.csv("bank marketing campaign.csv", header = TRUE)

# Inspect dataset structure and summary
str(bank_data)      # Check structure of dataset
summary(bank_data)  # Summary statistics

# Check for missing values
missing_values_by_column <- colSums(is.na(bank_data)) / nrow(bank_data) * 100
print(missing_values_by_column)

# Handle missing values based on average missing percentage
avg_missing <- mean(missing_values_by_column)
print(paste("Average missing percentage:", avg_missing))

if (avg_missing > 1) {
  # If average missing percentage >1%, use MICE for imputation
  imputed_data <- mice(bank_data, m = 5, method = "pmm", seed = 123)
  clean_data <- complete(imputed_data)
  print("Missing values imputed using MICE.")
} else {
  # Otherwise, remove rows with missing values
  clean_data <- na.omit(bank_data)
  print("Rows with missing values removed.")
}

# Verify missing values after cleaning
print(colSums(is.na(clean_data)) / nrow(clean_data) * 100)

# Identify numeric columns for analysis
numeric_columns <- which(sapply(clean_data, is.numeric))
numeric_data <- clean_data[, numeric_columns]

# Function to detect outliers using IQR method
detect_outliers_boxplot <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(which(x < lower_bound | x > upper_bound))
}

# Compute and visualize correlation matrix
corr_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(corr_matrix, method = "color", title = "Correlation Matrix")

# Detect outliers for each numeric column
for (col in names(numeric_data)) {
  outliers <- detect_outliers_boxplot(numeric_data[[col]])
  cat("Column:", col, "- Number of outliers:", length(outliers), "\n")
}

# Histogram for 'balance' variable
ggplot(clean_data, aes(x = balance)) +
  geom_histogram(binwidth = 500, fill = "black", color = "yellow") +
  labs(title = "Distribution of Balance", x = "Balance", y = "Count") +
  theme_minimal()

# Bar plot for job roles
ggplot(clean_data, aes(x = job)) +
  geom_bar(fill = "#404080", color = "black") +
  labs(title = "Distribution of Job Roles", x = "Job Role", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Perform PCA Analysis
pca_data <- clean_data[, numeric_columns]
pca_result <- PCA(pca_data, scale.unit = TRUE, graph = FALSE)

df_pca <- data.frame(PC1 = pca_result$ind$coord[, 1], PC2 = pca_result$ind$coord[, 2])
ggplot(df_pca, aes(x = PC1, y = PC2)) +
  geom_point(alpha = 0.5) +
  ggtitle("PCA of Bank Marketing Campaign Data") +
  xlab("Principal Component 1") +
  ylab("Principal Component 2") +
  theme_minimal()

# Display descriptive statistics
describe(numeric_data)

# Standardize numeric data
data_scaled <- scale(numeric_data)

describe(data_scaled)

# Boxplot of duration by age groups and yes/no subscription
ggplot(clean_data, aes(x = cut(age, breaks = seq(15, 95, 10)), y = duration, fill = y)) +
  geom_boxplot() +
  labs(title = "Call Duration by Age Group and Subscription", 
       x = "Age Group", y = "Duration (seconds)")
