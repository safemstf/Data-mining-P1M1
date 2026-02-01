################################################################################
# CSCE 5380 Data Mining Project - Milestone 1
# German Credit Dataset Analysis
# Classifiers: Decision Tree, PART, and Ripper (JRip)
#
# IMPORTANT PREREQUISITES:
# 1. Java JDK must be installed (required for RWeka)
# 2. All R packages must be installed (see installation section below)
# 3. credit-g.csv must be in the working directory
#
# To run: source("german_credit_analysis.R")
################################################################################

################################################################################
# PACKAGE INSTALLATION (RUN ONCE)
################################################################################
# Uncomment and run this section if packages are not installed:
#
# install.packages(c("tidyverse", "mlbench", "rpart", "rpart.plot", 
#                    "caret", "FSelector", "Rtsne", "cluster"))
# 
# # Java-dependent packages (install AFTER installing Java JDK)
# install.packages("rJava")
# install.packages("RWekajars") 
# install.packages("RWeka")

################################################################################
# ENVIRONMENT CHECK
################################################################################

cat("\n")
cat("================================================================================\n")
cat("GERMAN CREDIT ANALYSIS - ENVIRONMENT CHECK\n")
cat("================================================================================\n\n")

# Function to check package installation
check_package <- function(pkg_name) {
  if (require(pkg_name, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg_name, "loaded successfully\n")
    return(TRUE)
  } else {
    cat("✗", pkg_name, "NOT FOUND\n")
    cat("  Install with: install.packages('", pkg_name, "')\n", sep = "")
    return(FALSE)
  }
}

# Check required packages
cat("Checking required packages...\n")
packages_ok <- TRUE

packages_ok <- packages_ok & check_package("tidyverse")
packages_ok <- packages_ok & check_package("mlbench")
packages_ok <- packages_ok & check_package("rpart")
packages_ok <- packages_ok & check_package("rpart.plot")
packages_ok <- packages_ok & check_package("caret")
packages_ok <- packages_ok & check_package("FSelector")
packages_ok <- packages_ok & check_package("Rtsne")
packages_ok <- packages_ok & check_package("cluster")

cat("\nChecking Java-dependent packages...\n")
java_ok <- TRUE

# Check rJava first
if (!check_package("rJava")) {
  java_ok <- FALSE
  cat("\n")
  cat("ERROR: rJava is not installed or Java is not configured properly.\n")
  cat("\n")
  cat("SOLUTION:\n")
  cat("1. Install Java JDK from https://adoptium.net/ or https://www.oracle.com/java/\n")
  cat("2. Set JAVA_HOME environment variable\n")
  cat("3. Restart R/RStudio\n")
  cat("4. Run: install.packages('rJava')\n")
  cat("\nSee README_INSTALLATION.md for detailed instructions.\n")
} else {
  # Test Java initialization
  tryCatch({
    .jinit()
    java_version <- .jcall("java/lang/System", "S", "getProperty", "java.version")
    cat("  Java version:", java_version, "\n")
  }, error = function(e) {
    cat("  WARNING: Java initialization failed:", conditionMessage(e), "\n")
    java_ok <<- FALSE
  })
}

java_ok <- java_ok & check_package("RWekajars")
java_ok <- java_ok & check_package("RWeka")

if (!packages_ok || !java_ok) {
  cat("\n")
  cat("================================================================================\n")
  cat("INSTALLATION REQUIRED\n")
  cat("================================================================================\n")
  cat("Some required packages are missing. Please install them before proceeding.\n")
  cat("See README_INSTALLATION.md for detailed installation instructions.\n")
  cat("================================================================================\n\n")
  stop("Missing required packages. Installation required.")
}

cat("\n✓ All packages loaded successfully!\n")
cat("✓ Java is properly configured\n")

################################################################################
# LOAD LIBRARIES
################################################################################

cat("\n")
cat("================================================================================\n")
cat("LOADING LIBRARIES\n")
cat("================================================================================\n\n")

suppressPackageStartupMessages({
  library(tidyverse)
  library(mlbench)
  library(rpart)
  library(rpart.plot)
  library(caret)
  library(RWeka)
  library(FSelector)
  library(Rtsne)
  library(cluster)
})

cat("All libraries loaded successfully.\n")

################################################################################
# WORKING DIRECTORY AND FILE CHECK
################################################################################

cat("\n")
cat("================================================================================\n")
cat("CHECKING WORKING DIRECTORY\n")
cat("================================================================================\n\n")

cat("Current working directory:", getwd(), "\n")
cat("Files in working directory:\n")
print(list.files())

# Check if credit-g.csv exists
if (!file.exists("credit-g.csv")) {
  cat("\n")
  cat("================================================================================\n")
  cat("ERROR: DATASET NOT FOUND\n")
  cat("================================================================================\n")
  cat("The file 'credit-g.csv' was not found in the working directory.\n")
  cat("\n")
  cat("SOLUTIONS:\n")
  cat("1. Copy credit-g.csv to:", getwd(), "\n")
  cat("2. OR set working directory: setwd('path/to/folder/with/csv')\n")
  cat("3. OR use full path in read.csv() command\n")
  cat("================================================================================\n\n")
  stop("Dataset file not found: credit-g.csv")
}

cat("\n✓ credit-g.csv found in working directory\n")

################################################################################
# DATA LOADING
################################################################################

cat("\n")
cat("================================================================================\n")
cat("LOADING GERMAN CREDIT DATASET\n")
cat("================================================================================\n\n")

# Load the German Credit dataset
credit_data <- read.csv("credit-g.csv", stringsAsFactors = TRUE)

cat("Dataset dimensions:", nrow(credit_data), "rows ×", ncol(credit_data), "columns\n")
cat("Number of samples:", nrow(credit_data), "\n")
cat("Number of features:", ncol(credit_data) - 1, "(plus 1 target variable)\n\n")

cat("First few rows of the dataset:\n")
print(head(credit_data, 3))

cat("\nDataset structure:\n")
str(credit_data)

cat("\nClass distribution:\n")
print(table(credit_data$class))
cat("\n")

################################################################################
# TASK 1: DATA PREPROCESSING
################################################################################

cat("\n")
cat("================================================================================\n")
cat("TASK 1: DATA PREPROCESSING\n")
cat("================================================================================\n\n")

# --------------------------------------------------
# PREPROCESSING OPERATION 1: Check for Missing Values
# --------------------------------------------------
cat("--- Preprocessing Operation 1: Missing Value Analysis ---\n\n")

missing_counts <- colSums(is.na(credit_data))
cat("Missing values per feature:\n")
print(missing_counts)

total_missing <- sum(missing_counts)
cat("\nTotal missing values:", total_missing, "\n")

# Justification (printed for documentation):
cat("\nJUSTIFICATION:\n")
cat("Missing values can significantly impact model performance and lead to biased results.\n")
cat("If missing values exist, we need to either impute them or remove affected samples.\n")

# Handle missing values if they exist
if(total_missing > 0) {
  cat("\nACTION: Removing rows with missing values...\n")
  credit_data <- credit_data %>% drop_na()
  cat("New dataset size:", nrow(credit_data), "samples\n")
} else {
  cat("\nACTION: No missing values detected. No action needed.\n")
}

# --------------------------------------------------
# PREPROCESSING OPERATION 2: Check for Duplicate Samples
# --------------------------------------------------
cat("\n--- Preprocessing Operation 2: Duplicate Detection ---\n\n")

duplicate_count <- sum(duplicated(credit_data))
cat("Number of duplicate rows:", duplicate_count, "\n")

cat("\nJUSTIFICATION:\n")
cat("Duplicate samples can artificially inflate model accuracy and lead to overfitting.\n")
cat("They provide no additional information and should be removed to ensure unbiased training.\n")

if(duplicate_count > 0) {
  cat("\nACTION: Removing duplicate samples...\n")
  credit_data <- credit_data %>% distinct()
  cat("New dataset size:", nrow(credit_data), "samples\n")
} else {
  cat("\nACTION: No duplicates detected. No action needed.\n")
}

# --------------------------------------------------
# PREPROCESSING OPERATION 3: Ensure Proper Data Types
# --------------------------------------------------
cat("\n--- Preprocessing Operation 3: Data Type Conversion ---\n\n")

cat("JUSTIFICATION:\n")
cat("Proper data types are essential for algorithms to process features correctly.\n")
cat("Categorical features must be factors, and numeric features should be numeric or integer types.\n\n")

# Check if all categorical columns are factors
cat("Converting categorical features to factors if needed...\n")
categorical_cols <- c("checking_status", "credit_history", "purpose", 
                      "savings_status", "employment", "personal_status",
                      "other_parties", "property_magnitude", "other_payment_plans",
                      "housing", "job", "own_telephone", "foreign_worker", "class")

conversion_count <- 0
for(col in categorical_cols) {
  if(!is.factor(credit_data[[col]])) {
    credit_data[[col]] <- as.factor(credit_data[[col]])
    cat("  Converted", col, "to factor\n")
    conversion_count <- conversion_count + 1
  }
}

if(conversion_count == 0) {
  cat("  All categorical features were already factors. No conversion needed.\n")
} else {
  cat("  Total conversions:", conversion_count, "\n")
}

# --------------------------------------------------
# PREPROCESSING OPERATION 4: Check for Class Imbalance
# --------------------------------------------------
cat("\n--- Preprocessing Operation 4: Class Imbalance Analysis ---\n\n")

cat("JUSTIFICATION:\n")
cat("Class imbalance can bias classifiers toward the majority class, resulting in poor\n")
cat("prediction accuracy for the minority class. Understanding the imbalance helps us\n")
cat("choose appropriate algorithms (like Ripper) and evaluation metrics.\n\n")

class_distribution <- table(credit_data$class)
class_proportions <- prop.table(class_distribution)

cat("Class distribution:\n")
print(class_distribution)
cat("\nClass proportions:\n")
print(round(class_proportions, 3))

imbalance_ratio <- max(class_proportions) / min(class_proportions)
cat("\nImbalance ratio:", round(imbalance_ratio, 2), ":1\n")

if(imbalance_ratio > 1.5) {
  cat("WARNING: Dataset shows class imbalance. Consider using Ripper which handles imbalanced data.\n")
} else {
  cat("INFO: Dataset is relatively balanced.\n")
}

# --------------------------------------------------
# PREPROCESSING OPERATION 5: Feature Scaling (if needed for t-SNE)
# --------------------------------------------------
cat("\n--- Preprocessing Operation 5: Identify Numeric Features ---\n\n")

cat("JUSTIFICATION:\n")
cat("Some algorithms and visualization methods (like t-SNE) work better with normalized data.\n")
cat("Identifying numeric features helps us apply appropriate preprocessing for visualization.\n\n")

numeric_cols <- sapply(credit_data, is.numeric)
numeric_features <- names(credit_data)[numeric_cols]
numeric_features <- numeric_features[numeric_features != "class"]

cat("Numeric features identified (", length(numeric_features), "):\n", sep = "")
print(numeric_features)

categorical_features <- setdiff(names(credit_data), c(numeric_features, "class"))
cat("\nCategorical features identified (", length(categorical_features), "):\n", sep = "")
print(categorical_features)

# --------------------------------------------------
# PREPROCESSING SUMMARY
# --------------------------------------------------
cat("\n--- Preprocessing Summary ---\n")
cat("Final dataset dimensions:", nrow(credit_data), "rows ×", ncol(credit_data), "columns\n")
cat("Final class distribution:\n")
print(table(credit_data$class))
cat("\n")

################################################################################
# TASK 1 PART 3: t-SNE VISUALIZATION
################################################################################

cat("\n")
cat("================================================================================\n")
cat("TASK 1 PART 3: t-SNE VISUALIZATION\n")
cat("================================================================================\n\n")

# Prepare data for t-SNE
cat("Preparing data for t-SNE visualization...\n")

# Create a copy for t-SNE
tsne_data <- credit_data

# Convert factors to numeric using model.matrix (one-hot encoding)
# Remove the class column temporarily
class_labels <- tsne_data$class
tsne_features <- tsne_data %>% select(-class)

# Create dummy variables for all categorical features
dummy_model <- dummyVars(" ~ .", data = tsne_features)
tsne_numeric <- predict(dummy_model, newdata = tsne_features)

cat("Numeric matrix for t-SNE dimensions:", nrow(tsne_numeric), "×", ncol(tsne_numeric), "\n")

# Run t-SNE
cat("\nRunning t-SNE dimensionality reduction...\n")
cat("This may take a few moments...\n\n")

set.seed(42)  # For reproducibility
tsne_result <- Rtsne(tsne_numeric, 
                     dims = 2,           # Reduce to 2 dimensions
                     perplexity = 30,    # Default perplexity
                     verbose = TRUE,
                     max_iter = 500)

# Create a data frame with t-SNE results
tsne_df <- data.frame(
  X = tsne_result$Y[, 1],
  Y = tsne_result$Y[, 2],
  Class = class_labels
)

# Create the t-SNE plot
cat("\nGenerating t-SNE visualization...\n")

tsne_plot <- ggplot(tsne_df, aes(x = X, y = Y, color = Class)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(
    title = "t-SNE Visualization of German Credit Dataset",
    subtitle = "2D projection of high-dimensional feature space",
    x = "t-SNE Dimension 1",
    y = "t-SNE Dimension 2",
    color = "Credit Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "right"
  ) +
  scale_color_manual(values = c("good" = "#2E7D32", "bad" = "#C62828"))

print(tsne_plot)

# Save the plot
ggsave("tsne_visualization.png", plot = tsne_plot, width = 10, height = 7, dpi = 300)
cat("✓ t-SNE plot saved as 'tsne_visualization.png'\n")

# Analysis of t-SNE plot
cat("\n--- t-SNE Insights ---\n")
cat("INTERPRETATION:\n")
cat("1. Class Separation: Examine if 'good' and 'bad' credit classes form distinct clusters.\n")
cat("   - Well-separated clusters indicate easier classification\n")
cat("   - Overlapping clusters suggest difficult classification\n\n")

cat("2. Cluster Density: Look at how tightly packed each class is.\n")
cat("   - Dense clusters indicate homogeneous class characteristics\n")
cat("   - Scattered points suggest high within-class variability\n\n")

cat("3. Boundary Clarity: Check if there's a clear decision boundary.\n")
cat("   - Clear boundaries favor simple models (Decision Trees)\n")
cat("   - Fuzzy boundaries may require more complex rules (Ripper)\n\n")

# Calculate silhouette score
cat("Calculating silhouette score for class separation...\n")
tsne_matrix <- as.matrix(tsne_df[, c("X", "Y")])
silhouette_score <- silhouette(as.numeric(tsne_df$Class), dist(tsne_matrix))
avg_silhouette <- mean(silhouette_score[, 3])

cat("Average Silhouette Score:", round(avg_silhouette, 3), "\n")
cat("Interpretation: Higher values (>0.5) indicate better class separation\n")
if(avg_silhouette > 0.5) {
  cat("  → Well-separated classes (easy classification)\n")
} else if(avg_silhouette > 0.2) {
  cat("  → Moderate separation (medium difficulty)\n")
} else {
  cat("  → Poor separation (difficult classification)\n")
}
cat("\n")

################################################################################
# DATA SPLITTING
################################################################################

cat("\n")
cat("================================================================================\n")
cat("DATA SPLITTING\n")
cat("================================================================================\n\n")

# Split data into training (70%) and testing (30%)
set.seed(3333)
inTrain <- createDataPartition(y = credit_data$class, p = 0.7, list = FALSE)
training <- credit_data %>% slice(inTrain)
testing <- credit_data %>% slice(-inTrain)

cat("Training set size:", nrow(training), "(70%)\n")
cat("Testing set size:", nrow(testing), "(30%)\n\n")
cat("Training class distribution:\n")
print(table(training$class))
cat("\nTesting class distribution:\n")
print(table(testing$class))
cat("\n")

################################################################################
# FEATURE SELECTION
################################################################################

cat("\n")
cat("================================================================================\n")
cat("FEATURE SELECTION\n")
cat("================================================================================\n\n")

# Chi-squared feature selection
cat("Performing Chi-squared feature selection...\n")

weights <- chi.squared(class ~ ., training)
weights_df <- as_tibble(weights, rownames = "feature") %>%
  arrange(desc(attr_importance))

cat("\nFeature importance rankings:\n")
print(weights_df)

# Select top features
num_features <- 13  # You can adjust this
subset <- cutoff.k(weights, num_features)
f <- as.simple.formula(subset, "class")

cat("\nSelected formula:\n")
print(f)
cat("\nNumber of selected features:", length(subset), "\n\n")

################################################################################
# TASK 2 PART 1: MODEL TRAINING
################################################################################

cat("\n")
cat("================================================================================\n")
cat("TASK 2 PART 1: MODEL TRAINING\n")
cat("================================================================================\n\n")

# Set up cross-validation
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# --------------------------------------------------
# 1. DECISION TREE (rpart)
# --------------------------------------------------
cat("--- Training Decision Tree Classifier ---\n")
cat("This may take a few moments...\n\n")

set.seed(3333)
dtree_fit <- train(
  form = f,
  data = training,
  method = "rpart",
  parms = list(split = "information"),
  control = rpart.control(minsplit = 2),
  trControl = trctrl,
  tuneLength = 5
)

cat("Decision Tree Model Results:\n")
print(dtree_fit)

cat("\nBest tuning parameters:\n")
print(dtree_fit$bestTune)

# Visualize the decision tree
cat("\nGenerating Decision Tree visualization...\n")
png("decision_tree_plot.png", width = 1200, height = 800)
prp(dtree_fit$finalModel, 
    box.palette = "Reds", 
    tweak = 1.2,
    main = "Decision Tree for German Credit Data")
dev.off()
cat("✓ Decision Tree plot saved as 'decision_tree_plot.png'\n\n")

# --------------------------------------------------
# 2. PART CLASSIFIER
# --------------------------------------------------
cat("--- Training PART Classifier ---\n")
cat("This may take a few moments...\n\n")

set.seed(3333)
part_fit <- train(
  form = f,
  data = training,
  method = "PART",
  tuneLength = 5,
  control = rpart.control(minsplit = 2),
  trControl = trctrl
)

cat("PART Model Results:\n")
print(part_fit)

cat("\nBest tuning parameters:\n")
print(part_fit$bestTune)
cat("\n")

# --------------------------------------------------
# 3. RIPPER (JRip) CLASSIFIER
# --------------------------------------------------
cat("--- Training Ripper (JRip) Classifier ---\n")
cat("Note: JRip may take several minutes to train...\n")
cat("Please be patient, this is normal for Ripper.\n\n")

set.seed(3333)
ripper_fit <- train(
  form = f,
  data = training,
  method = "JRip",
  tuneLength = 5,
  trControl = trctrl
)

cat("Ripper Model Results:\n")
print(ripper_fit)

cat("\nBest tuning parameters:\n")
print(ripper_fit$bestTune)
cat("\n")

################################################################################
# TASK 2 PART 2: GENERATE RULE BASES
################################################################################

cat("\n")
cat("================================================================================\n")
cat("TASK 2 PART 2: RULE BASES\n")
cat("================================================================================\n\n")

# --------------------------------------------------
# Decision Tree Rules
# --------------------------------------------------
cat("--- DECISION TREE RULES ---\n")
cat("(Generated from tree structure)\n\n")
print(dtree_fit$finalModel)
cat("\n")

# --------------------------------------------------
# PART Rules
# --------------------------------------------------
cat("--- PART CLASSIFIER RULES ---\n\n")
print(part_fit$finalModel)
cat("\n")

# --------------------------------------------------
# Ripper Rules
# --------------------------------------------------
cat("--- RIPPER (JRip) CLASSIFIER RULES ---\n\n")
print(ripper_fit$finalModel)
cat("\n")

################################################################################
# MODEL EVALUATION ON TEST SET
################################################################################

cat("\n")
cat("================================================================================\n")
cat("MODEL EVALUATION ON TEST SET\n")
cat("================================================================================\n\n")

# --------------------------------------------------
# Decision Tree Evaluation
# --------------------------------------------------
cat("--- Decision Tree Test Performance ---\n")
dtree_pred <- predict(dtree_fit, newdata = testing)
dtree_cm <- confusionMatrix(dtree_pred, testing$class)
print(dtree_cm)
cat("\n")

# --------------------------------------------------
# PART Evaluation
# --------------------------------------------------
cat("--- PART Classifier Test Performance ---\n")
part_pred <- predict(part_fit, newdata = testing)
part_cm <- confusionMatrix(part_pred, testing$class)
print(part_cm)
cat("\n")

# --------------------------------------------------
# Ripper Evaluation
# --------------------------------------------------
cat("--- Ripper (JRip) Test Performance ---\n")
ripper_pred <- predict(ripper_fit, newdata = testing)
ripper_cm <- confusionMatrix(ripper_pred, testing$class)
print(ripper_cm)
cat("\n")

################################################################################
# TASK 3: STATISTICAL COMPARISON WITH ANOVA
################################################################################

cat("\n")
cat("================================================================================\n")
cat("TASK 3: STATISTICAL COMPARISON (F-TEST)\n")
cat("================================================================================\n\n")

# --------------------------------------------------
# Collect accuracy values from all folds
# --------------------------------------------------
cat("--- Collecting Cross-Validation Accuracy Values ---\n\n")

# Decision Tree accuracies (30 folds: 10-fold CV x 3 repeats)
dtree_accuracies <- dtree_fit$resample$Accuracy
cat("Decision Tree CV accuracies (first 5):", head(dtree_accuracies, 5), "\n")
cat("Decision Tree mean accuracy:", round(mean(dtree_accuracies), 4), "\n")
cat("Decision Tree std dev:", round(sd(dtree_accuracies), 4), "\n\n")

# PART accuracies
part_accuracies <- part_fit$resample$Accuracy
cat("PART CV accuracies (first 5):", head(part_accuracies, 5), "\n")
cat("PART mean accuracy:", round(mean(part_accuracies), 4), "\n")
cat("PART std dev:", round(sd(part_accuracies), 4), "\n\n")

# Ripper accuracies
ripper_accuracies <- ripper_fit$resample$Accuracy
cat("Ripper CV accuracies (first 5):", head(ripper_accuracies, 5), "\n")
cat("Ripper mean accuracy:", round(mean(ripper_accuracies), 4), "\n")
cat("Ripper std dev:", round(sd(ripper_accuracies), 4), "\n\n")

# --------------------------------------------------
# Prepare data for ANOVA
# --------------------------------------------------
cat("--- Preparing data for ANOVA ---\n\n")

# Create a data frame for ANOVA
anova_data <- data.frame(
  Accuracy = c(dtree_accuracies, part_accuracies, ripper_accuracies),
  Classifier = factor(rep(c("Decision_Tree", "PART", "Ripper"), 
                          each = length(dtree_accuracies)))
)

cat("ANOVA data structure (first and last 3 rows):\n")
print(head(anova_data, 3))
cat("...\n")
print(tail(anova_data, 3))
cat("\n")

# --------------------------------------------------
# Perform One-Way ANOVA
# --------------------------------------------------
cat("--- Performing One-Way ANOVA (F-Test) ---\n\n")

anova_model <- aov(Accuracy ~ Classifier, data = anova_data)
anova_summary <- summary(anova_model)

cat("=== ANOVA TABLE ===\n")
print(anova_summary)
cat("\n")

# Extract F-statistic and p-value
f_statistic <- anova_summary[[1]]$`F value`[1]
p_value <- anova_summary[[1]]$`Pr(>F)`[1]

cat("F-statistic:", round(f_statistic, 4), "\n")
cat("P-value:", format(p_value, scientific = TRUE), "\n")
cat("Significance level: 0.05 (95% confidence)\n\n")

# --------------------------------------------------
# Interpret ANOVA Results
# --------------------------------------------------
if(p_value < 0.05) {
  cat("RESULT: The classifiers have SIGNIFICANTLY DIFFERENT accuracies (p < 0.05)\n")
  cat("Proceeding with Tukey HSD post-hoc test...\n\n")
  
  # --------------------------------------------------
  # Perform Tukey HSD Test
  # --------------------------------------------------
  cat("--- Tukey HSD Post-Hoc Test ---\n\n")
  
  tukey_result <- TukeyHSD(anova_model, conf.level = 0.95)
  print(tukey_result)
  
  cat("\n=== INTERPRETATION ===\n")
  cat("Pairwise comparisons:\n")
  cat("- If 'p adj' < 0.05: The pair has significantly different accuracies\n")
  cat("- If 'p adj' >= 0.05: No significant difference between the pair\n\n")
  
  # Identify the best classifier
  mean_accuracies <- aggregate(Accuracy ~ Classifier, data = anova_data, mean)
  mean_accuracies <- mean_accuracies[order(-mean_accuracies$Accuracy), ]
  
  cat("Mean accuracies by classifier (ranked):\n")
  print(mean_accuracies)
  
  cat("\n*** BEST CLASSIFIER: ", as.character(mean_accuracies$Classifier[1]), " ***\n", sep = "")
  cat("    Mean Accuracy: ", round(mean_accuracies$Accuracy[1], 4), "\n\n")
  
  # Visualize Tukey results
  png("tukey_hsd_plot.png", width = 800, height = 600)
  plot(tukey_result, las = 1, main = "Tukey HSD: 95% Confidence Intervals")
  dev.off()
  cat("✓ Tukey HSD plot saved as 'tukey_hsd_plot.png'\n\n")
  
} else {
  cat("RESULT: The classifiers have NO SIGNIFICANT DIFFERENCE in accuracies (p >= 0.05)\n")
  cat("All three classifiers perform similarly on this dataset.\n\n")
  
  # Still show mean accuracies for reference
  mean_accuracies <- aggregate(Accuracy ~ Classifier, data = anova_data, mean)
  mean_accuracies <- mean_accuracies[order(-mean_accuracies$Accuracy), ]
  
  cat("Mean accuracies by classifier (for reference):\n")
  print(mean_accuracies)
  cat("\n")
}

# --------------------------------------------------
# Visualize Accuracy Distributions
# --------------------------------------------------
cat("--- Creating Accuracy Distribution Visualizations ---\n\n")

# Boxplot
accuracy_boxplot <- ggplot(anova_data, aes(x = Classifier, y = Accuracy, fill = Classifier)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.3) +
  labs(
    title = "Accuracy Distribution by Classifier",
    subtitle = "10-fold Cross-Validation with 3 Repeats (30 folds)",
    x = "Classifier",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "none"
  ) +
  scale_fill_brewer(palette = "Set2")

print(accuracy_boxplot)
ggsave("accuracy_boxplot.png", plot = accuracy_boxplot, width = 10, height = 7, dpi = 300)
cat("✓ Boxplot saved as 'accuracy_boxplot.png'\n")

# Violin plot
accuracy_violin <- ggplot(anova_data, aes(x = Classifier, y = Accuracy, fill = Classifier)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.1, alpha = 0.5) +
  labs(
    title = "Accuracy Distribution by Classifier (Violin Plot)",
    subtitle = "Shows density of accuracy values across folds",
    x = "Classifier",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "none"
  ) +
  scale_fill_brewer(palette = "Pastel1")

print(accuracy_violin)
ggsave("accuracy_violin.png", plot = accuracy_violin, width = 10, height = 7, dpi = 300)
cat("✓ Violin plot saved as 'accuracy_violin.png'\n\n")

################################################################################
# FINAL SUMMARY
################################################################################

cat("\n")
cat("================================================================================\n")
cat("PROJECT SUMMARY\n")
cat("================================================================================\n\n")

cat("=== Dataset Information ===\n")
cat("Total samples:", nrow(credit_data), "\n")
cat("Training samples:", nrow(training), "\n")
cat("Testing samples:", nrow(testing), "\n")
cat("Number of features:", ncol(credit_data) - 1, "\n")
cat("Selected features:", length(subset), "\n\n")

cat("=== Test Set Performance ===\n")
cat("Decision Tree Accuracy:", round(dtree_cm$overall['Accuracy'], 4), "\n")
cat("PART Accuracy:", round(part_cm$overall['Accuracy'], 4), "\n")
cat("Ripper Accuracy:", round(ripper_cm$overall['Accuracy'], 4), "\n\n")

cat("=== Cross-Validation Performance (Mean ± SD) ===\n")
cat("Decision Tree:", round(mean(dtree_accuracies), 4), "±", round(sd(dtree_accuracies), 4), "\n")
cat("PART:", round(mean(part_accuracies), 4), "±", round(sd(part_accuracies), 4), "\n")
cat("Ripper:", round(mean(ripper_accuracies), 4), "±", round(sd(ripper_accuracies), 4), "\n\n")

cat("=== Statistical Test Results ===\n")
cat("F-statistic:", round(f_statistic, 4), "\n")
cat("P-value:", format(p_value, scientific = TRUE), "\n")
if(p_value < 0.05) {
  cat("Conclusion: Classifiers differ significantly\n")
  cat("Best classifier:", as.character(mean_accuracies$Classifier[1]), "\n")
} else {
  cat("Conclusion: No significant difference between classifiers\n")
}
cat("\n")

cat("=== Generated Files ===\n")
cat("1. tsne_visualization.png - t-SNE plot\n")
cat("2. decision_tree_plot.png - Decision tree visualization\n")
cat("3. accuracy_boxplot.png - Accuracy comparison boxplot\n")
cat("4. accuracy_violin.png - Accuracy comparison violin plot\n")
if(p_value < 0.05) {
  cat("5. tukey_hsd_plot.png - Tukey HSD confidence intervals\n")
}
cat("\n")

cat("================================================================================\n")
cat("✓✓✓ PROJECT COMPLETED SUCCESSFULLY! ✓✓✓\n")
cat("================================================================================\n\n")

cat("All visualizations have been saved to:", getwd(), "\n")
cat("Results have been printed to the console.\n")
cat("You can now:\n")
cat("1. Review the console output for detailed results\n")
cat("2. View the generated PNG files for visualizations\n")
cat("3. Copy results into your project report\n\n")