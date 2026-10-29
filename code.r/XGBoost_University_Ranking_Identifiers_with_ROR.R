# ============================================================
# XGBOOST MODEL FOR UNIVERSITY RANKING RECOMMENDATIONS 
# Objective: Predict Top, High, Middle, and Low categories
# IDENTIFIERS
# ============================================================

# STEP 0: INSTALL REQUIRED PACKAGES
# ------------------------------------------------------------
# Run this section only once if the packages are not installed.

# install.packages(c(
#   "readxl", "dplyr", "xgboost", "caret", "data.table",
#   "readr", "Matrix", "ggplot2", "pROC","tidyr"
# ))

# STEP 1: LOAD LIBRARIES
# ------------------------------------------------------------

library(readxl)   # Import Excel files
library(dplyr)    # Data manipulation
library(xgboost)  # XGBoost machine learning
library(caret)    # Train-test splitting
library(readr)    # Export CSV files
library(Matrix)   # Matrix operations
library(ggplot2)  # Plotting
library(tidyr)    # For reshaping data

# STEP 2: LOAD THE DATASET
# ------------------------------------------------------------

# Load the Excel file and select the required worksheet.
DATASAMPLING <- read_excel(
  "C:/Users/sanju/Desktop/Project2/DATASAMPLING_sparql.xlsx",
  sheet = 'with_ror'
)

# Inspect the dataset.
View(DATASAMPLING)
str(DATASAMPLING)
summary(DATASAMPLING)

# STEP 3: PREPARE THE IDENTIFIER DATASET
# ------------------------------------------------------------

# Select the first 27 columns containing the university identifier
# and the 26 predictor variables.
Identifier <- DATASAMPLING[, 1:27]

# Add the original ranking category.
Identifier$Ranked <- DATASAMPLING$Ranked

# Assign meaningful column names.
colnames(Identifier) <- c("University (QID)",
  "Iden1", "Iden2", "Iden3", "Iden4", "Iden5", "Iden6", "Iden7",
  "Iden8", "Iden9", "Iden10", "Iden11", "Iden12", "Iden13",
  "Iden14", "Iden15", "Iden16", "Iden17", "Iden18", "Iden19",
  "Iden20", "Iden21", "Iden22", "Iden23", "Iden24", "Iden25",
  "Iden26","Ranked"
)

# Inspect the renamed dataset.
View(Identifier)
str(Identifier)

# STEP 4: CLEAN THE DATA
# ------------------------------------------------------------

# View structure and Cleaning data
any(is.na(Identifier))  # check any missing values
# Count missing values in the dataset.
sum(is.na(Identifier))

# Remove rows containing missing values.
data <- na.omit(Identifier)

# See full structure
str(data)     
head(data)
View(data)
summary(data) # Provide summary of the dataset

# STEP 5: Data Preparation
# ------------------------------------------------------------
# Convert categorical variables

dataRank<-data # to keep the original ranked words
data$Ranked <- factor(data$Ranked,
                            levels = c("Top", "High", "Middle", "Low"),
                            labels = c(1, 2, 3, 4))

any(is.na(data))  # check any missing values

# Remove unnecessary columns
# Remove ID and date (not useful for prediction)
data_model <- data %>%
  select(-`University (QID)`)

# STEP 6: Create Target Variable (VERY IMPORTANT)
# ------------------------------------------------------------
# Create grouped ranked categories

data_model <- data_model %>%
 mutate(
    Top = ifelse(Ranked==1, 1, 0),
    
    High = ifelse( Ranked==2, 1, 0),
    
    Middle = ifelse( Ranked==3, 1,0),
    
    Low = ifelse( Ranked==4, 1,0)
  )

# STEP 7: SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------ 
# Train-Test Split
# Prepare Data for XGBoost

# Train Models (One per class)
# >>> Top-Ranked class <<<
set.seed(123)   # To reproduce same randomness
trainIndex <- createDataPartition(data_model$Top, p = 0.8, list = FALSE)
train <- data_model[trainIndex, ]
test  <- data_model[-trainIndex, ] 

unique(train$Top) # checking for only binary (e.g. 0, 1)

# Remove target leakage (VERY IMPORTANT)
train_features <- train %>%
  select(-Ranked, -Top, -High, -Middle, -Low)
test_features <- test %>%
  select(-Ranked, -Top, -High, -Middle, -Low)

# Then create matrix
train_matrix <- model.matrix(~ . -1, data = train_features)
test_matrix  <- model.matrix(~ . -1, data = test_features) 

# Fix 4 (MOST LIKELY ROOT CAUSE) 
# if you see (e.g. 0 1000) No variation → model cannot train
table(train$Top)
table(test$Top)

# Ensure the label is integer
train$Top <- as.integer(train$Top)

# Compute class weights for imbalance
# Count of each class
table(train$Top)

# Compute scale_pos_weight
neg <- sum(train$Top == 0)
pos <- sum(train$Top == 1)
scale_pos_weight <- neg / pos
scale_pos_weight

# Create DMatrix
dtrain <- xgb.DMatrix(
  data = train_matrix,
  label = train$Top
)

# Set parameters
params <- list(
  objective = "binary:logistic",  # binary classification
  eval_metric = "auc",            # evaluation metric
  scale_pos_weight = scale_pos_weight,
  max_depth = 6,                  # you can tune later
  eta = 0.1,                      # learning rate
  subsample = 0.8,                # row sampling
  colsample_bytree = 0.8          # feature sampling
)

# Train the model
Top_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 1
)

# Make Predictions for Top-Ranked class
# Predict probabilities
Top_pred <- predict(Top_model, test_matrix)


# Train Models (One per class)
# >>> High-ranked class <<<
set.seed(123) # For Reproducibility
trainIndex <- createDataPartition(data_model$High, p = 0.8, list = FALSE)
train <- data_model[trainIndex, ]
test  <- data_model[-trainIndex, ]

unique(train$High) # checking for only binary (e.g. 0, 1)

# Remove target leakage (VERY IMPORTANT)
train_features <- train %>%
  select(-Ranked, -Top, -High, -Middle, -Low)
test_features <- test %>%
  select(-Ranked, -Top, -High, -Middle, -Low)

# Then create matrix
train_matrix <- model.matrix(~ . -1, data = train_features)
test_matrix  <- model.matrix(~ . -1, data = test_features)

# Fix 4 (MOST LIKELY ROOT CAUSE) 
# if you see (e.g. 0 1000) No variation → model cannot train
table(train$High)
table(test$High)

# Ensure the label is integer
train$High <- as.integer(train$High)

# Compute class weights for imbalance
# Count of each class
table(train$High)

# Compute scale_pos_weight
neg <- sum(train$High == 0)
pos <- sum(train$High == 1)
scale_pos_weight <- neg / pos
scale_pos_weight

# Create DMatrix
dtrain <- xgb.DMatrix(
  data = train_matrix,
  label = train$High
)
# Set parameters
params <- list(
  objective = "binary:logistic",  # binary classification
  eval_metric = "auc",            # evaluation metric
  scale_pos_weight = scale_pos_weight,
  max_depth = 6,                  # you can tune later
  eta = 0.1,                      # learning rate
  subsample = 0.8,                # row sampling
  colsample_bytree = 0.8          # feature sampling
)

# Train the model
High_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 1
)

# Make Predictions for High-ranked model
# Predict probabilities
High_pred <- predict(High_model, test_matrix)

# Train Models (One per class)
# >>> Middle-ranked class <<<
set.seed(123)
trainIndex <- createDataPartition(data_model$Middle, p = 0.8, list = FALSE)
train <- data_model[trainIndex, ]
test  <- data_model[-trainIndex, ]

unique(train$Middle) # checking for only binary (e.g. 0, 1)

# Remove target leakage (VERY IMPORTANT)
train_features <- train %>%
  select(-Ranked, -Top, -High, -Middle, -Low)
test_features <- test %>%
  select(-Ranked, -Top, -High, -Middle, -Low)

# Then create matrix
train_matrix <- model.matrix(~ . -1, data = train_features)
test_matrix  <- model.matrix(~ . -1, data = test_features)

# Fix 4 (MOST LIKELY ROOT CAUSE) 
# if you see (e.g. 0 1000) No variation → model cannot train
table(train$Middle)
table(test$Middle) 

# Ensure the label is integer
train$Middle <- as.integer(train$Middle)

# Compute class weights for imbalance
# Count of each class
table(train$Middle)

# Compute scale_pos_weight
neg <- sum(train$Middle == 0)
pos <- sum(train$Middle == 1)
scale_pos_weight <- neg / pos
scale_pos_weight

# Create DMatrix
dtrain <- xgb.DMatrix(
  data = train_matrix,
  label = train$Middle
)

# Set parameters
params <- list(
  objective = "binary:logistic",  # binary classification
  eval_metric = "auc",            # evaluation metric
  scale_pos_weight = scale_pos_weight,
  max_depth = 6,                  # you can tune later
  eta = 0.1,                      # learning rate
  subsample = 0.8,                # row sampling
  colsample_bytree = 0.8          # feature sampling
)

# Train the model
Middle_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 1
)

# Make Predictions for Middle-Ranked class
# Predict probabilities
Middle_pred <- predict(Middle_model, test_matrix)

# Train Models (One per product class)
# >>> Low-ranked class <<<
set.seed(123)
trainIndex <- createDataPartition(data_model$Low, p = 0.8, list = FALSE)
train <- data_model[trainIndex, ]
test  <- data_model[-trainIndex, ]

unique(train$Low) # checking for only binary (e.g. 0, 1)

# Remove target leakage (VERY IMPORTANT)
train_features <- train %>%
  select(-Ranked, -Top, -High, -Middle, -Low)
test_features <- test %>%
  select(-Ranked, -Top, -High, -Middle, -Low)

# Then create matrix
train_matrix <- model.matrix(~ . -1, data = train_features)
test_matrix  <- model.matrix(~ . -1, data = test_features)

# Fix 4 (MOST LIKELY ROOT CAUSE) 
# if you see (e.g. 0 1000) No variation → model cannot train
table(train$Low)
table(test$Low)

# Ensure the label is integer
train$Low <- as.integer(train$Low)

# Compute class weights for imbalance
# Count of each class
table(train$Low)

# Compute scale_pos_weight
neg <- sum(train$Low == 0)
pos <- sum(train$Low == 1)
scale_pos_weight <- neg / pos
scale_pos_weight

# Create DMatrix
dtrain <- xgb.DMatrix(
  data = train_matrix,
  label = train$Low
)

# Set parameters
params <- list(
  objective = "binary:logistic",  # binary classification
  eval_metric = "auc",            # evaluation metric
  scale_pos_weight = scale_pos_weight,
  max_depth = 6,                  # you can tune later
  eta = 0.1,                      # learning rate
  subsample = 0.8,                # row sampling
  colsample_bytree = 0.8          # feature sampling
)

# Train the model
Low_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 1
)

# Make Predictions for Low-ranked
# Predict probabilities
Low_pred <- predict(Low_model, test_matrix)

# STEP 8: Combine Predictions
# Create dataframe of predictions
predictions <- data.frame(
  Top = Top_pred,
  High = High_pred,
  Middle = Middle_pred,
  Low = Low_pred
)

# STEP 9: Recommend Top 4 classes
# Function to get top 4 classes
get_top4 <- function(row) {
  names(sort(row, decreasing = TRUE))[1:4]
}

# Apply function row-wise
top4_recommendations <- t(apply(predictions, 1, get_top4))

# Convert to dataframe
top4_recommendations <- as.data.frame(top4_recommendations)

colnames(top4_recommendations) <- c("Top1", "Top2", "Top3","Top4")

# STEP 10: Final Output
# Combine with test data
final_output <- cbind(test, top4_recommendations)

head(final_output)

#STEP 11: Explainability 

# Feature importance Top-Ranked class
importance_matrix <- xgb.importance(model = Top_model)
xgb.plot.importance(importance_matrix)
xgb.plot.importance(
  importance_matrix,
  measure = "Gain",
  xlab = "Feature Importance (Gain)", 
  main = "Top-Ranking" , cex = 1
)

# Feature importance High-Ranked class
importance_matrix <- xgb.importance(model = High_model)
xgb.plot.importance(importance_matrix)
xgb.plot.importance(
  importance_matrix,
  measure = "Gain",
  xlab = "Feature Importance (Gain)",
  main = "High-Ranking", cex = 0.8
)

# Feature importance Middle-Ranked class
importance_matrix <- xgb.importance(model = Middle_model)
xgb.plot.importance(importance_matrix)
xgb.plot.importance(
  importance_matrix,
  measure = "Gain",
  xlab = "Feature Importance (Gain)",
  main = "Middle-Ranking", cex = 0.8
)

# Feature importance Low-Ranked class
importance_matrix <- xgb.importance(model = Low_model)
xgb.plot.importance(importance_matrix)
xgb.plot.importance(
  importance_matrix,
  measure = "Gain",
  xlab = "Feature Importance (Gain)",
  main = "Low-Ranking", cex = 0.8
)

# Combining 
Comdata<-dataRank[-trainIndex,]
resultwithRORIden <- cbind(Comdata, top4_recommendations)

# Export a data frame to Excel
library(readr)
write_csv(resultwithRORIden, "C:/Users/sanju/Downloads/resultwithRORIden.csv")

# STEP 12: CLEAR THE R ENVIRONMENT
# ------------------------------------------------------------

rm(list = ls())  # Remove all objects from the environment
cat("\014")      # Clear the console
gc()             # Release unused memory
