# ============================================================
# DECISION TREE MODEL WITHOUTH ROR
# University Ranking Prediction
#
# Analysis 1: Identifier Analysis
# Analysis 2: Statement Analysis
#
# Target variable: Ranked
# Categories: Top, High, Middle, Low
# Algorithm: Decision Tree (rpart)
# ============================================================

# STEP 0: INSTALL PACKAGES
# ------------------------------------------------------------
# Run this section once if the packages are not installed.

# install.packages(c("readxl", "dplyr", "caret",
#                    "rpart", "rpart.plot"))

# STEP 1: LOAD LIBRARIES
# ------------------------------------------------------------

library(readxl)      # Import Excel files
library(dplyr)      # Data manipulation and cleaning
library(caret)      # Train and evaluate machine learning models
library(rpart)      # Build decision tree models
library(rpart.plot) # Visualize decision trees

# STEP 2: LOAD DATASET
# ------------------------------------------------------------

DATASAMPLING <- read_excel(
  "C:/Users/sanju/Desktop/Project2/DATASAMPLING_sparql.xlsx",
  sheet = 'without_ror'
)

# Inspect the original dataset.
View(DATASAMPLING)
str(DATASAMPLING)
print(dim(DATASAMPLING))

# STEP 3: REMOVE UNNECESSARY COLUMNS
# ------------------------------------------------------------

#remove first column
DATASAMPLING$`university` <- NULL
DATASAMPLING$`ROR`<- NULL

# Check the remaining columns.
str(DATASAMPLING)
print(dim(DATASAMPLING)) 

# STEP 4: SEPARATE IDENTIFIER AND STATEMENT DATASETS
# ------------------------------------------------------------

# Identifier dataset: first 25 predictor columns.
Identifier <- DATASAMPLING[, 1:25];

# Add the target variable.
Identifier$Ranked <- DATASAMPLING$Ranked 

# Inspect the original dataset.
View(Identifier)
str(Identifier)
print(dim(Identifier))

# Statement dataset: columns 26 to 41.
Statement <- DATASAMPLING[, 26:41]

# Add the target variable.
Statement$Ranked <- DATASAMPLING$Ranked

# Inspect the original dataset.
View(Statement)
str(Statement)
print(dim(Statement))

# STEP 5: CONVERT IDENTIFIER VARIABLES TO CATEGORICAL DATA
# ------------------------------------------------------------

# These variables are binary indicators:
# 1 = Yes; 0 = No.

#Identifier Analysis
Identifier$Ranked <- as.factor(Identifier$Ranked) #convert to factor
Identifier$`VIAF` <- as.factor(ifelse(Identifier$`VIAF`==1,"Yes","No"))
Identifier$`ISNI` <- as.factor(ifelse(Identifier$`ISNI`==1,"Yes","No"))
Identifier$`NLI` <- as.factor(ifelse(Identifier$`NLI`==1,"Yes","No"))
Identifier$`LOC` <- as.factor(ifelse(Identifier$`LOC`==1,"Yes","No"))
#Identifier$`ROR ID` <- as.factor(ifelse(Identifier$`ROR ID`==1,"Yes","No"))
Identifier$`IdRef` <- as.factor(ifelse(Identifier$`IdRef`==1,"Yes","No"))
Identifier$`ARWU` <- as.factor(ifelse(Identifier$`ARWU`==1,"Yes","No"))
Identifier$`Freebase` <- as.factor(ifelse(Identifier$`Freebase`==1,"Yes","No"))
Identifier$`GRID` <- as.factor(ifelse(Identifier$`GRID`==1,"Yes","No"))
Identifier$`Microsoft` <- as.factor(ifelse(Identifier$`Microsoft`==1,"Yes","No"))
Identifier$`OpenFunder` <- as.factor(ifelse(Identifier$`OpenFunder`==1,"Yes","No"))
Identifier$`QS` <- as.factor(ifelse(Identifier$`QS`==1,"Yes","No"))
Identifier$`Ringgold` <- as.factor(ifelse(Identifier$`Ringgold`==1,"Yes","No"))
Identifier$`THE`<-as.factor(ifelse(Identifier$`THE`==1,"Yes","No"))
Identifier$`UMultirank` <- as.factor(ifelse(Identifier$`UMultirank`==1,"Yes","No"))
Identifier$`BNF` <- as.factor(ifelse(Identifier$`BNF`==1,"Yes","No"))
Identifier$`Quora` <- as.factor(ifelse(Identifier$`Quora`==1,"Yes","No"))
Identifier$`GMaps` <- as.factor(ifelse(Identifier$`GMaps`==1,"Yes","No"))
Identifier$`GND` <- as.factor(ifelse(Identifier$`GND`==1,"Yes","No"))
Identifier$`NLCR` <- as.factor(ifelse(Identifier$`NLCR`==1,"Yes","No"))
Identifier$`SNARC` <- as.factor(ifelse(Identifier$`SNARC`==1,"Yes","No"))
#Identifier$`WorldCat Identities ID (superseded)` <- as.factor(ifelse(Identifier$`WorldCat Identities ID (superseded)`==1,"Yes","No"))
Identifier$`Twitter` <- as.factor(ifelse(Identifier$`Twitter`==1,"Yes","No"))
Identifier$`GoogleScholar` <- as.factor(ifelse(Identifier$`GoogleScholar`==1,"Yes","No"))
Identifier$`Facebook` <- as.factor(ifelse(Identifier$`Facebook`==1,"Yes","No"))
Identifier$`LibrariesAU` <- as.factor(ifelse(Identifier$`LibrariesAU`==1,"Yes","No"))
Identifier$`Subreddit` <- as.factor(ifelse(Identifier$`Subreddit`==1,"Yes","No"))

# STEP 6: TRAIN AND EVALUATE THE IDENTIFIER DECISION TREE
# ------------------------------------------------------------

# Set the random seed for reproducibility.
set.seed(123) 

# STEP 7: SPLIT IDENTIFIERS DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------
# Split dataset into training (80%) and testing (20%)

trainIndex <- createDataPartition(
  Identifier$Ranked,
  p = 0.8,
  list = FALSE
)

trainData <- Identifier[trainIndex, ]
testData <- Identifier[-trainIndex, ]

# STEP 8: BUILD THE IDENTIFIER DECISION TREE
# ------------------------------------------------------------

# Train a classification tree using Gini impurity.
tree_model <- rpart(
  Ranked ~ .,
  data = trainData,
  method = "class",
  control = rpart.control(minsplit = 7, cp = 0.007)
)

# Display tree details.
print(tree_model)
summary(tree_model)

# Display the complexity parameter table.
printcp(tree_model)

# Select the complexity parameter with minimum cross-validated error.
best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
best_cp

# Prune the tree to reduce unnecessary branches.
pruned_tree <- prune(tree_model, cp = best_cp)

# Display the pruned tree.
print(pruned_tree)
summary(pruned_tree)

# STEP 9: PLOT THE IDENTIFIER DECISION TREE
# ------------------------------------------------------------
#Plot the tree (very helpful)
rpart.plot(tree_model,cex = 0.7)
rpart.plot(tree_model,
           cex = 0.8,
           #tweak = 1.2,
           type = 0,
           extra = 0,
           fallen.leaves = TRUE,
           branch.type = 0,
           box.palette = "RdYlGn",
           legend.x = NA,
           shadow.col = "gray",
           nn = FALSE,)

# STEP 10: PREDICT IDENTIFIER RANKING
# ------------------------------------------------------------
# Predict using test data
predictions <- predict(
  tree_model,
  testData,
  type = "class"
)

# STEP 11: EVALUATE IDENTIFIER MODEL
# ------------------------------------------------------------
# Generate confusion matrix
cm <- confusionMatrix(
  predictions,
  testData$Ranked
)

# Display the confusion matrix and overall results.
print(cm)

# Extract overall performance measures.
accuracy <- cm$overall['Accuracy']
kappa <- cm$overall["Kappa"]
print(accuracy)
print(kappa)

# Extract class-wise precision, recall, and specificity.
precision <- cm$byClass[, "Pos Pred Value"]
recall <- cm$byClass[, "Sensitivity"]
specificity <- cm$byClass[, "Specificity"]
print(precision)
print(recall)
print(specificity)

# Calculate F1-score safely.
f1_score <- 2 * (precision * recall) / (precision + recall)
print(f1_score)

table(trainData$Ranked) # to check the number of training data
table(testData$Ranked) # to check the number of testing data


# ============================================================
# PART 2: STATEMENT ANALYSIS
# ============================================================

# STEP 12: CONVERT STATEMENT VARIABLES TO CATEGORICAL DATA
# ------------------------------------------------------------

# Binary statement indicators:
# 1 = Yes; 0 = No.

# Statement Analyisis
Statement$Ranked <- as.factor(Statement$Ranked) #convert to factor
Statement$`instanceOf` <- as.factor(ifelse(Statement$`instanceOf`==1,"Yes","No"))
Statement$`inception` <- as.factor(ifelse(Statement$`inception`==1,"Yes","No"))
Statement$`country` <- as.factor(ifelse(Statement$`country`==1,"Yes","No"))
Statement$`coordinateLocation` <- as.factor(ifelse(Statement$`coordinateLocation`==1,"Yes","No"))
Statement$`memberOf` <- as.factor(ifelse(Statement$`memberOf`==1,"Yes","No"))
Statement$`officialWebsite` <- as.factor(ifelse(Statement$`officialWebsite`==1,"Yes","No"))
Statement$`CommonsCategory` <- as.factor(ifelse(Statement$`CommonsCategory`==1,"Yes","No"))
Statement$`categoryForAlumni` <- as.factor(ifelse(Statement$`categoryForAlumni`==1,"Yes","No"))
Statement$`image` <- as.factor(ifelse(Statement$`image`==1,"Yes","No"))
Statement$`hasSubsidiary` <- as.factor(ifelse(Statement$`hasSubsidiary`==1,"Yes","No"))
Statement$`mainTopicCategory` <- as.factor(ifelse(Statement$`mainTopicCategory`==1,"Yes","No"))
Statement$`categoryForEmployees` <-as.factor(ifelse(Statement$`categoryForEmployees`==1,"Yes","No"))
Statement$`socialMediaFollowers` <- as.factor(ifelse(Statement$`socialMediaFollowers`==1,"Yes","No"))
Statement$`locatedInAdministrativeTerritory` <- as.factor(ifelse(Statement$`locatedInAdministrativeTerritory`==1,"Yes","No"))
Statement$`studentsCount` <- as.factor(ifelse(Statement$`studentsCount`==1,"Yes","No"))
Statement$`languageUsed` <- as.factor(ifelse(Statement$`languageUsed`==1,"Yes","No"))

# STEP 13: TRAIN AND EVALUATE THE STATEMENT DECISION TREE
# ------------------------------------------------------------
set.seed(123) # For reproducibility

# STEP 14: SPLIT STATEMENT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------
# Split dataset into training (80%) and testing (20%)
trainIndex <- createDataPartition(
  Statement$Ranked,
  p = 0.8,
  list = FALSE
)

trainData <- Statement[trainIndex, ]
testData <- Statement[-trainIndex, ]

# STEP 15: BUILD THE IDENTIFIER DECISION TREE
# ------------------------------------------------------------

# Train a classification tree using Gini impurity.

tree_model <- rpart(
  Ranked ~ .,
  data = trainData,
  method = "class",
  control = rpart.control(minsplit = 4, cp = 0.007)
)

# Display tree details.
print(tree_model)
summary(tree_model)

# Display the complexity parameter table.
printcp(tree_model)

# Select the complexity parameter with minimum cross-validated error.
best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
best_cp

# Prune the tree to reduce unnecessary branches.
pruned_tree <- prune(tree_model, cp = best_cp)

# Display the pruned tree.
print(pruned_tree)
summary(pruned_tree)

# STEP 16: PLOT THE IDENTIFIER DECISION TREE
# ------------------------------------------------------------
# Plot the tree (very helpful)
rpart.plot(tree_model)
rpart.plot(tree_model,
           cex = 0.8,
           #tweak = 1.2,
           type = 0,
           extra = 0,
           fallen.leaves = TRUE,
           branch.type = 0,
           box.palette = "RdYlGn",
           legend.x = NA,
           shadow.col = "gray",
           nn = FALSE,)

# STEP 17: PREDICT IDENTIFIER RANKING
# ------------------------------------------------------------
# Predict using test data
predictions <- predict(
  tree_model,
  testData,
  type = "class"
)

# STEP 18: EVALUATE IDENTIFIER MODEL
# ------------------------------------------------------------
# Generate confusion matrix
cm <- confusionMatrix(
  predictions,
  testData$Ranked
)

# Display the confusion matrix and overall results.
print(cm)

# Extract overall performance measures.
accuracy <- cm$overall['Accuracy']
kappa <- cm$overall["Kappa"]
print(accuracy)
print(kappa)

# Extract class-wise precision, recall, and specificity.
precision <- cm$byClass[, "Pos Pred Value"]
recall <- cm$byClass[, "Sensitivity"]
specificity <- cm$byClass[, "Specificity"]
print(precision)
print(recall)
print(specificity)

# Calculate F1-score safely.
f1_score <- 2 * (precision * recall) / (precision + recall)
print(f1_score)

table(trainData$Ranked) # to check the number of training data
table(testData$Ranked) # to check the number of testing data

# STEP 19: CLEAR THE R ENVIRONMENT
# ------------------------------------------------------------

rm(list = ls())  # Remove all objects from the environment
cat("\014")      # Clear the console
gc()             # Release unused memory

