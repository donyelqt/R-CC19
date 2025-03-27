.libPaths("C:/Users/Donielr Arys Antonio/R-CC19")

library(randomForest)    # For random forest model
library(ggplot2)         # For visualization
library(caret)           # For model training and evaluation
library(dplyr)           # For data manipulation

set.seed(123)

data(iris)
cat("First few rows of the dataset:\n")
print(head(iris))
cat("\nSummary of the dataset:\n")
print(summary(iris))

train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]
