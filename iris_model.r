.libPaths("C:/Users/Donielr Arys Antonio/R-CC19")

# Step 1: Load required packages
library(randomForest)    # For random forest model
library(ggplot2)         # For visualization
library(caret)           # For model training and evaluation
library(dplyr)           # For data manipulation

# Set seed for reproducibility
set.seed(123)

# Step 2: Load and Explore the iris dataset
data(iris)
cat("First few rows of the dataset:\n")
print(head(iris))
cat("\nSummary of the dataset:\n")
print(summary(iris))

# Step 3: Split data into training (70%) and testing (30%) sets
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]


# Step 4: Train the Random Forest model
rf_model <- randomForest(Species ~ ., 
                         data = train_data,
                         ntree = 100,
                         importance = TRUE)

# Step 5: Make predictions on test data
predictions <- predict(rf_model, test_data)

# Step 6: Evaluate Model Performance
conf_matrix <- confusionMatrix(predictions, test_data$Species)
print(conf_matrix)


# Step 7: Feature Importance Visualization
var_importance <- as.data.frame(importance(rf_model, type = 1))
var_importance$Feature <- rownames(var_importance)
colnames(var_importance) <- c("Importance", "Feature")
p <- ggplot(var_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", aes(fill = Feature), color = "#2F4F4F", width = 0.7) +
  geom_text(aes(label = round(Importance, 1)), vjust = -0.5, size = 4.5, color = "#2F4F4F", fontface = "bold") +
  coord_flip() +
  labs(title = "Feature Importance in Random Forest",
       subtitle = "Based on Mean Decrease in Accuracy",
       x = NULL, 
       y = "Mean Decrease in Accuracy") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", color = "#2F4F4F"),
        plot.subtitle = element_text(hjust = 0.5, color = "#2F4F4F"),
        axis.text = element_text(color = "#2F4F4F", size = 12),
        axis.title = element_text(face = "bold", color = "#2F4F4F"),
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        panel.grid.major.x = element_line(color = "#D3D3D3", linetype = "dotted"),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none") +
  scale_fill_manual(values = c("Sepal.Length" = "#FF6B6B", 
                               "Sepal.Width" = "#4ECDC4", 
                               "Petal.Length" = "#45B7D1", 
                               "Petal.Width" = "#96CEB4")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))
ggsave("C:/Users/Donielr Arys Antonio/R-CC19/var_importance.png", 
       p, 
       width = 12, height = 8, 
       bg = "white", 
       device = "png")


# Step 8: Scatter Plot Predictions and Visualization
plot_data <- test_data
plot_data$Prediction <- predictions
plot_data$Correct <- plot_data$Species == plot_data$Prediction
p1 <- ggplot(plot_data, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point(aes(color = Species, shape = Prediction, alpha = Correct), size = 4.5, stroke = 1, fill = NA) +
  scale_color_manual(values = c("setosa" = "#E63946", 
                                "versicolor" = "#1D9A97", 
                                "virginica" = "#457B9D"),
                     name = "Actual Species") +
  scale_shape_manual(values = c(16, 17, 18), 
                     name = "Predicted Species") +
  scale_alpha_manual(values = c(0.3, 1), 
                     guide = "none") +
  labs(title = "Actual vs Predicted Species Classification",
       subtitle = "Random Forest Model on Iris Dataset",
       x = "Sepal Length (cm)",
       y = "Sepal Width (cm)",
       caption = "Points: Actual Species | Shapes: Predicted Species") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold", size = 12),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 18, color = "#2F4F4F"),
        plot.subtitle = element_text(hjust = 0.5, size = 12, color = "#2F4F4F"),
        plot.caption = element_text(hjust = 1, size = 10, color = "gray50"),
        axis.title = element_text(face = "bold", color = "#2F4F4F"),
        axis.text = element_text(color = "#2F4F4F", size = 12),
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "#D3D3D3", linetype = "dotted"),
        panel.grid.minor = element_line(color = "#E8ECEF", linetype = "dotted")) +
  guides(color = guide_legend(order = 1),
         shape = guide_legend(order = 2))
ggsave("C:/Users/Donielr Arys Antonio/R-CC19/scatter_plot.png", 
       p1, 
       width = 12, height = 8, 
       bg = "white", 
       device = "png")

# Step 9: Model Accuracy Visualization
accuracy <- conf_matrix$overall['Accuracy']
stats <- data.frame(
  Metric = "Accuracy",
  Value = accuracy
)

p2 <- ggplot(stats, aes(x = Metric, y = Value)) +
  geom_bar(stat = "identity", fill = "#66C2A5", width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", Value)), 
            vjust = -0.5, size = 5, fontface = "bold") +
  coord_cartesian(ylim = c(0, 1)) +  # Better than ylim() to avoid clipping
  labs(title = "Model Performance: Classification Accuracy",
       subtitle = paste("Random Forest, ntree =", rf_model$ntree),
       y = "Accuracy Score",
       x = NULL) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text.x = element_text(size = 12, face = "bold"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "white", color = NA),  # White panel
        plot.background = element_rect(fill = "white", color = NA)) +  # White plot bg
  scale_y_continuous(labels = scales::percent_format())  # Show as percentage
ggsave("C:/Users/Donielr Arys Antonio/R-CC19/accuracy_plot.png", 
       p2, 
       width = 6, height = 4, 
       bg = "white", 
       device = "png")


# Step 10: Conclusion and Summary
cat("\nRandom Forest Model Summary:\n")
print(rf_model)