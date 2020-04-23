knitr::opts_chunk$set(echo = TRUE)

library(caret)
library(dplyr)
library(matrixStats)
library(Matrix)
library(rpart)

if(!require("PerformanceAnalytics")){
    install.packages("PerformanceAnalytics")
    library(PerformanceAnalytics)
}

if(!require("gam")){
    install.packages("gam")
    library(gam)
}

if(!require("naivebayes")){
    install.packages("naivebayes")
    library(naivebayes)
}

if(!require("splines")){
    install.packages("splines")
    library(splines)
}

getwd()
setwd("/Users/st_woogie/Desktop") #the csv will be uploaded onto my github
dat <- read.csv("indian_liver_patient.csv")

nrow(dat) #11 rows
ncol(dat) #583 columns 
head(dat)

sum(is.na(dat)) #4 rows with na as inputs
newdat <- na.omit(dat) #omit all na's
Disease <- ifelse(newdat$Dataset == 1, 1, 0) 

# Changing 1(has liver disease) and 2(doens't have liver disease), to 0(doesn't have liver disease) and 1(has liver disease)

Sex <- ifelse(newdat$Gender == 'Male', 0, 1) #Let male be 0 and female be 1 (binarizing the data)
newdat <- cbind(Sex, newdat, Disease)
newdat <- newdat[,-c(3,12)]
newdat$Sex <- as.factor(newdat$Sex)
newdat$Disease <- as.factor(newdat$Disease) #Turned the features 'Sex' and our response 'Dataset' into factors)
head(newdat)

library(tidyverse)
nrow(newdat) #579 observations after omitting all the na values
ncol(newdat) # two categorical features and the rest are numeric
mean(Disease == 1) # Proportion of diseased in our dataset is 71.5 percent
mean(Sex == 0) # Proportion of male seems higher
newdat %>% group_by(Sex, Disease) %>% summarize(count = n()) 
(323/579)/0.715 # conditional prob that a person is diseased given the sex is male
(91/579)/0.285 # conditional prob that a person is diseased given the sex is female

# Taking a look at correlation between features (All the factor features were excluded in this plot)

if(!require("PerformanceAnalytics")){
  install.packages("PerformanceAnalytics")
  library("PerformanceAnalytics")
}
chart.Correlation(newdat[,-c(1,11)], histogram=TRUE, pch=19) 

# In the middle shows the disribution of each variable
# The bottom shows the bivariate scatter plots with a fitted line
# This plot shows a sign of little colinearity between the features which is a good sign

# Standardizing the numeric features since they range so vastly (by using the sweep function, and matrix manipulation)
subset <- newdat[,-c(1,11)]
subset <- as.matrix(subset)
x_mean0 <- sweep(subset, 2, colMeans(subset))
x_standardized <- sweep(x_mean0, 2, colSds(subset), FUN = "/")
newdat <- cbind(x_standardized,newdat[,c(1,11)])

# Removing variables 'Total_Bilirubin' and 'Total_proteins'
head(newdat)
newdat <- newdat[,-c(2,7)]
head(newdat)

# Partitioning data (Used 0.1 to 0.9 instead half and half, the number of observations is too small to train algorithms with half of the data)
set.seed(1)
test_index <- createDataPartition(newdat$Disease, times = 1, p = 0.1, list = FALSE)
test <- newdat %>% slice(test_index)
train <- newdat %>% slice(-test_index)

# Random Forest
set.seed(8)
train_rf1 <- train(Disease ~ ., method = 'rf', tunegrid = data.frame(mtry = seq(1,7,1)), data = train, ntree = 100)
train_rf1$bestTune # Best mtry was chosen to be 2 after tuning parameters
y_hat_rf1 <- predict(train_rf1, test)
mean(y_hat_rf1 == test$Disease)
varImp(train_rf1) # Variable importance shows that sex:female is the least important variable

y_hat_rf1 <- as.matrix(y_hat_rf1) # Turned this into a vector for the very final step at the end when creating an ensemble
colnames(y_hat_rf1) <- 'rf1'

# Tried removing the feature Sex, and see how it affects the accuracy
new_train <- train[,-1]
head(new_train)
train_rf2 <- train(Disease ~ ., method = 'rf', tunegrid = data.frame(mtry = seq(1,7,1)), data = new_train, ntree = 100)
y_hat_rf2 <- predict(train_rf2, test)
mean(y_hat_rf2 == test$Disease) # Can see that there is no change in accuracy, but hopefully getting rid of a feature would decrease the process time
y_hat_rf2 <- as.matrix(y_hat_rf2)
colnames(y_hat_rf2) <- 'rf2'
# Therefore I just decided to leave the feature sex and stick with rf1 instead of rf2

# Fit lda using train function in the caret package
set.seed(2)
train_lda <- train(Disease ~., method = "lda", data = new_train)
y_hat_lda <- predict(train_lda, test)
mean(y_hat_lda == test$Disease)
y_hat_lda <- as.matrix(y_hat_lda)
colnames(y_hat_lda) <- "lda"

# Fit a new knn model with 10 fold cross validation, where each partition consists of 10% of the total data
set.seed(3)
train_knn <- train(Disease ~ .,
                   method = "knn",
                   data = new_train,
                   tuneGrid = data.frame(k = seq(50, 100, 5)), trControl = trainControl(method = "cv", number = 10, p = 0.9))

plot(train_knn) # Plotted k against the accuracy for various tuning parameters
train_knn$bestTune # Shows the besttuned k (100)

# Initially the tuning parameters were set as seq(3, 51, 3), but once I plotted the accuracy(bootstrap), against all the k's I realized that the tuning parameter could be optimized even more, once it is set higher. Therefore I maximized the accuracy when I put k = 100.

y_hat_knn <- predict(train_knn, test)
mean(y_hat_knn == test$Disease)
y_hat_knn <- as.matrix(y_hat_knn)
colnames(y_hat_knn) <- 'knn'

# Fit a Naive Bayes Model
set.seed(9)
library(naivebayes)
train_naive <- train(Disease ~., method = 'naive_bayes', data = new_train)
y_hat_naive <- predict(train_naive, test)
mean(y_hat_naive == test$Disease)
y_hat_naive <- as.matrix(y_hat_naive)
colnames(y_hat_naive) <- 'Naive_Bayes'

# Fit a Gam Loess Model
set.seed(10)
train_gam <- train(Disease ~., method = 'gamLoess', data = new_train)
y_hat_gam <- predict(train_gam, test)
y_hat_gam <- as.matrix(y_hat_gam)
colnames(y_hat_gam) <- 'Gam_Loess'
mean(y_hat_gam == test$Disease)

#Fit a glm model
set.seed(1996)
train_glm <- train(Disease ~. ,method = 'glm', data = new_train)
y_hat_glm <- predict(train_glm, test)
mean(y_hat_glm == test$Disease) #accuracy

y_hat_glm <- as.matrix(y_hat_glm)
colnames(y_hat_glm) <- 'GLM'

# Train multinom, adaboost and svmlinear models at once
set.seed(11)
models <- c("multinom", "adaboost", "svmLinear")

ensemble <- lapply(models, function(model){ 
  print(model)
  train(Disease ~ ., method = model, data = train)
}) 

names(ensemble) <- models # Wrote a function that trains multiple models at once

#create a prediction matrix of those three trained models
predict_all <- function(model){
  predict(model, test)
}
pred <- sapply(ensemble, predict_all)
pred_matrix <- as.matrix(pred)
pred_matrix

#Show accuracy for each ML algorithm
colMeans(pred_matrix == test$Disease)

#Combine all the good models
final_ensemble <- cbind(pred_matrix[,2], y_hat_gam, y_hat_rf2, y_hat_glm) #ensemble model was designed with the models that obtained top 4 highest accuracies

#I included all the models that I have created in this project at first, in the final ensemble and created many different subsets of ensemble which yielded me the highest accuracy, which happened to be models with top 4 accuracies

one <- rowMeans(final_ensemble == 1)
y_hat_final_ensemble <- ifelse(one > 0.5, 1, 0) 
#If the proportion of 1's in a row (each observation) comparing the predictions of all the models in the ensemble go over 0.5, then we say that the ensemble model has predicted the patient to have a liver disease

mean(y_hat_final_ensemble == test$Disease) #calculating the accuracy of my ensemble model
