knitr::opts_chunk$set(echo = TRUE)

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

library(tidyverse)
library(caret)
library(data.table)
if(!require(lubridate)) install.packages("lubridate", 
                                         repos = "http://cran.us.r-project.org")
library(lubridate)

head(edx)
nrow(edx)
ncol(edx)
names(edx)
summary(edx) # Gives general statistical summary of the data

# We will omit any rows with na as inputs

na.omit(edx)
na.omit(validation)

# Let's clean the data a little bit. We will change the column timestamp to dates using the lubridate package, to see the exact date when the movies were rated

library(lubridate)
edx <- edx %>% mutate(dates = as_datetime(timestamp)) %>% select(-timestamp)
validation <- validation %>% mutate(date = as_datetime(timestamp)) %>% select(-timestamp)
head(validation)

# Create a new column, years
edx  <- edx %>% mutate(years = as.numeric(str_sub(title,-5,-2))) 
# extract years the movie came out and create a new column called years

results <- tibble()
#create a tibble which will organize all the RMSE's for different models

if(!require(tidyverse)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
library(ggplot2)

# Let's take a closer look at the edx data and see if there is any bias within the data
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "red") + 
  ggtitle("Movies")
# We can see that some movies are rated more than others

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "blue") + 
  ggtitle("Users")

# Some users rate the movies more often than others
# Let's also take a look at the frequency/distribution of the ratings

ratings <- as.vector(edx$rating)
ratings <- factor(ratings)
qplot(ratings) +
  ggtitle("Ratings Frequencies")

# There is a high tendency that people often give out 3,4 as ratings

# We can also analyze the relationship between the variable time and other predictors
# We can take a look at the relationship between years the movies were released and the mean ratings of each year

edx %>% group_by(years) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(years, mean_rating)) +
  geom_smooth()

# Used Loess method to smooth by default
# We can see a generally see the that the average ratings for more recent movies are lower and movies that were in the mid-early 1900's have higher mean ratings

# Create a functon called 'RMSE'
RMSE <- function(actual, predicted){
  sqrt(mean((actual-predicted)^2))
}

# Clean the data for the validation set
validation <- validation %>% mutate(years = as.numeric(str_sub(title,-5,-2)))

# Splitting edx into test and training set
set.seed(1996)
test_index2 <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
temporary_test <- edx %>% slice(test_index2)
train <- edx %>% slice(-test_index2)

# Making sure testset and train set have same movieIds and userIds
test <- temporary_test %>% semi_join(train, by = "movieId") %>% 
  semi_join(train, by = "userId")

# Putting the removed rows back into the training set
removed <- anti_join(temporary_test, test)
train <- rbind(train, removed)

test_ratings <- test$rating # ratings in test set

set.seed(2020)
random <- sample(c(0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0), length(test_ratings), replace = TRUE)
RMSE(test_ratings,random)
# RMSE turns out 2.156021 which is terrible considering that the prediction could differ by up to two stars!
# Will test it on the validation set

random <- sample(c(0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0), length(validation$rating), replace = TRUE)
rmse_random <- RMSE(validation$rating, random)
results <- bind_rows(results, data_frame(method="Random Model",  RMSE = rmse_random))

mu <- mean(train$rating)
RMSE(test_ratings, mu)

# RMSE has now decreased to 1.59691 which is much better, but more improvement could be made to this model
# Let's test it on our validation set to obtain our RMSE

mean_rmse <- RMSE(validation$rating, mu)
results <- bind_rows(results, data_frame(method= "Mean Model",  RMSE = mean_rmse))

permovie_averages1 <- train %>% group_by(movieId) %>% 
  summarize(bi = mean(rating - mu))

model1_prediction <- test %>% left_join(permovie_averages1, by='movieId') %>%
  mutate(prediction = mu + bi)

RMSE(test_ratings, model1_prediction$prediction) 
# This was the RMSE for the test set, we will now obtain RMSE for the validation set

model1_prediction_valid <- validation %>% left_join(permovie_averages1, by='movieId') %>%
  mutate(prediction = mu + bi)
model1_rmse <- RMSE(validation$rating, model1_prediction_valid$prediction)
results <- bind_rows(results, data_frame(method= "Movie Specific Effect Model",  RMSE = model1_rmse))

permovie_averages2 <- train %>% 
  left_join(permovie_averages1, by='movieId') %>%
  group_by(userId) %>%
  summarize(bu = mean(rating - mu - bi))

model2_prediction <- test %>% left_join(permovie_averages2, 'userId') %>% 
  left_join(permovie_averages1, by='movieId') %>%
  mutate(prediction = mu + bi + bu)

RMSE(test_ratings, model2_prediction$prediction)
# We did a better job at estimating the rating by incorporating the user specific effect to our model
# Let's give it a try on our validation set

model2_prediction_valid <- validation %>% left_join(permovie_averages1, by='movieId') %>%
  left_join(permovie_averages2, by = 'userId') %>%
  mutate(prediction = mu + bi + bu)

model2_rmse <- RMSE(validation$rating, model2_prediction_valid$prediction)
results <- bind_rows(results, data_frame(method= "Movie + userId Specific Effect Model",  RMSE = model2_rmse))

permovie_averages5 <- train %>% 
  left_join(permovie_averages1, by='movieId') %>%
  left_join(permovie_averages2, by = 'userId') %>%
  group_by(years) %>%
  summarize(by = mean(rating - mu - bi - bu))

model5_prediction <- test %>% left_join(permovie_averages2, 'userId') %>% 
  left_join(permovie_averages1, 'movieId') %>%
  left_join(permovie_averages5, 'years') %>%
  mutate(prediction = mu + bi + bu + by)

RMSE(test_ratings, model5_prediction$prediction)
# By incorporating the year effect we were able to make an improvement on our model

# Let's show that using our validation set
model5_prediction_valid <- validation %>% left_join(permovie_averages1, by='movieId') %>%
  left_join(permovie_averages2, by = 'userId') %>%
  left_join(permovie_averages5, 'years') %>%
  mutate(prediction = mu + bi + bu + by) 

model5_rmse <- RMSE(validation$rating, model5_prediction_valid$prediction)
results <- bind_rows(results, data_frame(method= "Movie + userId + year Specific Effect Model",  RMSE = model5_rmse))

lambdas <- seq(0,7,0.25) 
# Here the lambdas are tuning parameters and we will find the best lambda through cross validation method
# For each lambda, bi & bu is calculated and ratings are predicted & tested against the testset
# Cross validation code requires some time to run

list_RMSE <- function(lambda){
  
  mu <- mean(train$rating)
  
  permovie_averages3 <- train %>% 
    group_by(movieId) %>% 
    summarize(bi = sum(rating - mu)/(n() + lambda)) # movie specific effect regularized
  
  permovie_averages4 <- train %>% 
    left_join(permovie_averages3, by='movieId') %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(n() + lambda)) # userId specific effect regualarized
  
  permovie_averages6 <- train %>% 
    left_join(permovie_averages3, by='movieId') %>%
    left_join(permovie_averages4, by ='userId') %>%
    group_by(years) %>%
    summarize(by = sum(rating - mu - bi - bu)/(n() + lambda)) # year specific effect regualarized
  # predict     
  test_prediction <- test %>% left_join(permovie_averages3, by = 'movieId') %>%
    left_join(permovie_averages4, by ='userId') %>%
    left_join(permovie_averages6, by ='years') %>%
    mutate(prediction = mu + bi + bu + by)
  
  RMSE(test_ratings, test_prediction$prediction)
}

RMSEs <- sapply(lambdas, list_RMSE)
plot(lambdas, RMSEs)

lambdas[which.min(RMSEs)] 
# Lambda which minimized RMSE the most (optimal RMSE) against the test data was 4.5
# Now that we have our model lambda, we will test it out on our validation set

mu <- mean(train$rating)

permovie_averages3 <- train %>% 
  group_by(movieId) %>% 
  summarize(bi = sum(rating - mu)/(n() + 4.5)) # movie specific effect regularized

permovie_averages4 <- train %>% 
  left_join(permovie_averages3, by='movieId') %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(n() + 4.5)) # userId specific effect regualarized

permovie_averages6 <- train %>% 
  left_join(permovie_averages3, by='movieId') %>%
  left_join(permovie_averages4, by ='userId') %>%
  group_by(years) %>%
  summarize(by = sum(rating - mu - bi - bu)/(n() + 4.5)) # year specific effect regualarized

validation_prediction <- validation %>% left_join(permovie_averages3, by = 'movieId') %>%
  left_join(permovie_averages4, by ='userId') %>%
  left_join(permovie_averages6, by ='years') %>%
  mutate(prediction = mu + bi + bu + by)

final <- RMSE(validation$rating, validation_prediction$prediction)
results <- bind_rows(results, data_frame(method= "Regularization on Movie + userId Specific Effect Model",  RMSE = final))

results %>% knitr::kable() #final results