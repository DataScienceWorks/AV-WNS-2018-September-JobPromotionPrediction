library(keras)
train <- read.csv("C:/Users/ssharan/Desktop/WNS/train.csv", stringsAsFactors = T,na.strings = "")
test <- read.csv("C:/Users/ssharan/Desktop/WNS/test.csv",stringsAsFactors = T,na.strings = "")

submission <- test$employee_id

########  Data Preprocessing ########

train$previous_year_rating <- ifelse(is.na(train$previous_year_rating),0,train$previous_year_rating)
train$education <- as.factor(ifelse(is.na(train$education),4,train$education))
train$no_of_trainings <- ifelse(train$no_of_trainings==10,9,train$no_of_trainings)

train$no_of_trainings <- as.factor(train$no_of_trainings)
train$awards_won. <- as.factor(train$awards_won.)
train$KPIs_met..80. <- as.factor(train$KPIs_met..80.)
train$previous_year_rating  <- as.factor(train$previous_year_rating)
train$is_promoted <- as.factor(train$is_promoted)
train$employee_id <- NULL



test$previous_year_rating <- ifelse(is.na(test$previous_year_rating),0,test$previous_year_rating)

test$no_of_trainings <- as.factor(test$no_of_trainings)
test$awards_won. <- as.factor(test$awards_won.)
test$KPIs_met..80. <- as.factor(test$KPIs_met..80.)
test$previous_year_rating  <- as.factor(test$previous_year_rating)

test$education <- as.factor(ifelse(is.na(test$education),4,test$education))
test$employee_id <- NULL



num_var <- train[,sapply(train, is.numeric) == T]
train[,sapply(train, is.numeric) == T] <- NULL

num_var_test <- test[,sapply(test, is.numeric) == T]
test[,sapply(test, is.numeric) == T] <- NULL

# scale to [0,1]
train <- cbind( apply(num_var, 2, function(x) (x-min(x))/(max(x) - min(x))) , train)
test  <- cbind( apply(num_var_test, 2, function(x) (x-min(x))/(max(x) - min(x))) , test)


library(Matrix)


x_test <- model.matrix(~.+0,data = test)
x_train <- model.matrix(~.+0,data = train[,-13]) 

y <- as.matrix(train[,13])

total_train <- model.matrix(~.+0,data = train)


use_session_with_seed(1,disable_parallel_cpu = FALSE)



rec_obj <- recipe(is_promoted1 ~ ., data = total_train) %>%
            prep(data = total_train)

x_train_tbl <- bake(rec_obj, newdata = total_train) %>% select(-is_promoted1)

x_test_tbl  <- bake(rec_obj, newdata = x_test) 

glimpse(x_train_tbl)

# Response variables for training and testing sets
y_train_vec <- as.matrix(train[,13])




# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )


# fit model with our training data set, training will be done for 40 times data set

history <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = 40,
  validation_split = 0.30
)



plot(history)


# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()



result_keras<- cbind.data.frame(submission,data.frame(yhat_keras_class_vec))

colnames(result_keras)<- c("employee_id","is_promoted")

write.csv(result_keras,"C:/Users/ssharan/Desktop/WNS/result_keras.csv",row.names = F)
