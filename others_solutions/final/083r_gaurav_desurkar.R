#Checking Summary of data
summary(test_2umaH9m)

#checking structure
str(train_LZdllcl) 

#missing Value %
summary(train_LZdllcl$education) #0.04% is missing
summary(train_LZdllcl$previous_year_rating) #0.08% is missing

#checking Bias
prop.table(table(train_LZdllcl$is_promoted))
#0          1 
#0.91482995 0.08517005 # Data is bias

#Feature Engineering
#Gender - converting into numbers
train_LZdllcl$gender <- as.numeric(ifelse(train_LZdllcl$gender == "m",1,0)) 
test_2umaH9m$gender <- as.numeric(ifelse(test_2umaH9m$gender == "m",1,0))

#Finding mean of avg_training_score
train_LZdllcl$avg_training_score_mean <- as.numeric(mean(train_LZdllcl$avg_training_score))
test_2umaH9m$avg_training_score_mean <-  as.numeric(mean(test_2umaH9m$avg_training_score))

#Flag_high_training_score - If training score is greater than mean and awards won is true then 1 else o
train_LZdllcl$Flag_high_training_score <- ifelse(train_LZdllcl$avg_training_score > train_LZdllcl$avg_training_score_mean  & train_LZdllcl$awards_won. == "1",1,0)
test_2umaH9m$Flag_high_training_score  <- ifelse(test_2umaH9m$avg_training_score > test_2umaH9m$avg_training_score_mean & test_2umaH9m$awards_won. == "1",1,0)

#Missing Value imputation 
#Replacing with mode
train_LZdllcl$previous_year_rating[is.na(train_LZdllcl$previous_year_rating)] <- '3'
levels(train_LZdllcl$education)[levels(train_LZdllcl$education) == ""] <- "Bachelor's"

train_LZdllcl$previous_year_rating <- as.numeric(train_LZdllcl$previous_year_rating)

test_2umaH9m$previous_year_rating[is.na(test_2umaH9m$previous_year_rating)] <- '3'
levels(test_2umaH9m$education)[levels(test_2umaH9m$education) == ""] <- "Bachelor's"

test_2umaH9m$previous_year_rating <- as.numeric(test_2umaH9m$previous_year_rating)

# One hot encoding
x_train <- dummy.data.frame(data = train_LZdllcl, names = c("department","region","education","recruitment_channel","previous_year_rating")
                            , sep = "_")

x_test <- dummy.data.frame(data = test_2umaH9m, names = c("department","region","education","recruitment_channel","previous_year_rating")
                            , sep = "_")

#----Creating training and test data 
input_ones  <- x_train[which(x_train$is_promoted == "1"),]
input_zeros <- x_train[which(x_train$is_promoted == "0"),]
set.seed(1234)  # for repeatability of samples
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones)) #1.s for training data
input_zeros_training_rows <-sample(1:nrow(input_zeros), 0.7*nrow(input_zeros)) # 0's for training data, same count as 1's

#Training Data
train_ones <- input_ones[input_ones_training_rows,] # Creataing train one data
train_zeros<- input_zeros[input_zeros_training_rows,] # Creataing train zero data
traindata <- rbind(train_ones,train_zeros) # Final train prepared

prop.table(table(traindata$is_promoted))

#Test Data
test_ones  <- input_ones[-input_ones_training_rows,]
test_zeros <- input_zeros[-input_zeros_training_rows,]
testdata <- rbind(test_ones, test_zeros)

prop.table(table(testdata$is_promoted))

#----------XGboost-------------------------------------------------------------------

#Store labels
tr_label <- traindata$is_promoted
ts_label <- testdata$is_promoted

#Remove irrelevant field
traindata$is_promoted <-NULL
testdata$is_promoted <- NULL

traindata$avg_training_score_mean <- NULL
testdata$avg_training_score_mean <- NULL

df_train <- subset(traindata, select = -c(employee_id))
df_test  <- subset(testdata,  select = -c(employee_id))

dtrain <- xgb.DMatrix(data = as.matrix(df_train), label = as.matrix(tr_label))
dtest  <- xgb.DMatrix(data = as.matrix(df_test),  label = as.matrix(ts_label))

#default parameters
set.seed(999)
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3
               , gamma=5, max_depth=10, min_child_weight=5, subsample=0.85, colsample_bytree=0.8)

set.seed(999)
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 500,nfold = 15,  metrics = "error",
                showsd = T, stratified = T, print_every_n = 1, early_stopping_rounds = 20, maximize = F)

#Stopping. Best iteration:
#  [45] train-error:0.056539+0.000454	test-error:0.057996+0.003239

#Best iteration:
# iter train_error_mean train_error_std test_error_mean test_error_std
# 48        0.0556208    0.0004113762       0.0582049    0.003721092

set.seed(999)
xgb1 <- xgb.train(params = params, data = dtrain, nrounds = 300, watchlist = list(val=dtest,train=dtrain)
                  , print_every_n = 1, early_stopping_rounds = 10, maximize = F , eval_metric = "error")

min(xgb1$evaluation_log)#0.055128

#view variable importance plot
mat <- xgb.importance (model = xgb1)
xgb.plot.importance (importance_matrix = mat)

#model prediction on test data
xgbpred_test <- predict (xgb1,dtest)

#------------------------------------------------------------------------------------------
#Loop To decide threshold
max_F_score <- 0
threhold_cutoff <- min(xgbpred_test)
#predicted <- plogis(predict(LRfit, LR_training_data_complete, type = "response"))
for (i in seq( min(xgbpred_test),  max(xgbpred_test) ,.02)){
  
  optCutOff <- i
  predict = ifelse(xgbpred_test >= i,1,0)
  #confusion_train <- confusionMatrix(LR_training_data_complete$target, predicted , positive = "1", threshold = optCutOff)
  retrieved<-sum(predict)
  
  precision <- sum(predict & ts_label) / retrieved
  recall    <- sum(predict & ts_label) / sum(ts_label)
  
  #F1 Score
  Fscore <- 2*precision*recall / (precision+recall)
  if(Fscore > max_F_score){
    threhold_cutoff <- i
    max_F_score <- Fscore
  }	
  print(sprintf("Highest Value of Fscore is %f at cut-off %f\n",Fscore,i))
} 
print(threhold_cutoff)#0.3400404

#----------------------------------------------------------------------------------------------


#Deciding on optimal prediction probability cutoff for the model
library(InformationValue)
#optCutOff <- optimalCutoff(ts_label,xgbpred_test)#[1]#0.1781765
#optCutOff#0.0199938

optCutOff <- threhold_cutoff

#Confusion Matrix
confusionMatrix(ts_label, xgbpred_test, threshold = optCutOff)

#ROC curve
plotROC(actuals=ts_label, predictedScores=xgbpred_test)

#Model Accuracy 
confusion <- table(xgbpred_test > optCutOff, ts_label)
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy #0.9092623'

#sensitivity, REcall
Recall <-  sensitivity(ts_label, xgbpred_test, threshold = optCutOff)#0.5046395
Recall
#precision
precision <- precision(ts_label, xgbpred_test, threshold = optCutOff) #0.4697674
precision
#F1 Score
Fmeasure <- 2 * precision * Recall / (precision + Recall)#0.5079051
Fmeasure #0.5019194 - 39 |  0.5048544 -40 | 0.5024534 - 41
#---------------------------------------------------------------------------------
#Actual Test Data

#Using all training data
remove(df_training)
traindata_temp <- traindata
testdata_temp <- testdata

#adding target
traindata_temp$is_promoted <- tr_label
testdata_temp$is_promoted <- ts_label

#All training data
df_training <- rbind(traindata_temp,testdata_temp)

#Store labels
Target <- df_training$is_promoted

#Removing fields
df_training$employee_id <- NULL
df_training$is_promoted <- NULL

#XGBoost final
train_Dmatrix <- xgb.DMatrix(data = as.matrix(df_training), label = as.matrix(Target))

set.seed(999)
xgb1 <- xgb.train(params = params, data = train_Dmatrix, nrounds = 300
                  , print_every_n = 1, maximize = F , eval_metric = "error"
                  )


xgbpred_train <- predict(xgb1,train_Dmatrix )

#Loop To decide threshold
max_F_score <- 0
threhold_cutoff <- min(xgbpred_train)
#predicted <- plogis(predict(LRfit, LR_training_data_complete, type = "response"))
for (i in seq( min(xgbpred_train),  max(xgbpred_train) ,.05)){
  
  optCutOff <- i
  predict=ifelse(xgbpred_train>=i,1,0)
  #confusion_train <- confusionMatrix(LR_training_data_complete$target, predicted , positive = "1", threshold = optCutOff)
  retrieved<-sum(predict)
  
  precision <- sum(predict & Target) / retrieved
  recall    <- sum(predict & Target) / sum(Target)
  
  #F1 Score
  Fscore <- 2*precision*recall / (precision+recall)
  if(Fscore > max_F_score){
    threhold_cutoff <- i
    max_F_score <- Fscore
  }	
  print(sprintf("Highest Value of Fscore is %f at cut-off %f\n",Fscore,i))
} 
print(threhold_cutoff)#0.3000123


#model prediction

#Removing fields
x_test$employee_id <- NULL
x_test$avg_training_score_mean <- NULL

ActualTest <- xgb.DMatrix(data = as.matrix(x_test))

#prediction on actual data
xgbpred <- predict(xgb1,ActualTest)

optCutOff <- 0.3000123
is_promoted <- ifelse(xgbpred > optCutOff ,"1", "0")

test_2umaH9m$is_promoted <-NULL
test_2umaH9m$is_promoted <- is_promoted

Submit <- subset(test_2umaH9m, select = c("employee_id","is_promoted"))

write.csv(Submit, "D:/Statastics/Analytics Vidhya/WNS Hackathon/submit.csv", row.names=F)






