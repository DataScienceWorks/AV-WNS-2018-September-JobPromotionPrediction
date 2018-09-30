# weight of evidence####

setwd("C:/Users/abhishek2.s/Downloads/Hackathon")
# Importing the dataset
training_set = read.csv('train_LZdllcl.csv')
training_set = training_set[2:14]

test_set = read.csv('test_2umaH9m.csv')
test_set = test_set[2:13]

variables <- names(training_set)

str(training_set)

##########Feature Engineering#############
WOE <- list()

#Categorical
for(j in 1:5){
  WOE[[j]] <- data.frame(class = names(table(training_set[,j],training_set$is_promoted)[,1]),
                    non_event = table(training_set[,j],training_set$is_promoted)[,1],
                    event = table(training_set[,j],training_set$is_promoted)[,2],row.names=c())
  
  WOE[[j]]$perc_non_event <- WOE[[j]]$non_event/(WOE[[j]]$non_event+WOE[[j]]$event)
  WOE[[j]]$perc_event <- WOE[[j]]$event/(WOE[[j]]$non_event+WOE[[j]]$event)
  
  WOE[[j]]$woe <- log(WOE[[j]]$perc_non_event/WOE[[j]]$perc_event)
  
  for(i in 1:length(unique(training_set[,j]))){
    training_set[as.character(training_set[,j])==as.character(WOE[[j]][i,1]),(13+j)] <- WOE[[j]][i,6]
    test_set[as.character(test_set[,j])==as.character(WOE[[j]][i,1]),(12+j)] <- WOE[[j]][i,6]
  }
}

training_set <- training_set[,c(14:18,6:13)]
names(training_set)[1:5] <- variables[1:5]

test_set <- test_set[,c(13:17,6:12)]
names(test_set)[1:5] <- variables[1:5]

##continous 
##no_of_trainings

woe_no_of_trainings <- data.frame(class = names(table(training_set[,6],training_set$is_promoted)[,1]),
                                  non_event = table(training_set[,6],training_set$is_promoted)[,1],
                                  event = table(training_set[,6],training_set$is_promoted)[,2],row.names=c())

woe_no_of_trainings[7,2] <- sum(woe_no_of_trainings[7:10,2])
woe_no_of_trainings[7,3] <- sum(woe_no_of_trainings[7:10,3])

woe_no_of_trainings <- woe_no_of_trainings[1:7,]
woe_no_of_trainings[7,2] <- woe_no_of_trainings[7,2] + 1
woe_no_of_trainings[7,3] <- woe_no_of_trainings[7,3] + 1

woe_no_of_trainings$perc_non_event <- woe_no_of_trainings$non_event/(woe_no_of_trainings$non_event+woe_no_of_trainings$event)
woe_no_of_trainings$perc_event <- woe_no_of_trainings$event/(woe_no_of_trainings$non_event+woe_no_of_trainings$event)

woe_no_of_trainings$woe <- log(woe_no_of_trainings$perc_non_event/woe_no_of_trainings$perc_event)

for(i in 1:7){
  training_set[training_set[,6]==woe_no_of_trainings[i,1],14] <- woe_no_of_trainings[i,6]
  test_set[test_set[,6]==woe_no_of_trainings[i,1],13] <- woe_no_of_trainings[i,6]
}

for(i in 8:10){
  training_set[training_set[,6]==i,14] <- woe_no_of_trainings[7,6]
  test_set[test_set[,6]==i,13] <- woe_no_of_trainings[7,6]
}

names(training_set)[14] <- "woe_no_of_training"
names(test_set)[13] <- "woe_no_of_training"

###age
age_grouping<- data.frame(vars = training_set[,7],flag = training_set[,13],
                                  group = cut(training_set[,7], 
                                      unique(quantile(training_set[,7], na.rm=TRUE,
                              probs = c(seq(0,0.90, by = 0.1),0.99,1))),
                              include.lowest=TRUE))

woe_age <- data.frame(class = names(table(age_grouping$group,age_grouping$flag)[,1]),
                                  non_event = table(age_grouping$group,age_grouping$flag)[,1],
                                  event = table(age_grouping$group,age_grouping$flag)[,2],row.names=c())


woe_age$perc_non_event <- woe_age$non_event/(woe_age$non_event+woe_age$event)
woe_age$perc_event <- woe_age$event/(woe_age$non_event+woe_age$event)

woe_age$woe <- log(woe_age$perc_non_event/woe_age$perc_event)

woe_age$lower <- c(19,unique(quantile(training_set[,7], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:11])

woe_age$upper <- unique(quantile(training_set[,7], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:12]

for(i in 1:11){
  training_set[(training_set[,7] > woe_age[i,7]) & (training_set[,7] <= woe_age[i,8]) ,15] <- woe_age[i,6]
  test_set[(test_set[,7] > woe_age[i,7]) & (test_set[,7] <= woe_age[i,8]) ,14] <- woe_age[i,6]
}

names(training_set)[15] <- "woe_age"
names(test_set)[14] <- "woe_age"

###length_of_service
length_of_service_grouping<- data.frame(vars = training_set[,9],flag = training_set[,13],
                          group = cut(training_set[,9], 
                                      unique(quantile(training_set[,9], na.rm=TRUE,
                                                      probs = c(seq(0,0.90, by = 0.1),0.99,1))),
                                      include.lowest=TRUE))

woe_length_of_service <- data.frame(class = names(table(length_of_service_grouping$group,length_of_service_grouping$flag)[,1]),
                      non_event = table(length_of_service_grouping$group,length_of_service_grouping$flag)[,1],
                      event = table(length_of_service_grouping$group,length_of_service_grouping$flag)[,2],row.names=c())


woe_length_of_service$perc_non_event <- woe_length_of_service$non_event/(woe_length_of_service$non_event+woe_length_of_service$event)
woe_length_of_service$perc_event <- woe_length_of_service$event/(woe_length_of_service$non_event+woe_length_of_service$event)

woe_length_of_service$woe <- log(woe_length_of_service$perc_non_event/woe_length_of_service$perc_event)

woe_length_of_service$lower <- c(0,unique(quantile(training_set[,9], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:10])

woe_length_of_service$upper <- unique(quantile(training_set[,9], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:11]

for(i in 1:10){
  training_set[(training_set[,9] > woe_length_of_service[i,7]) & (training_set[,9] <= woe_length_of_service[i,8]) ,16] <- woe_length_of_service[i,6]
  test_set[(test_set[,9] > woe_length_of_service[i,7]) & (test_set[,9] <= woe_length_of_service[i,8]) ,15] <- woe_length_of_service[i,6]
}

names(training_set)[16] <- "woe_length_of_service"
names(test_set)[15] <- "woe_length_of_service"


##avg training score

avg_training_score_grouping<- data.frame(vars = training_set[,12],flag = training_set[,13],
                                        group = cut(training_set[,12], 
                                                    unique(quantile(training_set[,12], na.rm=TRUE,
                                                                    probs = c(seq(0,0.90, by = 0.1),0.99,1))),
                                                    include.lowest=TRUE))

woe_avg_training_score <- data.frame(class = names(table(avg_training_score_grouping$group,avg_training_score_grouping$flag)[,1]),
                                    non_event = table(avg_training_score_grouping$group,avg_training_score_grouping$flag)[,1],
                                    event = table(avg_training_score_grouping$group,avg_training_score_grouping$flag)[,2],row.names=c())


woe_avg_training_score$perc_non_event <- woe_avg_training_score$non_event/(woe_avg_training_score$non_event+woe_avg_training_score$event)
woe_avg_training_score$perc_event <- woe_avg_training_score$event/(woe_avg_training_score$non_event+woe_avg_training_score$event)

woe_avg_training_score$woe <- log(woe_avg_training_score$perc_non_event/woe_avg_training_score$perc_event)

woe_avg_training_score$lower <- c(38,unique(quantile(training_set[,12], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:11])

woe_avg_training_score$upper <- unique(quantile(training_set[,12], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:12]

for(i in 1:11){
  training_set[(training_set[,12] > woe_avg_training_score[i,7]) & (training_set[,12] <= woe_avg_training_score[i,8]) ,17] <- woe_avg_training_score[i,6]
  test_set[(test_set[,12] > woe_avg_training_score[i,7]) & (test_set[,12] <= woe_avg_training_score[i,8]) ,16] <- woe_avg_training_score[i,6]
}

names(training_set)[17] <- "woe_avg_training_score"
names(test_set)[16] <- "woe_avg_training_score"

##missing values
training_set[is.na(training_set$previous_year_rating),"previous_year_rating"] <- 0
test_set[is.na(test_set$previous_year_rating),"previous_year_rating"] <- 0

###previous_year_rating

previous_year_rating_grouping<- data.frame(vars = training_set[,8],flag = training_set[,13],
                                         group = cut(training_set[,8], 
                                                     unique(quantile(training_set[,8], na.rm=TRUE,
                                                                     probs = c(seq(0,0.90, by = 0.1),0.99,1))),
                                                     include.lowest=TRUE))

woe_previous_year_rating <- data.frame(class = names(table(previous_year_rating_grouping$group,previous_year_rating_grouping$flag)[,1]),
                                     non_event = table(previous_year_rating_grouping$group,previous_year_rating_grouping$flag)[,1],
                                     event = table(previous_year_rating_grouping$group,previous_year_rating_grouping$flag)[,2],row.names=c())


woe_previous_year_rating$perc_non_event <- woe_previous_year_rating$non_event/(woe_previous_year_rating$non_event+woe_previous_year_rating$event)
woe_previous_year_rating$perc_event <- woe_previous_year_rating$event/(woe_previous_year_rating$non_event+woe_previous_year_rating$event)

woe_previous_year_rating$woe <- log(woe_previous_year_rating$perc_non_event/woe_previous_year_rating$perc_event)

woe_previous_year_rating$lower <- c(-1,unique(quantile(training_set[,8], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:5])

woe_previous_year_rating$upper <- unique(quantile(training_set[,8], na.rm=TRUE, probs = c(seq(0,0.90, by = 0.1),0.99,1)))[2:6]

for(i in 1:5){
  training_set[(training_set[,8] > woe_previous_year_rating[i,7]) & (training_set[,8] <= woe_previous_year_rating[i,8]) ,18] <- woe_previous_year_rating[i,6]
  test_set[(test_set[,8] > woe_previous_year_rating[i,7]) & (test_set[,8] <= woe_previous_year_rating[i,8]) ,17] <- woe_previous_year_rating[i,6]
}

names(training_set)[18] <- "woe_previos_year_rating"
names(test_set)[17] <- "woe_previos_year_rating"

#####new variables######
training_set$perc_time_in_firm <- training_set$age/training_set$length_of_service
test_set$perc_time_in_firm <- test_set$age/test_set$length_of_service

training_set$avg_avg_train_score <- training_set$avg_training_score/training_set$no_of_trainings
test_set$avg_avg_train_score <- test_set$avg_training_score/test_set$no_of_trainings

###remove variables###

training_set <- training_set[,-c(6,7,9,17,18)]
test_set <- test_set[,-c(6,7,9,16,17)]

#############model##############
library(xgboost)


param = list(max.depth = 6,eta=0.01,silent=1,
             objective = "binary:logistic", eval_metric = "auc",max_delta_step=1)

cv = xgb.cv(data = as.matrix(training_set[-10]), params = param, label = training_set$is_promoted, nrounds = 2000,nfold =5,early_stopping_rounds = 50)

which(max(cv$evaluation_log[,4]) == cv$evaluation_log[,4])
#0.9397442 cv$evaluation_log[which(max(cv$evaluation_log[,4]) == cv$evaluation_log[,4]),2]
#0.9096436 max(cv$evaluation_log[,4])
classifier1 = xgboost(data = as.matrix(training_set[-10]), params = param, label = training_set$is_promoted, nrounds = which(max(cv$evaluation_log[,4]) == cv$evaluation_log[,4]))

library("MLmetrics")

cutoff = 0.36


#param = list(max.depth = 3,eta=0.05,nthread=2,silent=1,
#             objective = "binary:logistic", eval_metric = "auc")
y_fit <- predict(classifier1, newdata = as.matrix(training_set[-10]))
y_fit[y_fit >= cutoff] <- 1
y_fit[y_fit < cutoff] <- 0
F1_Score(training_set[,10],y_fit) #0.9721147



# Predicting the Test set results
y_pred = predict(classifier1, newdata = as.matrix(test_set))
y_pred = (y_pred >= cutoff)

##result to upload
result <- data.frame(cbind(read.csv('test_2umaH9m.csv')[,1],y_pred))
names(result) <- c('employee_id','is_promoted')

write.csv(result,"xgboost9_woe3.csv",row.names=F)





