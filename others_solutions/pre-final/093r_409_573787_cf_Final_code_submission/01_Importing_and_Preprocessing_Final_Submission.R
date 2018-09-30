library(BBmisc)
library(caret)
library(dummies)

#Import train data into a dataframe
setwd("~/Downloads/AV WNS HACKATHON")
orig_train_data <- read.csv('train_LZdllcl.csv', header = T,strip.white = T,stringsAsFactors = F)

#Check first few rows of the dataframe
head(orig_train_data)

#Check summary
summary(orig_train_data)

#Check for NA values across differenct columns of the dataframe
colSums(is.na(orig_train_data))

# nrow(orig_train_data) - nrow(orig_train_data[orig_train_data$length_of_service>1,]) 
# nrow(orig_train_data) - nrow(orig_train_data[orig_train_data$length_of_service==1 & orig_train_data$is_promoted==1,])
# nrow(orig_train_data[orig_train_data$length_of_service==1 & orig_train_data$is_promoted==1,])
# sum(is.na(orig_train_data$length_of_service))

# Check pattern of data - basic analysis
hist(orig_train_data$age)
hist(orig_train_data$length_of_service)
hist(orig_train_data$no_of_trainings)
hist(orig_train_data$previous_year_rating)
hist(orig_train_data$awards_won.)
hist(orig_train_data$avg_training_score)
hist(orig_train_data$region)

# Inspecting few columns more closely
sum(is.na(orig_train_data$education))

# Identify columns with blank values - education
nrow(orig_train_data[orig_train_data$education=='',])

# Identify columns with blank values - department
nrow(orig_train_data[orig_train_data$department=='',])

# Save original data frame into another df for further processing of the data
processed_data <- orig_train_data

# From previous analysis it is clear that gender column has just two values, convert it into factor
processed_data$gender <- as.factor(processed_data$gender)
summary(processed_data$gender)

# From previous analysis it is clear that department column does not have any blanks/nas, convert it into factor
processed_data$department <- as.factor(processed_data$department)
summary(processed_data$department)

# Replace blank values in education column with 'NA'
processed_data[which(processed_data$education==''),'education'] <- 'NA'

# Convert education column to a factor
processed_data$education <- as.factor(as.character(trimws(processed_data$education,which = 'both')))
summary(processed_data$education)

# Convert region column to a factor
processed_data$region <- as.factor(as.character(trimws(processed_data$region,which = 'both')))
summary(processed_data$region)

# Convert region column to a factor
processed_data$recruitment_channel <- as.factor(as.character(trimws(processed_data$recruitment_channel,which = 'both')))
summary(processed_data$recruitment_channel)

# Check structure of the processed data frame to see whether all the factor variables show-up correctly
str(processed_data)

# Convert KPI_met column to a factor column as it has only 0 and 1 values
processed_data$KPIs_met..80. <- as.factor(processed_data$KPIs_met..80.)
levels(processed_data$KPIs_met..80.) <- c('no','yes')
summary(processed_data$KPIs_met..80.)

# Convert is_promoted column the target column as a factor
processed_data$is_promoted <- as.factor(processed_data$is_promoted)
levels(processed_data$is_promoted) <- c('no','yes')

# Double check whether there are blanks or nas in target variable
summary(processed_data$is_promoted)

# Convert awards_won column to a factor column as there are only 0 or 1 values in this column
processed_data$awards_won. <- as.factor(processed_data$awards_won.)
levels(processed_data$awards_won.) <- c('no','yes')
summary(processed_data$awards_won.)
# Double check whether there are blanks or nas 
summary(processed_data$awards_won.)

# Import the test data as a dataframe, do some prilimnary analysis and treat it the same was as training data set

test_data <- read.csv('test_2umaH9m.csv', header = T,strip.white = T,stringsAsFactors = F)
summary(test_data)

#----------Begin preprocessing of test data--------#

nrow(test_data[test_data$department=='',])
nrow(test_data[test_data$education=='',])

test_data$gender <- as.factor(test_data$gender)
summary(test_data$gender)

test_data$department <- as.factor(test_data$department)
summary(test_data$department)

test_data[which(test_data$education==''),'education'] <- 'NA'
test_data$education <- as.factor(as.character(trimws(test_data$education,which = 'both')))
summary(test_data$education)

test_data$region <- as.factor(as.character(trimws(test_data$region,which = 'both')))
summary(test_data$region)

colnames(test_data)

test_data$recruitment_channel <- as.factor(as.character(trimws(test_data$recruitment_channel,which = 'both')))
summary(test_data$recruitment_channel)

str(test_data)

test_data$KPIs_met..80. <- as.factor(test_data$KPIs_met..80.)
levels(test_data$KPIs_met..80.) <- c('no','yes')
summary(test_data$KPIs_met..80.)

test_data$awards_won. <- as.factor(test_data$awards_won.)
levels(test_data$awards_won.) <- c('no','yes')
summary(test_data$awards_won.)

colSums(is.na(test_data))

#----------End preprocessing of test data--------#

#Check whether the test data contains significantly different levels in factored variables
train_set_dept <- levels(processed_data$department)
test_set_dept <- levels(test_data$department)

setdiff(train_set_dept,test_set_dept)

train_set_edu <- levels(processed_data$education)
test_set_edu <- levels(test_data$education)

setdiff(train_set_edu,test_set_edu)

train_set_reg <- levels(processed_data$region)
test_set_reg <- levels(test_data$region)

setdiff(train_set_reg,test_set_reg)

#Clean-up some space
rm(train_set_dept,train_set_edu,train_set_reg,test_set_dept,test_set_edu,test_set_reg)

#Normalize and dummify the training data
std_dumified_train_data <- processed_data
colnames(std_dumified_train_data)

#Remove the employee_id variable which is of no use
std_dumified_train_data$employee_id <- NULL

#standardize
std_dumified_train_data <- normalize(std_dumified_train_data, method = "standardize", range = c(0, 1), margin = 1L, on.constant = "quiet")

#dummify factor columns, excluding the target variable
std_dumified_train_data <- dummy.data.frame(data = std_dumified_train_data[,!names(std_dumified_train_data) %in% 'is_promoted'],drop = F)

#Add the target variable back to the normalized and dummified dataframe
std_dumified_train_data$is_promoted <- processed_data$is_promoted



