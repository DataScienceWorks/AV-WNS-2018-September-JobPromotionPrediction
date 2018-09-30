library(h2o)

response <- 'is_promoted'
predictors <- setdiff(names(std_dumified_train_data),response)

# Start H2O on the local machine using all available cores and with 2 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "2g")
h2o.removeAll()
h2o.ls()

origtrain.hex <- as.h2o(std_dumified_train_data, destination_frame = 'origtrain.hex')

promotion.splits <- h2o.splitFrame(data =origtrain.hex, ratios = c(0.7), seed = 1234)
train <- promotion.splits[[1]]
validation <- promotion.splits[[2]]

# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = train, destination_frame = "origtrain.hex")

h2o.ls()

# Import a local R test data frame to the H2O cloud
validation.hex <- as.h2o(x = validation, destination_frame = "validation.hex")

# Generate a random grid of models and stack them together

# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 5,
                        seed = 123)
# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid_binomial",
                     x = predictors,
                     y = response,
                     training_frame = train,
                     ntrees = 1500,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = predictors,
                                y = response,
                                training_frame = train,
                                model_id = "ensemble_gbm_grid_binomial",
                                base_models = gbm_grid@model_ids)

# Eval ensemble performance on a test set
perf_en_valid <- h2o.performance(ensemble, newdata = validation)

# Compare to base learner performance on the test set
.getaucs <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = validation))
baselearner_aucs <- sapply(gbm_grid@model_ids, .getaucs)
baselearner_best_auc_valid <- max(baselearner_aucs)
ensemble_auc_valid <- h2o.auc(perf_en_valid)
print(sprintf("Best Base-learner Validation AUC:  %s", baselearner_best_auc_valid))
print(sprintf("Ensemble Validation AUC:  %s", ensemble_auc_valid))

# Actual test data nomralizing & dummifying
std_dumified_test_data <- test_data
std_dumified_test_data$employee_id <- NULL

std_dumified_test_data <- normalize(std_dumified_test_data, method = "standardize", range = c(0, 1), margin = 1L, on.constant = "quiet")
std_dumified_test_data <- dummy.data.frame(data = std_dumified_test_data,drop = F)

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = std_dumified_test_data, destination_frame = "test.hex")

# Predict on actual test data set
actual.predict.hex = h2o.predict(ensemble, 
                                 newdata = test.hex)

data_en = h2o.cbind(actual.predict.hex)

# Copy predictions from H2O to R
test_en = as.data.frame(data_en)

head(test_en)
is_promoted <- test_en$predict
employee_id <- test_data$employee_id

output_test_pred_en = data.frame(cbind(employee_id,is_promoted))

summary(output_test_pred_en)

output_test_pred_en$employee_id <- as.character(output_test_pred_en$employee_id)
output_test_pred_en$is_promoted <- as.character(ifelse(output_test_pred_en$is_promoted==1,0,1))

table(output_test_pred_en$is_promoted)

write.csv(x = output_test_pred_en,file = 'STACKED_ENSEMBLE_GBM_GRID_1.5K_TREES.csv',row.names=FALSE)
