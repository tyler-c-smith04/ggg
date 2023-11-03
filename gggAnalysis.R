#################################
# ggg Analysis
#################################

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding
library(ranger)
library(rpart)
library(discrim)
library(naivebayes)
library(kknn)
library(doParallel)
library(themis)
library(stacks)
library(kernlab)

train <- vroom("./train.csv")
test <- vroom("./test.csv")
# train_na <- vroom("./trainWithMissingValues.csv")

columns_with_na <- names(train_na)[which(colSums(is.na(train_na)) > 0)]
print(columns_with_na)

na_recipe <- recipe(type ~., train_na) %>% 
  step_impute_mean(bone_length, rotting_flesh, hair_length)

prepped_na_recipe <- prep(na_recipe)
baked <- bake(prepped_na_recipe, new_data = train_na)
baked

rmse_vec(train[is.na(train_na)], baked[is.na(train_na)])

predict_and_format <- function(model, newdata, filename){
  predictions <- predict(model, new_data = newdata)
  
  submission <- predictions %>% 
    mutate(id = test$id) %>% 
    rename("type" = ".pred_class") %>% 
    select(2,1)
  
  vroom_write(submission, filename, delim = ',')
}

my_recipe <- recipe(type ~ ., data = train) %>%
  step_rm(id) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(color)

# KNN --------------------------------------------------------------------

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

# cross validation
knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 5)

knn_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tuning_grid,
            metrics = metric_set(roc_auc))

knn_bestTune <- CV_results %>%
  select_best("roc_auc")

# finalize workflow
final_knn_wf <- knn_wf %>%
  finalize_workflow(knn_bestTune) %>%
  fit(data = train)

predict_and_format(final_knn_wf, test, "./knn_preds.csv")

# Random Forests

rand_forest_mod <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees=500) %>% # or 1000
  set_engine("ranger") %>%
  set_mode("classification")

rand_forest_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
                                        min_n(),
                                        levels = 5) ## L^2 total tuning possibilities

## Split data for CV
forest_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- rand_forest_workflow %>%
  tune_grid(resamples = forest_folds,
            grid = rand_forest_tuning_grid,
            metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy

## Find Best Tuning Parameters
forest_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_forest_wf <- rand_forest_workflow %>%
  finalize_workflow(forest_bestTune) %>%
  fit(data = train)

predict_and_format(final_forest_wf, test, "./rand_forest_preds.csv")


# Naive Bayes -------------------------------------------------------------
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode('classification') %>%
  set_engine('naivebayes')

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

## Split data for CV
nb_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_grid,
            metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy

## Find Best Tuning Parameters
nb_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_nb_wf <- nb_wf %>%
  finalize_workflow(nb_bestTune) %>%
  fit(data = train)

# Predict
predict(final_nb_wf, new_data = test, type = 'prob')

predict_and_format(final_nb_wf, test, "./nb_preds.csv")


# SVM ---------------------------------------------------------------------
svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab')

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

# cross validation
svm_tuning_grid <- grid_regular(cost(),
                                rbf_sigma(),
                                levels = 5)

svm_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples = svm_folds,
            grid = svm_tuning_grid,
            metrics = metric_set(roc_auc))

svm_bestTune <- CV_results %>%
  select_best("roc_auc")

# finalize workflow
final_svm_wf <- svm_wf %>%
  finalize_workflow(svm_bestTune) %>%
  fit(data = train)

predict_and_format(final_svm_wf, test, "./svm_radial_preds.csv")

