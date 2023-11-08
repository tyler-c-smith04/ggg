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
library(keras)
library(tensorflow)
library(bonsai)
library(lightgbm)
library(dbarts)

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

# Neural Networks ---------------------------------------------------------
nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = 'id') %>% 
  step_mutate(color = as.factor(color)) %>% 
  step_dummy(color) %>% 
  step_range(all_numeric_predictors(), min = 0, max = 1) # Scale to [0,1]

nn_mod <- mlp(hidden_units = tune(),
                epochs = 50,
                ) %>% 
  set_engine('nnet') %>% 
  set_mode('classification')

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_mod)

nn_tuning_grid <- grid_regular(hidden_units(range = c(1, 50)),
                            levels = 5)

nn_folds <- vfold_cv(train, v = 5, repeats = 1)

tuned_nn <- nn_wf %>%
  tune_grid(
    resamples = nn_folds,
    grid = nn_tuning_grid,
    metrics = metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
   filter(.metric=="accuracy") %>%
   ggplot(aes(x=hidden_units, y=mean)) + geom_line()

CV_results <- nn_wf %>%
  tune_grid(resamples = nn_folds,
            grid = nn_tuning_grid,
            metrics = metric_set(accuracy))

nn_bestTune <- CV_results %>%
  select_best("accuracy")

final_nn_wf <- nn_wf %>%
  finalize_workflow(nn_bestTune) %>%
  fit(data = train)

predict_and_format(final_nn_wf, test, "./nn_preds.csv")

# Boosted Trees --------------------------------------------------
boost_mod <- boost_tree(tree_depth = tune(),
                        trees = tune(),
                        learn_rate = tune()) %>% 
  set_engine('lightgbm') %>%
  set_mode('classification')

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_mod)

boost_tuning_grid <- grid_regular(tree_depth(),
                                  trees(),
                                  learn_rate(),
                               levels = 5)

boost_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- boost_wf %>%
  tune_grid(resamples = boost_folds,
            grid = boost_tuning_grid,
            metrics = metric_set(accuracy))

boost_bestTune <- CV_results %>%
  select_best("accuracy")

final_boost_wf <- boost_wf %>%
  finalize_workflow(boost_bestTune) %>%
  fit(data = train)

predict_and_format(final_boost_wf, test, "./boost_preds.csv")

# BART --------------------------------------------------------------------
bart_mod <- parsnip::bart(trees = tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_mod)

bart_tuning_grid <- grid_regular(trees(),
                                  levels = 5)

bart_folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- bart_wf %>%
  tune_grid(resamples = bart_folds,
            grid = bart_tuning_grid,
            metrics = metric_set(accuracy))

bart_bestTune <- CV_results %>%
  select_best("accuracy")

final_bart_wf <- bart_wf %>%
  finalize_workflow(bart_bestTune) %>%
  fit(data = train)

predict_and_format(final_bart_wf, test, "./bart_preds.csv")
