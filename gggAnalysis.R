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

train <- vroom("./train.csv")
test <- vroom("./test.csv")
train_na <- vroom("./trainWithMissingValues.csv")

columns_with_na <- names(train_na)[which(colSums(is.na(train_na)) > 0)]
print(columns_with_na)

na_recipe <- recipe(type ~., train_na) %>% 
  step_impute_mean(bone_length, rotting_flesh, hair_length)

prepped_na_recipe <- prep(na_recipe)
baked <- bake(prepped_na_recipe, new_data = train_na)
baked

rmse_vec(train[is.na(train_na)], baked[is.na(train_na)])

