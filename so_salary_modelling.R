library(tidyverse)
library(caret)
library(tidymodels)

# read in survey data
survey <- read_csv("developer_survey_2019/survey_results_public.csv")

# filter data based on country and prof coders
prof_coder <- survey %>%
  filter(Employment == "Employed full-time",
         CompFreq == "Yearly") %>%
  mutate(YearsCodePro = ifelse(YearsCodePro == "Less than 1 Year", 1, as.numeric(YearsCodePro)))

us <- prof_coder %>%
  filter(CurrencySymbol == "USD")

eu <- prof_coder %>%
  filter(CurrencySymbol == "EUR")

uk <- prof_coder %>% 
  filter(CurrencySymbol == "GBP")

row <- prof_coder %>%
  filter(!CurrencySymbol %in% c("USD", "EURO", "GBP"))

# define predictors
preds <- c("EdLevel", "DevType", "YearsCodePro", "LanguageWorkedWith")

# keep useful data for modelling
# use local currency for modelling EU and UK. use USD for US and ROW

us <- us %>%
  select(c("ConvertedComp", preds)) %>%
  rename(salary = ConvertedComp) %>%
  filter(!is.na(salary))

eu <- eu %>%
  select(c("CompTotal", preds)) %>%
  rename(salary = CompTotal) %>%
  filter(!is.na(salary))

uk <- uk %>%
  select(c("CompTotal", preds)) %>%
  rename(salary = CompTotal) %>%
  filter(!is.na(salary))

row <- row %>%
  select(c("ConvertedComp", preds)) %>%
  rename(salary = ConvertedComp) %>%
  filter(!is.na(salary))


# define recipe

## get all dev types
devtypes <- str_split(uk$DevType, ";") %>% unlist() %>% unique()
languages <- str_split(uk$LanguageWorkedWith, ";") %>% unlist() %>% unique()

# define recipe
rec <- recipe(salary ~ ., data = uk)

step_regex2 <- function(rec, x, col) {
  
  col <- enquo(col)
  col_name <- quo_name(col)
  
  step_regex(rec, !!col, pattern = x, options = list(fixed = TRUE))
  
}

rec_update <- reduce(devtypes[!is.na(devtypes)], step_regex2, col = DevType, .init = rec) %>% 
  reduce(languages[!is.na(languages)], step_regex2, col = LanguageWorkedWith, .init = .) %>%
  update_role(DevType, new_role = "not_use") %>%
  update_role(LanguageWorkedWith, new_role = "not_use") %>%
  step_modeimpute(EdLevel) %>%
  step_other(EdLevel) %>%
  step_dummy(EdLevel) %>%
  step_medianimpute(YearsCodePro) %>%
  step_center(YearsCodePro, salary) %>%
  step_scale(YearsCodePro, salary) %>%
  step_zv(all_predictors())

# start training
set.seed(1234)
uk_split <- initial_split(uk)
uk_train <- training(uk_split)
uk_test <- testing(uk_split)

us_split <- initial_split(us)
us_train <- training(us_split)
us_test <- testing(us_split)


## first glmnet
ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

glmn_grid <- expand.grid(alpha = c(0, 0.25), 
                         lambda = 10^seq(-3, 0, length = 20))


test <- prep(rec_update, training = uk_train, verbose = TRUE) %>% 
  juice()
  

uk_glmnet <- train(rec_update, data = uk_train,
                   method = "glmnet",
                   trControl = ctrl,
                   tuneGrid = glmn_grid)
plot(uk_glmnet)
plot(uk_glmnet, metric = "Rsquared")


us_glmnet <- train(rec_update, data = us_train,
                   method = "glmnet",
                   trControl = ctrl,
                   tuneGrid = glmn_grid)


## gbm
uk_gbm <- train(rec_update, data = uk_train,
                method = "gbm",
                trControl = ctrl)

plot(uk_gbm)
plot(uk_gbm, metric = "Rsquared")


## random forrest
uk_rf <- train(rec_update, data = uk_train,
               method = "ranger",
               trControl = ctrl)

plot(uk_rf)

## neural network
uk_keras <- train(rec_update, data = uk_train,
                  method = "mlpKerasDropout",
                  trControl = ctrl)





