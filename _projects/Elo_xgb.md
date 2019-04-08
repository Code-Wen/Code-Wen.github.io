# 1. Data Pre-Processing

## 1.1 Load the data

``` r
## check data
# load train
train <- fread('data/train.csv')
test <- fread('data/test.csv')
sample_submission <- fread('data/sample_submission.csv')
history <- fread('data/historical_transactions.csv')
new <- fread('data/new_merchant_transactions.csv')
merchant <- fread('data/merchants.csv')
```
## 1.2 Examing the data and feature engineering

After a quick examination of the data tables, we listed out features that we think that are important. We also list how we are going to manipulate on these features as well as the codes.

### 1.2.1 The `train` table

1.  card age, which can be calculated from `first_active_month`
2.  `feature_1`, `feature_2`, and `feature_3`. We don't know what these represent for since the data is anonymized. But there are only a few categories in each of these categorical variables, so we can simply transform them into factors.

Code for processing the features from `train`:

``` r

target <- train[, .(card_id, target)]
train$target <- NULL
# save ids for train and test
id_tr <- train$card_id
id_te <- test$card_id
# bind train and test
data <- rbind(train, test)
ind_1 <- paste0('feature_1_', 1:5)
ind_2 <- paste0('feature_2_', 1:3)
ind_3 <- paste0('feature_3_', 0:1)
data[, `:=`(years_open = as.numeric(as.Date('2018-02-15') - as.Date( paste0(first_active_month,'-15')) )/365,
            first_active_month = NULL,
            feature_1 = factor(feature_1, levels = 1:5, labels = ind_1),
            feature_2 = factor(feature_2, levels = 1:3, labels = ind_2),
            feature_3 = factor(feature_3, levels = 0:1, labels = ind_3)
            )]
data[, (ind_1[-1]) := lapply(ind_1[-1], function(x) as.numeric(feature_1 == x))]
data[, (ind_2[-1]) := lapply(ind_2[-1], function(x) as.numeric(feature_2 == x))]
data[, (ind_3[-1]) := lapply(ind_3[-1], function(x) as.numeric(feature_3 == x))]
data[, `:=` (feature_1 = NULL, feature_2 = NULL, feature_3 = NULL)]
rm(train, test); invisible(gc())
```

### 1.2.2 The `history` table


1.  the frequency of transactions being denied (both count and ratio), which can be calculated from the column `authorized_flag`
2.  `category_1`, `category_2`, `category_3`, these anonymized variables will be transformed into factors, then each factor level would be tallied level for each card. For each card, we can also look at the distribution among each category.
3.  card usage over time, we can calculate a mean and a variance of the card usage over time.
4.  card debt situation, we can calculate the number of installments the card has and how that changes over time; another index would be the purchase amount
5.  the categories of purchases, customers might be more loyal to certain categories of purchase, e.g., I have a travel card dedicated for travel, `merchant_category_id` and `subsector_id` can be used to identify the spread and the mode of merchant category for each card.

The `historical_transactions.csv` data is more complex. Let's take a look at the summary first.

``` r
sum_hist <- history[!(purchase_amount > 100 & authorized_flag == 'N'), 
                    .(n_trans = .N,  # number of transactions
                      n_denied = sum(authorized_flag == 'N', na.rm = T), # number of denied transactions
                      ratio_denied = sum(authorized_flag == 'N', na.rm = T) / .N, # the ratio of denied transactions
                      
                      cat_1_N = sum(category_1 == 'N', na.rm = T),  # category 1 level counts
                      cat_1_Y = sum(category_1 == 'Y', na.rm = T),
                      
                      cat_2_1 = sum(category_2 == 1, na.rm = T),  # category 2 level counts
                      cat_2_2 = sum(category_2 == 2, na.rm = T),
                      cat_2_3 = sum(category_2 == 3, na.rm = T),
                      cat_2_4 = sum(category_2 == 4, na.rm = T),
                      cat_2_5 = sum(category_2 == 5, na.rm = T),
                      
                      cat_3_A = sum(category_3 == 'A', na.rm = T),  # category 3 level counts
                      cat_3_B = sum(category_3 == 'B', na.rm = T),
                      cat_3_C = sum(category_3 == 'C', na.rm = T),
                      
                      spend = sum(purchase_amount[authorized_flag == 'Y'], na.rm = T),
                      n_installment_outlier = sum(installments < 0 | installments > 99, na.rm = T),
                      mean_installment = mean(installments[installments >= 0 & installments < 99], na.rm = T),
                      
                      n_merchant = uniqueN(merchant_category_id),
                      n_subsector = uniqueN(paste0(merchant_category_id, subsector_id)),
                      
                      n_city = uniqueN(paste0(state_id, city_id))
                      
                     ), 
                    by = card_id]
```

Get some stats related to the trend of each card.

``` r
trend_hist <- history[!(purchase_amount > 100 & authorized_flag == 'N'), 
                    .(n_trans = .N,  # number of transactions
                      
                      ratio_denied = sum(authorized_flag == 'N', na.rm = T) / .N, # the ratio of denied transactions
                      
                      cat_1_N = sum(category_1 == 'N', na.rm = T),  # category 1 level counts
                      cat_2_1 = sum(category_2 == 1, na.rm = T),  # category 2 level counts
                      cat_2_2 = sum(category_2 == 2, na.rm = T),
                      cat_2_3 = sum(category_2 == 3, na.rm = T),
                      cat_2_4 = sum(category_2 == 4, na.rm = T),
                      
                      cat_3_A = sum(category_3 == 'A', na.rm = T),  # category 3 level counts
                      cat_3_B = sum(category_3 == 'B', na.rm = T),
                      
                      spend = sum(purchase_amount[authorized_flag == 'Y'], na.rm = T),
                      mean_installment = mean(installments[installments >= 0 & installments < 99], na.rm = T),
                      
                      n_merchant = uniqueN(merchant_category_id),
                      
                      n_city = uniqueN(paste0(state_id, city_id))
                      
                     ), 
                    by = .(card_id, month_lag)]

# trend_hist[is.na(mean_installment), mean_installment := (-1)]
```

Now, summarize trend into the variance over time and the delta over time.

``` r
rm(history); invisible(gc())

sum_trend_hist <- trend_hist[, .(
                             v_n_trans = var(n_trans),
                             d_n_trans = n_trans[which.max(month_lag)] -
                               n_trans[which.min(month_lag)],
                             
                             v_ratio_denied = var(ratio_denied),
                             d_ratio_denied = ratio_denied[which.max(month_lag)] -
                               ratio_denied[which.min(month_lag)],
                             
                             v_cat_1_N = var(cat_1_N),
                             d_cat_1_N = cat_1_N[which.max(month_lag)] -
                               cat_1_N[which.min(month_lag)],
                             
                             v_cat_2_1 = var(cat_2_1),
                             d_cat_2_1 = cat_2_1[which.max(month_lag)] -
                               cat_2_1[which.min(month_lag)],
                             
                             v_cat_2_2 = var(cat_2_2),
                             d_cat_2_2 = cat_2_2[which.max(month_lag)] - 
                               cat_2_2[which.min(month_lag)],
                             
                             v_cat_2_3 = var(cat_2_3),
                             d_cat_2_3 = cat_2_3[which.max(month_lag)] -
                               cat_2_3[which.min(month_lag)],
                             
                             v_cat_2_4 = var(cat_2_4),
                             d_cat_2_4 = cat_2_4[which.max(month_lag)] - 
                               cat_2_4[which.min(month_lag)],
                             
                             v_cat_3_A = var(cat_3_A),
                             d_cat_3_A = cat_3_A[which.max(month_lag)] - 
                               cat_3_A[which.min(month_lag)],
                             
                             v_cat_3_B = var(cat_3_B),
                             d_cat_3_B = cat_3_B[which.max(month_lag)] - 
                               cat_3_B[which.min(month_lag)],
                             
                             v_spend = var(spend),
                             d_spend = spend[which.max(month_lag)] - 
                               spend[which.min(month_lag)],
                             
                             v_mean_installment = var(mean_installment),
                             d_mean_installment =
                               mean_installment[which.max(month_lag)] - 
                               mean_installment[which.min(month_lag)],
                             
                             v_n_merchant = var(n_merchant),
                             d_n_merchant = n_merchant[which.max(month_lag)] -
                               n_merchant[which.min(month_lag)],
                             
                             v_n_city = var(n_city),
                             d_n_city = n_city[which.max(month_lag)] - 
                               n_city[which.min(month_lag)]
                             ), by = card_id
                             ]

# sum_trend_hist[is.na(d_mean_installment), d_mean_installment := 0]
# sum_trend_hist[is.na(v_mean_installment), v_mean_installment := 0]
```

### 1.2.3 The `new` table

The structure of this table is exactly the same as that of the `history` table. The only difference is that the transction dates are newer, which may lead to different weights/treatments in the model. We apply the same engineering process to this table as we did for  `history`.


``` r
sum_new <- new[!(purchase_amount > 100 & authorized_flag == 'N'), 
                    .(n_trans = .N,  # number of transactions
                      n_denied = sum(authorized_flag == 'N', na.rm = T), # number of denied transactions
                      ratio_denied = sum(authorized_flag == 'N', na.rm = T) / .N, # the ratio of denied transactions
                      
                      cat_1_N = sum(category_1 == 'N', na.rm = T),  # category 1 level counts
                      cat_1_Y = sum(category_1 == 'Y', na.rm = T),
                      
                      cat_2_1 = sum(category_2 == 1, na.rm = T),  # category 2 level counts
                      cat_2_2 = sum(category_2 == 2, na.rm = T),
                      cat_2_3 = sum(category_2 == 3, na.rm = T),
                      cat_2_4 = sum(category_2 == 4, na.rm = T),
                      cat_2_5 = sum(category_2 == 5, na.rm = T),
                      
                      cat_3_A = sum(category_3 == 'A', na.rm = T),  # category 3 level counts
                      cat_3_B = sum(category_3 == 'B', na.rm = T),
                      cat_3_C = sum(category_3 == 'C', na.rm = T),
                      
                      spend = sum(purchase_amount[authorized_flag == 'Y'], na.rm = T),
                      n_installment_outlier = sum(installments < 0 | installments > 99, na.rm = T),
                      mean_installment = mean(installments[installments >= 0 & installments < 99], na.rm = T),
                      
                      n_merchant = uniqueN(merchant_category_id),
                      n_subsector = uniqueN(paste0(merchant_category_id, subsector_id)),
                      
                      n_city = uniqueN(paste0(state_id, city_id))
                      
                     ), 
                    by = card_id]
# sum_new[is.na(mean_installment), mean_installment := (-1)]
```

Get some stats related to the trend of each card in the new transactions.

``` r
trend_new <- new[!(purchase_amount > 100 & authorized_flag == 'N'), 
                    .(n_trans = .N,  # number of transactions
                      
                      ratio_denied = sum(authorized_flag == 'N', na.rm = T) / .N, # the ratio of denied transactions
                      
                      cat_1_N = sum(category_1 == 'N', na.rm = T),  # category 1 level counts
                      cat_2_1 = sum(category_2 == 1, na.rm = T),  # category 2 level counts
                      cat_2_2 = sum(category_2 == 2, na.rm = T),
                      cat_2_3 = sum(category_2 == 3, na.rm = T),
                      cat_2_4 = sum(category_2 == 4, na.rm = T),
                      
                      cat_3_A = sum(category_3 == 'A', na.rm = T),  # category 3 level counts
                      cat_3_B = sum(category_3 == 'B', na.rm = T),
                      
                      spend = sum(purchase_amount[authorized_flag == 'Y'], na.rm = T),
                      mean_installment = mean(installments[installments >= 0 & installments < 99], na.rm = T),
                      
                      n_merchant = uniqueN(merchant_category_id),
                      
                      n_city = uniqueN(paste0(state_id, city_id))
                      
                     ), 
                    by = .(card_id, month_lag)]

# trend_new[is.na(mean_installment), mean_installment := (-1)]
```

Now, summarize trend into the variance over time and the delta over time in the new transactions data.

``` r
rm(new); invisible(gc())

sum_trend_new <- trend_new[, .(
                             v_n_trans = var(n_trans),
                             d_n_trans = n_trans[which.max(month_lag)] -
                               n_trans[which.min(month_lag)],
                             
                             v_ratio_denied = var(ratio_denied),
                             d_ratio_denied = ratio_denied[which.max(month_lag)] -
                               ratio_denied[which.min(month_lag)],
                             
                             v_cat_1_N = var(cat_1_N),
                             d_cat_1_N = cat_1_N[which.max(month_lag)] -
                               cat_1_N[which.min(month_lag)],
                             
                             v_cat_2_1 = var(cat_2_1),
                             d_cat_2_1 = cat_2_1[which.max(month_lag)] -
                               cat_2_1[which.min(month_lag)],
                             
                             v_cat_2_2 = var(cat_2_2),
                             d_cat_2_2 = cat_2_2[which.max(month_lag)] - 
                               cat_2_2[which.min(month_lag)],
                             
                             v_cat_2_3 = var(cat_2_3),
                             d_cat_2_3 = cat_2_3[which.max(month_lag)] -
                               cat_2_3[which.min(month_lag)],
                             
                             v_cat_2_4 = var(cat_2_4),
                             d_cat_2_4 = cat_2_4[which.max(month_lag)] - 
                               cat_2_4[which.min(month_lag)],
                             
                             v_cat_3_A = var(cat_3_A),
                             d_cat_3_A = cat_3_A[which.max(month_lag)] - 
                               cat_3_A[which.min(month_lag)],
                             
                             v_cat_3_B = var(cat_3_B),
                             d_cat_3_B = cat_3_B[which.max(month_lag)] - 
                               cat_3_B[which.min(month_lag)],
                             
                             v_spend = var(spend),
                             d_spend = spend[which.max(month_lag)] - 
                               spend[which.min(month_lag)],
                             
                             v_mean_installment = var(mean_installment),
                             d_mean_installment =
                               mean_installment[which.max(month_lag)] - 
                               mean_installment[which.min(month_lag)],
                             
                             v_n_merchant = var(n_merchant),
                             d_n_merchant = n_merchant[which.max(month_lag)] -
                               n_merchant[which.min(month_lag)],
                             
                             v_n_city = var(n_city),
                             d_n_city = n_city[which.max(month_lag)] - 
                               n_city[which.min(month_lag)]
                             ), by = card_id
                             ]

# sum_trend_new[is.na(v_n_trans), v_n_trans := 0]
# sum_trend_new[is.na(v_ratio_denied), v_ratio_denied := 0]
# sum_trend_new[is.na(v_cat_1_N), v_cat_1_N := 0]
# sum_trend_new[is.na(v_cat_2_1), v_cat_2_1 := 0]
# sum_trend_new[is.na(v_cat_2_2), v_cat_2_2 := 0]
# sum_trend_new[is.na(v_cat_2_3), v_cat_2_3 := 0]
# sum_trend_new[is.na(v_cat_2_4), v_cat_2_4 := 0]
# sum_trend_new[is.na(v_cat_3_A), v_cat_3_A := 0]
# sum_trend_new[is.na(v_cat_3_B), v_cat_3_B := 0]
# sum_trend_new[is.na(v_spend), v_spend := 0]
# sum_trend_new[is.na(v_mean_installment), v_mean_installment := 0]
# sum_trend_new[is.na(v_n_merchant), v_n_merchant := 0]
# sum_trend_new[is.na(v_n_city), v_n_city := 0]
```

## 1.3 Merge the features

We merge all the features we obtained so far into a single table `train_dt`.

``` r
data_dt <- merge( merge( merge( merge(data, sum_hist, by = 'card_id', all.x = T),
                   sum_trend_hist, by = 'card_id' , all.x = T),
                   sum_new, by = 'card_id', all.x = T),
                   sum_trend_new, by = 'card_id', all.x = T)
train_dt <- data_dt[card_id %in% id_tr,]
train_dt <- merge(train_dt, target, by = 'card_id', all.x = T)


test_dt <- data_dt[card_id %in% id_te,]
save(test_dt, train_dt, target, file = 'elo_xgb_v3.Rdata')
```
## 1.4 Additional notes about the data


1.  About the purchase amount, there are some extreme high values in `purchase_amount` in transactions. While the majority of the values are below 0, some values are up to 6 million! We looked at these data and seems they might be entry errors since a lot of them were not authorized, e.g., wrong digits by the cashier/merchant. We can exclude these transactions if they were not authorized.
2.  There are a lot of `NA`s in the `category_2` in historical transactions data, these seem to be out-of-country transactions, since the `state_id` and the `city_id` were both `-1` for these transactions, we can calculate the number of these transactions, which can be a proxy be how much/often the customzer traveled abroad.
3.  There are some `-1`s for `merchant_category_id` and `subsector_id`, a lot of which is due to transactions happening abroad (`state_id = -1`, probably in foreign language and hard to classify). There were about 40 with normal state and city id, these are probably just peculiar transactions via platforms like esty.




# 2. Training the model: `xgboost`

We chose to use `xgboost` as our model.

``` r
set.seed(33)
train_dt <- train_dt[target >= (-10) & target <= 10]
# use mean instead of rmse for measure aggregation
train_dt[is.nan(d_mean_installment.x), d_mean_installment.x := NA]
train_dt[is.nan(d_mean_installment.y), d_mean_installment.y := NA]
train_dt[is.nan(mean_installment.y), mean_installment.y := NA]
train_dt$card_id <- NULL

#### Train Model ####
start.time <- Sys.time()
train_task <- makeRegrTask(data = train_dt, target = 'target')
getParamSet('regr.xgboost')
```

    ##                                 Type  len        Def               Constr
    ## booster                     discrete    -     gbtree gbtree,gblinear,dart
    ## watchlist                    untyped    -     <NULL>                    -
    ## eta                          numeric    -        0.3               0 to 1
    ## gamma                        numeric    -          0             0 to Inf
    ## max_depth                    integer    -          6             1 to Inf
    ## min_child_weight             numeric    -          1             0 to Inf
    ## subsample                    numeric    -          1               0 to 1
    ## colsample_bytree             numeric    -          1               0 to 1
    ## colsample_bylevel            numeric    -          1               0 to 1
    ## num_parallel_tree            integer    -          1             1 to Inf
    ## lambda                       numeric    -          1             0 to Inf
    ## lambda_bias                  numeric    -          0             0 to Inf
    ## alpha                        numeric    -          0             0 to Inf
    ## objective                    untyped    - reg:linear                    -
    ## eval_metric                  untyped    -       rmse                    -
    ## base_score                   numeric    -        0.5          -Inf to Inf
    ## max_delta_step               numeric    -          0             0 to Inf
    ## missing                      numeric    -                     -Inf to Inf
    ## monotone_constraints   integervector <NA>          0              -1 to 1
    ## tweedie_variance_power       numeric    -        1.5               1 to 2
    ## nthread                      integer    -          -             1 to Inf
    ## nrounds                      integer    -          1             1 to Inf
    ## feval                        untyped    -     <NULL>                    -
    ## verbose                      integer    -          1               0 to 2
    ## print_every_n                integer    -          1             1 to Inf
    ## early_stopping_rounds        integer    -     <NULL>             1 to Inf
    ## maximize                     logical    -     <NULL>                    -
    ## sample_type                 discrete    -    uniform     uniform,weighted
    ## normalize_type              discrete    -       tree          tree,forest
    ## rate_drop                    numeric    -          0               0 to 1
    ## skip_drop                    numeric    -          0               0 to 1
    ## callbacks                    untyped    -     <list>                    -
    ##                        Req Tunable Trafo
    ## booster                  -    TRUE     -
    ## watchlist                -   FALSE     -
    ## eta                      -    TRUE     -
    ## gamma                    -    TRUE     -
    ## max_depth                -    TRUE     -
    ## min_child_weight         -    TRUE     -
    ## subsample                -    TRUE     -
    ## colsample_bytree         -    TRUE     -
    ## colsample_bylevel        -    TRUE     -
    ## num_parallel_tree        -    TRUE     -
    ## lambda                   -    TRUE     -
    ## lambda_bias              -    TRUE     -
    ## alpha                    -    TRUE     -
    ## objective                -   FALSE     -
    ## eval_metric              -   FALSE     -
    ## base_score               -   FALSE     -
    ## max_delta_step           -    TRUE     -
    ## missing                  -   FALSE     -
    ## monotone_constraints     -    TRUE     -
    ## tweedie_variance_power   Y    TRUE     -
    ## nthread                  -   FALSE     -
    ## nrounds                  -    TRUE     -
    ## feval                    -   FALSE     -
    ## verbose                  -   FALSE     -
    ## print_every_n            Y   FALSE     -
    ## early_stopping_rounds    -   FALSE     -
    ## maximize                 -   FALSE     -
    ## sample_type              Y    TRUE     -
    ## normalize_type           Y    TRUE     -
    ## rate_drop                Y    TRUE     -
    ## skip_drop                Y    TRUE     -
    ## callbacks                -   FALSE     -

``` r
#make learner with inital parameters
learner <- makeLearner("regr.xgboost")

#define parameters for tuning
params <- makeParamSet(
      makeIntegerParam("max_depth",lower=1,upper=10),
      makeNumericParam("lambda",lower=0,upper=10),
      makeNumericParam("eta", lower = 0.001, upper = 0.3),
      makeNumericParam("subsample", lower = 0.10, upper = 0.80),
      makeNumericParam("min_child_weight",lower=0.9,upper=5),
      makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8))

#define search function
rancontrol <- makeTuneControlRandom(maxit = 30L) #do 100 iterations

# 5 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 5L)

# tune parameters
setAggregation(measure = rmse, aggr = test.mean)
```

    ## Name: Root mean squared error
    ## Performance measure: rmse
    ## Properties: regr,req.pred,req.truth
    ## Minimize: TRUE
    ## Best: 0; Worst: Inf
    ## Aggregated by: test.mean
    ## Arguments: 
    ## Note: The RMSE is aggregated as sqrt(mean(rmse.vals.on.test.sets^2)). If you don't want that, you could also use `test.mean`.

``` r
tune <- tuneParams(learner = learner, 
                       task = train_task, 
                       resampling = set_cv,
                       measures = rmse, 
                       par.set = params, 
                       control = rancontrol)
```

    ## [Tune] Started tuning learner regr.xgboost for parameter set:

    ##                     Type len Def       Constr Req Tunable Trafo
    ## max_depth        integer   -   -      1 to 10   -    TRUE     -
    ## lambda           numeric   -   -      0 to 10   -    TRUE     -
    ## eta              numeric   -   - 0.001 to 0.3   -    TRUE     -
    ## subsample        numeric   -   -   0.1 to 0.8   -    TRUE     -
    ## min_child_weight numeric   -   -     0.9 to 5   -    TRUE     -
    ## colsample_bytree numeric   -   -   0.2 to 0.8   -    TRUE     -

    ## With control class: TuneControlRandom

    ## Imputation value: Inf

    ## [Tune-x] 1: max_depth=8; lambda=4.81; eta=0.0345; subsample=0.255; min_child_weight=4.97; colsample_bytree=0.589

    ## [Tune-y] 1: rmse.test.rmse=1.7645043; time: 0.1 min

    ## [Tune-x] 2: max_depth=4; lambda=9.6; eta=0.227; subsample=0.272; min_child_weight=4.14; colsample_bytree=0.543

    ## [Tune-y] 2: rmse.test.rmse=1.7402907; time: 0.1 min

    ## [Tune-x] 3: max_depth=10; lambda=7.02; eta=0.257; subsample=0.744; min_child_weight=2.37; colsample_bytree=0.479

    ## [Tune-y] 3: rmse.test.rmse=1.7375269; time: 0.1 min

    ## [Tune-x] 4: max_depth=10; lambda=0.476; eta=0.281; subsample=0.438; min_child_weight=4.63; colsample_bytree=0.465

    ## [Tune-y] 4: rmse.test.rmse=1.7352470; time: 0.1 min

    ## [Tune-x] 5: max_depth=10; lambda=1.78; eta=0.147; subsample=0.52; min_child_weight=1.68; colsample_bytree=0.343

    ## [Tune-y] 5: rmse.test.rmse=1.7521340; time: 0.1 min

    ## [Tune-x] 6: max_depth=10; lambda=2.45; eta=0.0457; subsample=0.213; min_child_weight=2.34; colsample_bytree=0.235

    ## [Tune-y] 6: rmse.test.rmse=1.7676664; time: 0.1 min

    ## [Tune-x] 7: max_depth=5; lambda=4.94; eta=0.221; subsample=0.177; min_child_weight=2.09; colsample_bytree=0.796

    ## [Tune-y] 7: rmse.test.rmse=1.7084184; time: 0.1 min

    ## [Tune-x] 8: max_depth=6; lambda=5.1; eta=0.104; subsample=0.285; min_child_weight=4.5; colsample_bytree=0.265

    ## [Tune-y] 8: rmse.test.rmse=1.7587794; time: 0.1 min

    ## [Tune-x] 9: max_depth=2; lambda=1.3; eta=0.2; subsample=0.638; min_child_weight=4.18; colsample_bytree=0.464

    ## [Tune-y] 9: rmse.test.rmse=1.7444550; time: 0.1 min

    ## [Tune-x] 10: max_depth=6; lambda=2.69; eta=0.0741; subsample=0.187; min_child_weight=3.98; colsample_bytree=0.529

    ## [Tune-y] 10: rmse.test.rmse=1.7626382; time: 0.1 min

    ## [Tune-x] 11: max_depth=5; lambda=2.54; eta=0.187; subsample=0.658; min_child_weight=1.14; colsample_bytree=0.572

    ## [Tune-y] 11: rmse.test.rmse=1.7236885; time: 0.1 min

    ## [Tune-x] 12: max_depth=5; lambda=0.0369; eta=0.277; subsample=0.259; min_child_weight=2.14; colsample_bytree=0.696

    ## [Tune-y] 12: rmse.test.rmse=1.6928579; time: 0.1 min

    ## [Tune-x] 13: max_depth=6; lambda=9.18; eta=0.0253; subsample=0.593; min_child_weight=3.25; colsample_bytree=0.748

    ## [Tune-y] 13: rmse.test.rmse=1.7663249; time: 0.1 min

    ## [Tune-x] 14: max_depth=1; lambda=0.726; eta=0.177; subsample=0.103; min_child_weight=4.11; colsample_bytree=0.701

    ## [Tune-y] 14: rmse.test.rmse=1.7328596; time: 0.1 min

    ## [Tune-x] 15: max_depth=3; lambda=4.66; eta=0.0759; subsample=0.77; min_child_weight=3.97; colsample_bytree=0.678

    ## [Tune-y] 15: rmse.test.rmse=1.7536960; time: 0.1 min

    ## [Tune-x] 16: max_depth=8; lambda=0.389; eta=0.158; subsample=0.248; min_child_weight=2.62; colsample_bytree=0.342

    ## [Tune-y] 16: rmse.test.rmse=1.7505078; time: 0.1 min

    ## [Tune-x] 17: max_depth=10; lambda=5.68; eta=0.138; subsample=0.65; min_child_weight=4.69; colsample_bytree=0.777

    ## [Tune-y] 17: rmse.test.rmse=1.7278849; time: 0.1 min

    ## [Tune-x] 18: max_depth=8; lambda=4.23; eta=0.114; subsample=0.544; min_child_weight=1.92; colsample_bytree=0.722

    ## [Tune-y] 18: rmse.test.rmse=1.7361475; time: 0.1 min

    ## [Tune-x] 19: max_depth=8; lambda=5.52; eta=0.192; subsample=0.659; min_child_weight=2.85; colsample_bytree=0.65

    ## [Tune-y] 19: rmse.test.rmse=1.7211587; time: 0.1 min

    ## [Tune-x] 20: max_depth=7; lambda=3.24; eta=0.125; subsample=0.456; min_child_weight=1.86; colsample_bytree=0.212

    ## [Tune-y] 20: rmse.test.rmse=1.7558817; time: 0.1 min

    ## [Tune-x] 21: max_depth=4; lambda=1.87; eta=0.112; subsample=0.536; min_child_weight=2.41; colsample_bytree=0.692

    ## [Tune-y] 21: rmse.test.rmse=1.7402109; time: 0.1 min

    ## [Tune-x] 22: max_depth=6; lambda=7; eta=0.162; subsample=0.609; min_child_weight=1.09; colsample_bytree=0.401

    ## [Tune-y] 22: rmse.test.rmse=1.7494700; time: 0.1 min

    ## [Tune-x] 23: max_depth=2; lambda=2.52; eta=0.091; subsample=0.786; min_child_weight=2.7; colsample_bytree=0.599

    ## [Tune-y] 23: rmse.test.rmse=1.7508730; time: 0.1 min

    ## [Tune-x] 24: max_depth=9; lambda=9.59; eta=0.25; subsample=0.679; min_child_weight=4.35; colsample_bytree=0.595

    ## [Tune-y] 24: rmse.test.rmse=1.7072511; time: 0.1 min

    ## [Tune-x] 25: max_depth=5; lambda=6.98; eta=0.245; subsample=0.221; min_child_weight=2.99; colsample_bytree=0.269

    ## [Tune-y] 25: rmse.test.rmse=1.7398043; time: 0.1 min

    ## [Tune-x] 26: max_depth=2; lambda=4.66; eta=0.032; subsample=0.592; min_child_weight=1.99; colsample_bytree=0.345

    ## [Tune-y] 26: rmse.test.rmse=1.7696255; time: 0.1 min

    ## [Tune-x] 27: max_depth=1; lambda=8.31; eta=0.245; subsample=0.52; min_child_weight=2.98; colsample_bytree=0.563

    ## [Tune-y] 27: rmse.test.rmse=1.7193074; time: 0.1 min

    ## [Tune-x] 28: max_depth=7; lambda=8.81; eta=0.177; subsample=0.559; min_child_weight=4.28; colsample_bytree=0.704

    ## [Tune-y] 28: rmse.test.rmse=1.7171898; time: 0.1 min

    ## [Tune-x] 29: max_depth=6; lambda=0.525; eta=0.0758; subsample=0.229; min_child_weight=3.81; colsample_bytree=0.596

    ## [Tune-y] 29: rmse.test.rmse=1.7528551; time: 0.1 min

    ## [Tune-x] 30: max_depth=9; lambda=6.48; eta=0.201; subsample=0.678; min_child_weight=4.82; colsample_bytree=0.247

    ## [Tune-y] 30: rmse.test.rmse=1.7454749; time: 0.1 min

    ## [Tune] Result: max_depth=5; lambda=0.0369; eta=0.277; subsample=0.259; min_child_weight=2.14; colsample_bytree=0.696 : rmse.test.rmse=1.6928579

``` r
# set parameters
new_learner <- setHyperPars(learner = learner, par.vals = tune$x)

## train model
new_model <- mlr::train(new_learner, train_task)

# get training time and print
end.time <- Sys.time()
print('Training Time:')
```

    ## [1] "Training Time:"

``` r
print(time.taken <- round(end.time-start.time,2))
```

    ## Time difference of 3.55 mins
    
# 3. Prediction

``` r
id_te <- test_dt$card_id
test_dt$card_id <- NULL
pred <- predict(new_model, newdata = test_dt)

pred_dt <- data.table(card_id = id_te, 
                      target = pred$data$response)
pred_dt <- pred_dt[match(sample_submission$card_id, id_te)]

if (! dir.exists('output') ) dir.create('output')
fwrite(x = pred_dt, file = 'output/sub_7.csv')
```

# 4. Discussion

1. This is my first-ever Kaggle competition. It was lots of fun! There are so much information hidden in large data sets. We are like adventurers looking for surprises.
2. Sometimes there are so many features involved, to select a good machine learning model is difficult, but even more so if we want to interpret the model.
