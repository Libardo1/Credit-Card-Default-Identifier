
############ Credit Default #################




# The dataset is taken from UCI's Machine Learning Repositiory at http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients. 
# The dataset includes demographics and payment information from credit card clients in Taiwan and I want to use this dataset to build a 
# predictive model to predict credit card default.

setwd('C:/Users/Dancy Pants/OneDrive/CLASS PROJECTS/Data Analytics')
rm(list=ls())


#### Load Packages & Data ####
library(klaR)
library(MASS)
library(caret)
library(randomForest)
library(doParallel)
library(pROC)
library(kernlab)
library(tidyverse)
source("class_functions.R")
load("creditdefault.RData")
cl <- makeCluster(3)
registerDoParallel(cl)

raw_data <- read_csv("default of credit card clients.csv") 
colnames(raw_data)[25] <- 'default'


#### Clean and Pre-process Data ####
# Check if there are any missing values:
sum(raw_data == ""| is.na(raw_data))

# Examine the variables:
#See the distribution across categorical variables:
table(raw_data$SEX)
table(raw_data$EDUCATION)
table(raw_data$MARRIAGE)
table(raw_data$default)/nrow(raw_data)
#Since the proportion of categories 0, 4,5,6 in education and categories 0,3 in marriage is low, I will group them to categories 0 in 
#each variable. This will help to decrease the number of features without unreasonable information loss.



whole <- raw_data %>%  select(-ID)  %>%
  mutate(EDUCATION = ifelse(EDUCATION > 3, 0, EDUCATION), MARRIAGE = ifelse(MARRIAGE == 3, 0, MARRIAGE)) %>% 
  mutate_each(funs(as.factor),c(SEX,EDUCATION,MARRIAGE)) %>% 
  mutate(default=factor(ifelse(default==1,"yes","no"), levels=c("yes","no"))) %>%  group_by(default) %>% 
  glimpse() 

### Partition to test set and train-cross-validation set
index <- sample(nrow(whole),.2*nrow(whole))
test_set <- whole[index,]
mydata <- whole[-index,]



#### Tuning for full model ####
# tuning control:
tune_ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)

# Tune for KNN
set.seed(1234)
train(default ~ .,
      data = mydata[sample(nrow(mydata),5000),],
      method = "knn",
      metric = "ROC",
      preProc = c("center","scale"),
      tuneLength = 5,
      trControl = tune_ctrl)   # Optimal k = 13


#Tune for Random Forest
set.seed(1234)
train(default ~ ., 
      data = mydata[sample(nrow(mydata),5000),],
      method = "rf",
      metric = "ROC",
      tuneLength = 5,
      #tuneGrid = c(3,6,9,12,15)
      trControl = tune_ctrl)    # Optimal mtry = 2  



#Tune for SVM Linear
set.seed(1234)
train(default ~ .,
      data = mydata[sample(nrow(mydata),5000),],
      method = "svmLinear2",
      preProc = c('center','scale'),
      metric = "ROC",
      tuneLength = 5,
      trControl = tune_ctrl)    # Optimal cost = 0.5 


#### Full model building -------

# evaluation control:
eva_ctrl <-  trainControl(method = 'repeatedcv',number = 10,repeats = 5,
                        savePredictions = T, summaryFunction = twoClassSummary, classProbs = T)



# *Full logistic regression: -----
set.seed(7777)
glm.full <- train(default ~ .,
                  data = mydata,
                  method = 'glm',
                  family = binomial,
                  metric = 'ROC',
                  trControl = eva_ctrl
                  )

# *Full KNN: -------
set.seed(7777)
knn.full <- train(default ~ .,
                  data = mydata,
                  method = 'knn',
                  metric = 'ROC',
                  preProc = c('center','scale'),
                  tuneGrid = data.frame(.k= 13),
                  trControl = eva_ctrl)

# *Full Random Forest: ------
set.seed(7777)
rf.full <- train(default ~ ., 
                data = mydata, 
                method = "rf",
                metric = "ROC",
                tuneGrid = data.frame(mtry=2),
                trControl = eva_ctrl)

# * Full Naive Bayes: -----
set.seed(7777)
nb.full <- train(default ~ ., 
                 data = mydata, 
                 method = "nb",
                 metric = "ROC",
                 tuneGrid = data.frame(fL=0,usekernel=TRUE, adjust=1),
                 trControl = eva_ctrl)

# * Full SVM Linear: ------
set.seed(7777)
svmLinear.full <- train(default ~ .,
                        data = mydata,
                        method = "svmLinear",
                        preProc = c('center','scale'),
                        metric = "ROC",
                        tuneGrid = data.frame(C = 0.5),
                        trControl = eva_ctrl) 



#### Feature Reduction #####
reduced <- step(glm(default ~ ., data = mydata, family = binomial),
                direction = 'both')
 


#### Tuning for Reduced Model Building ####

# Tune for KNN
set.seed(1234)
train(reduced$formula,
      data = mydata[sample(nrow(mydata),5000),],
      method = "knn",
      metric = "ROC",
      preProc = c("center","scale"),
      tuneLength = 5,
      trControl = tune_ctrl)   
# Optimal k = 13


#Tune for Random Forest
set.seed(1234)
train(reduced$formula,
      data = mydata[sample(nrow(mydata),5000),],
      method = "rf",
      metric = "ROC",
      tuneLength = 5,
      #tuneGrid = c(3,6,9,12,15)
      trControl = tune_ctrl)  #Optimal mtry = 2

set.seed(1234)
train(reduced$formula,
      data = mydata[sample(nrow(mydata),5000),],
      method = "svmLinear",
      preProc = c('center','scale'),
      metric = "ROC",
      tuneGrid = data.frame(C = 2^c(-2,-1,0,1,2)),
      trControl = tune_ctrl)    # Optimal C = 2


#### Reduced Model Building ####

# *Reduced logistic regression: -----
set.seed(7777)
glm.reduced <- train(reduced$formula,
                     data = mydata,
                  method = 'glm',
                  family = binomial,
                  metric = 'ROC',
                  trControl = eva_ctrl
)

# *Reduced KNN: -------
set.seed(7777)
knn.reduced <- train(reduced$formula,
                 data = mydata,
                  method = 'knn',
                  metric = 'ROC',
                  preProc = c('center','scale'),
                  tuneGrid = data.frame(.k= 13),
                  trControl = eva_ctrl)

# *Reduced Random Forest: ------
set.seed(7777)
rf.reduced <- train(reduced$formula,
                    data = mydata, 
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = data.frame(mtry=2),
                 trControl = eva_ctrl)

# * Reduced Naive Bayes: -----
set.seed(7777)
nb.reduced <- train(reduced$formula,
                 data = mydata, 
                 method = "nb",
                 metric = "ROC",
                 tuneGrid = data.frame(fL=0,usekernel=TRUE, adjust=1),
                 trControl = eva_ctrl)

# * Reduced SVM Linear: ------
set.seed(7777)
svmLinear.reduced <- train(reduced$formula,
                        data = mydata,
                        method = "svmLinear",
                        preProc = c('center','scale'),
                        metric = "ROC",
                        tuneGrid = data.frame(C = 2),
                        trControl = eva_ctrl) 



#### EVALUATE #######
summary(resamples(list(`Full KNN`=knn.full, 
                       `Full Logistic Regression`=glm.full,
                       `Full Random Forest` = rf.full,
                       `Full Naive Bayes`= nb.full,
                       `Full SVM Linear` = svmLinear.full,
                       `Reduced KNN` = knn.reduced,
                       `Reduced Logistic Regression` = glm.reduced,
                       `Reduced Random Forest` = rf.reduced,
                       `Reduced Naive Bayes` = nb.reduced,
                       `Reduced SVM Linear` = svmLinear.reduced)))
# Reduced Random Forest seem to have the highest performance in terms of AUC.
# Since the reduced models have have very similar or higher ROC compared to each's
# corresponding full model and since they are simpler, I assume they are superior
# to their corresponding full model. I will plot the ROC curves to compare them.
# ROC 
#                               Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
# Full KNN                    0.7029  0.7249 0.7320 0.7330  0.7415 0.7555    0
# Full Logistic Regression    0.7059  0.7192 0.7243 0.7255  0.7302 0.7523    0
# Full Random Forest          0.7395  0.7616 0.7700 0.7678  0.7736 0.7839    0
# Full Naive Bayes            0.7146  0.7339 0.7394 0.7395  0.7457 0.7621    0
# Full SVM Linear             0.6765  0.6922 0.6981 0.6993  0.7080 0.7271    0
# Reduced KNN                 0.7137  0.7278 0.7364 0.7365  0.7457 0.7605    0
# Reduced Logistic Regression 0.7063  0.7198 0.7251 0.7260  0.7309 0.7532    0
# Reduced Random Forest       0.7433  0.7624 0.7702 0.7683  0.7740 0.7849    0
# Reduced Naive Bayes         0.7150  0.7327 0.7390 0.7387  0.7447 0.7632    0
# Reduced SVM Linear          0.6660  0.6895 0.6988 0.6974  0.7047 0.7290    0


# Let's look at the plot
plot(roc(knn.reduced$pred$obs,knn.reduced$pred$yes),legacy.axes = T)
plot(roc(glm.reduced$pred$obs,glm.reduced$pred$yes), add = T, lty = 2)
plot(roc(nb.reduced$pred$obs,nb.reduced$pred$yes), add = T, lty = 3)
plot(roc(svmLinear.reduced$pred$obs,svmLinear.reduced$pred$yes), add = T, lty = 4)
plot(roc(rf.reduced$pred$obs,rf.reduced$pred$yes), add = T, col = "blue")
legend("bottomright",c("KNN","GLM","NB","SVM","RF"),lty = c(1,2,3,4,1),col = c("black","black","black","black","blue"))


# The ROC plots suggest that the reduced KNN (blue) model dominates at every cutoff and 
# proves itself to be the optimal model out of those constructed.

# Let's check the performance of the reduced random forest model on the training set:
prob <- predict(rf.reduced,mydata,type = 'prob')
roc(mydata$default,prob$yes)  # AUC on training dataset: 0.995

# Let's check the performance of the runner-up model, reduced Naive Bayes model, on the training set:
prob.nb<- predict(nb.reduced,mydata,type = 'prob')
roc(mydata$default,prob.nb$yes)     # AUC on training set for reduced Naive Bayes: 0.7528
# Little evidence of overfitting





#### Tuning for new model ####
# tuning control:
tune_ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)

# Tune for KNN
set.seed(1234)
train(default ~ .,
      data = mydata[sample(nrow(mydata),5000),c("default",newset)],
      method = "knn",
      metric = "ROC",
      preProc = c("center","scale"),
      tuneLength = 5,
      trControl = tune_ctrl)   # Optimal k = 13


#Tune for Random Forest
set.seed(1234)
train(default ~ ., 
      data = mydata[sample(nrow(mydata),5000),c("default",newset)],
      method = "rf",
      metric = "ROC",
      tuneLength = 5,
      trControl = tune_ctrl)    # Optimal mtry = 2  



#Tune for SVM Linear
set.seed(1234)
train(default ~ .,
      data = mydata[sample(nrow(mydata),5000),c("default",newset)],
      method = "svmLinear2",
      preProc = c('center','scale'),
      metric = "ROC",
      tuneLength = 5,
      trControl = tune_ctrl)    # Optimal cost = 0.25 


# *New logistic regression: -----
set.seed(7777)
glm.new <- train(default ~ .,
                  data = mydata[c("default",newset)],
                  method = 'glm',
                  family = binomial,
                  metric = 'ROC',
                  trControl = eva_ctrl
)

# *New KNN: -------
set.seed(7777)
knn.new <- train(default ~ .,
                  data = mydata[c("default",newset)],
                  method = 'knn',
                  metric = 'ROC',
                  preProc = c('center','scale'),
                  tuneGrid = data.frame(.k= 13),
                  trControl = eva_ctrl)

# *New Random Forest: ------
set.seed(7777)
rf.new <- train(default ~ ., 
                 data = mydata[c("default",newset)], 
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = data.frame(mtry=2),
                 trControl = eva_ctrl)

# * New Naive Bayes: -----
set.seed(7777)
nb.new <- train(default ~ ., 
                 data = mydata[c("default",newset)], 
                 method = "nb",
                 metric = "ROC",
                 tuneGrid = data.frame(fL=0,usekernel=TRUE, adjust=1),
                 trControl = eva_ctrl)

# * New SVM Linear: ------
set.seed(7777)
svmLinear.new <- train(default ~ .,
                        data = mydata[c("default",newset)],
                        method = "svmLinear",
                        preProc = c('center','scale'),
                        metric = "ROC",
                        tuneGrid = data.frame(C = 0.25),
                        trControl = eva_ctrl) 


