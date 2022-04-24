---
title: "Practical Machine Learning - Final Project"
author: "Mohamad Tayara"
date: "April 2022"
output: 
        html_document:
         keep_md: yes
---



# Overview
 

In this project, th data from accelerometers on the belt, forearm, arm, and dumbell of six participants wil be used  to predict the type exercise which is represented by  the “classe” variable in the training set. 3  models will be trained: **Decision Tree**, **Random Forest**, **Gradient Boosted Trees** using k-folds cross validation on the training set.Based on  **accuracy** and **out of sample error rate**, the best model will be chosen to predict 20 cases using the test csv set.


# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: (http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.


# Loading Data and Libraries
Loading all the libraries and the data

```r
library(lattice)
library(ggplot2)
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.1.3
```

```r
library(kernlab)
```

```
## Warning: package 'kernlab' was built under R version 4.1.3
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 4.1.3
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 4.1.3
```

```r
set.seed(4123)
```


```r
traincsv <- read.csv("./data/pml-training.csv")
testcsv <- read.csv("./data/pml-testing.csv")

dim(traincsv)
```

```
## [1] 19622   160
```

```r
dim(testcsv)
```

```
## [1]  20 160
```

There are 160 variables and 19622 observations in the training set, while 20 for the test set.


# Data Clenaning

Removing unnecessary variables. Starting with N/A variables.

```r
traincsv <- traincsv[,colMeans(is.na(traincsv)) < .8] #removing mostly na columns
traincsv <- traincsv[,-c(1:7)] #removing metadata which is irrelevant to the outcome
```

Removing near zero variance variables.

```r
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[,-nvz]
dim(traincsv)
```

```
## [1] 19622    53
```

Now the training set into a **validation** and **training** set. The testing set "testcsv" will be left alone, and used for the final quiz test cases. 

```r
inTrain <- createDataPartition(y=traincsv$classe, p=0.75, list=F)
train <- traincsv[inTrain,]
valid <- traincsv[-inTrain,]
```


# Creating and Testing the Models
Here we will test a few popular models including: **Decision Trees**, **Random Forest** and **Gradient Boosted Trees**

Set up control for training to use 3-fold cross validation. 

```r
control <- trainControl(method="cv", number=3, verboseIter=F)
```

## Decision Tree

**Model:** 


```r
mod_trees <- train(classe~., data=train, method="rpart", trControl = control, tuneLength = 5)
fancyRpartPlot(mod_trees$finalModel)
```

![](report_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

**Prediction:**


```r
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1258  406  392  338  133
##          B   26  281   22   10   96
##          C   74  111  374  124  105
##          D   33  151   67  332  145
##          E    4    0    0    0  422
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5438          
##                  95% CI : (0.5298, 0.5579)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.406           
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9018   0.2961  0.43743   0.4129  0.46837
## Specificity            0.6384   0.9611  0.89775   0.9034  0.99900
## Pos Pred Value         0.4978   0.6460  0.47462   0.4560  0.99061
## Neg Pred Value         0.9424   0.8505  0.88314   0.8870  0.89303
## Prevalence             0.2845   0.1935  0.17435   0.1639  0.18373
## Detection Rate         0.2565   0.0573  0.07626   0.0677  0.08605
## Detection Prevalence   0.5153   0.0887  0.16069   0.1485  0.08687
## Balanced Accuracy      0.7701   0.6286  0.66759   0.6582  0.73368
```

## Random Forest


```r
mod_rf <- train(classe~., data=train, method="rf", trControl = control, tuneLength = 5)

pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    3    0    0    0
##          B    0  942    5    0    0
##          C    0    4  848    6    2
##          D    0    0    2  798    0
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9932, 0.9972)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9943          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9918   0.9925   0.9978
## Specificity            0.9991   0.9987   0.9970   0.9995   1.0000
## Pos Pred Value         0.9979   0.9947   0.9860   0.9975   1.0000
## Neg Pred Value         1.0000   0.9982   0.9983   0.9985   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1729   0.1627   0.1833
## Detection Prevalence   0.2851   0.1931   0.1754   0.1631   0.1833
## Balanced Accuracy      0.9996   0.9957   0.9944   0.9960   0.9989
```

## Gradient Boosted Trees


```r
mod_gbm <- train(classe~., data=train, method="gbm", trControl = control, tuneLength = 5, verbose = F)

pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    5    0    0    0
##          B    2  940   13    0    1
##          C    0    4  836    6    2
##          D    0    0    6  795    2
##          E    0    0    0    3  896
## 
## Overall Statistics
##                                          
##                Accuracy : 0.991          
##                  95% CI : (0.988, 0.9935)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9887         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9905   0.9778   0.9888   0.9945
## Specificity            0.9986   0.9960   0.9970   0.9980   0.9993
## Pos Pred Value         0.9964   0.9833   0.9858   0.9900   0.9967
## Neg Pred Value         0.9994   0.9977   0.9953   0.9978   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1917   0.1705   0.1621   0.1827
## Detection Prevalence   0.2851   0.1949   0.1729   0.1637   0.1833
## Balanced Accuracy      0.9986   0.9932   0.9874   0.9934   0.9969
```



## Results (Accuracy & Out of Sample Error)


```
##      accuracy oos_error
## Tree    0.544     0.456
## RF      0.996     0.004
## GBM     0.991     0.009
```

**The best model is the Random Forest model, with 0.9955139 accuracy and 0.0044861 out of sample error rate.. ** 


# Predictions on Test Set

Running our test set to predict the classe outcome for 20 cases with the **Random Forest** model.

```r
pred <- predict(mod_rf, testcsv)
print(pred)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


# Appendix


correlation matrix of variables in training set

```r
corrPlot <- cor(train[, -length(names(train))])
corrplot(corrPlot, method="color")
```

![](report_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

Plotting the models

```r
plot(mod_trees)
```

![](report_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

```r
plot(mod_rf)
```

![](report_files/figure-html/unnamed-chunk-13-2.png)<!-- -->

```r
plot(mod_gbm)
```

![](report_files/figure-html/unnamed-chunk-13-3.png)<!-- -->
