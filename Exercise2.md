# Predicting Exercise Class
Friday, December 25, 2015  

## Executive Summary

Using the HAR study available from the website http://groupware.les.inf.puc-rio.br/har, six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

This project is to predict classes of the testing data based on model built from the training data.


### Load the libraries


```r
library(caret)
library(randomForest)
```

### Exploring attributes and data


```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##   [list output truncated]
```


Looking at the training set, some columns contain no value at all, ie, NA, #DIV/0!, or blank. In order to remove the columns containing no value and to turn off automatic factoring, set na.strings designation of a character vector of strings which are to be interpreted as NA values



```r
trainingSet <- read.csv(file="pml-training.csv", header=TRUE, stringsAsFactors = FALSE, na.strings=c('NA', '#DIV/0!', ''))
testingSet <- read.csv(file="pml-testing.csv", header=TRUE, stringsAsFactors = FALSE, na.strings=c('NA', '#DIV/0!', ''))
```


### Cleaning data

Pick columns containing less than 1000 NA's, and column names containing belt, forearm, arm and dumbell.


```r
NACounts <- colSums(is.na(trainingSet))       # getting NA counts for all columns
trindex <- (NACounts < 1000) & grepl("belt|forearm|arm|dumbell",names(trainingSet))
trainColName <- names(trainingSet[trindex])
cleanTraining <- trainingSet[trindex]
cleanTraining$classe <- as.factor(trainingSet$classe)
cleanTesting <- testingSet[trindex]
```


### Partitioning train data into training partition (75%) to build model and validation partition (25%) for cross validation


```r
set.seed(2015)
trainPartition = createDataPartition(cleanTraining$classe, p = .75, list=FALSE)
training = cleanTraining[trainPartition,]
validation = cleanTraining[-trainPartition,]
```


### Model building by applying random forest method to the training partition. 

Random forest method was chosen because it is known to produce relatively accurate result, compared to other methods. The parameters and options were chosen after trial and error.


```r
model <- train(classe ~ ., data = training, method = "rf", prox = TRUE, trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE))
```


### Cross validation error


```r
trainingPrediction <- predict(model, training)
confusionMatrix(trainingPrediction, training$classe)
```

````
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 5577    2    0    0    0
         B    1 3794    3    0    0
         C    0    1 3417    3    0
         D    0    0    2 3213    3
         E    2    0    0    0 3604

Overall Statistics
                                          
               Accuracy : 0.9991          
                 95% CI : (0.9986, 0.9995)
    No Information Rate : 0.2844          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9989          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9995   0.9992   0.9985   0.9991   0.9992
Specificity            0.9999   0.9997   0.9998   0.9997   0.9999
Pos Pred Value         0.9996   0.9989   0.9988   0.9984   0.9994
Neg Pred Value         0.9998   0.9998   0.9997   0.9998   0.9998
Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2842   0.1934   0.1741   0.1637   0.1837
Detection Prevalence   0.2843   0.1936   0.1743   0.1640   0.1838
Balanced Accuracy      0.9997   0.9995   0.9991   0.9994   0.9995
````

The accuracy is 0.9991, meaning error rate of 0.0009, i.e. 0.09%


### Out of sample error


```r
cvPrediction <- predict(model, validation)
confusionMatrix(cvPrediction, validation$classe)
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1392    2    0    0    0
         B    1  946    3    0    0
         C    0    1  850    3    0
         D    0    0    2  801    3
         E    2    0    0    0  898

Overall Statistics
                                         
               Accuracy : 0.9965         
                 95% CI : (0.9945, 0.998)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9956         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9978   0.9968   0.9942   0.9963   0.9967
Specificity            0.9994   0.9990   0.9990   0.9988   0.9995
Pos Pred Value         0.9986   0.9958   0.9953   0.9938   0.9978
Neg Pred Value         0.9991   0.9992   0.9988   0.9993   0.9993
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2838   0.1929   0.1733   0.1633   0.1831
Detection Prevalence   0.2843   0.1937   0.1741   0.1644   0.1835
Balanced Accuracy      0.9986   0.9979   0.9966   0.9975   0.9981
```

The accuracy is 0.9965, meaning error rate of 0.0035, i.e. 0.35%


### Prediction of the testing data


```r
testingPrediction <- predict(model, cleanTesting)
```


### Writing prediction to files using routines given in the assignment


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(testingPrediction)
```

## Conclusion

With random forest machine learning algorithm, the out of sample error was less than 1% indicating the goodness of the algorithm. The submission of the prediction of testing data results in 20/20, indicating the accuracy of the model.
