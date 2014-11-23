# Practical Machine Learning project
pikou  
##  INTRODUCTION:

  Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible
  to collect a large amount of data about personal activity relatively
  inexpensively. These type of devices are part of the quantified self movement 
  a group of enthusiasts who take measurements about themselves regularly to 
  improve their health, to find patterns in their behavior, or because they are
  tech geeks. One thing that people regularly do is quantify how much of a 
  particular activity they do, but they rarely quantify how well they do it. 
  In this project, the goal will be to use data from accelerometers on the belt,
  forearm, arm, and dumbell of 6 participants. They were asked to perform babell    
  lifts correctly and incorrectly in 5 different ways.

. More information is  available from the website here:
  http://groupware.les.inf.puc-rio.br/har 
  (see the section on the Weight Lifting Exercise Dataset).

. The goal of the project is to predict the manner in which the 6 partcipants
  did the exercise. This is the "classe" variable in the training set,and  omitted in the test set for grading purpose by Coursera Machine Learning class of Data Science Specialisation.
  
. 5 types of activities to detect : 1 correct method and 4 common mistakes:
    
  Exactly according to the specification (Class A)
  Ehrowing the elbows to the front (Class B)
  Lifting the dumbbell only halfway (Class C)
  Lowering the dumbbell only halfway (Class D)
  Throwing the hips to the front (Class E)
  
.  The training data for this project are available here: 
  
  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

   The test data are available here: 
  
  https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

##  DATA EXPLORATION:
       
  1. Loading and reading  data :
  

```r
training <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!"))
testing <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!"))
```
  
  The training data is a set of 19622 observations and 160 varaibles, the last 
 variable is the outcome labled 'classe',which is a factor with 5 levels(A,B,C,D
 ,E) defining the participant activity.
  The test data is a set of 20 observations and 160 variables, the oucome variable is replaced with the variable problem.id for the project grading purpose.
 

```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

```r
str(training$classe)
```

```
##  Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
str(testing[160])
```

```
## 'data.frame':	20 obs. of  1 variable:
##  $ problem_id: int  1 2 3 4 5 6 7 8 9 10 ...
```

```r
summary(training$classe)  
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

2. Training and testing data sets cleaning and features reduction:
    Both data sets contains considerable number of missing values.
    

```r
    sum(is.na(training))
```

```
## [1] 1925102
```

```r
    sum(is.na(testing))
```

```
## [1] 2000
```

 Removing  NAs


```r
  TrNAsCol <- which(colSums(is.na(training)) >0)
  training1 <- training[,-TrNAsCol]
  TsNAsCol <- which(colSums(is.na(testing)) >0)
  testing1 <- testing[,-TsNAsCol]
```
  
 Removing the first 7 varaibles (row numbers,time stamps, windows numbers) 
  irrelevants for this projects.
  

```r
  dim(training1)
```

```
## [1] 19622    60
```

```r
  dim(testing1)
```

```
## [1] 20 60
```

```r
  training2 <- training1[,8:60]
  testing2  <-  testing1[,8:60]
  
  dim(training2)
```

```
## [1] 19622    53
```

```r
  dim(testing2)
```

```
## [1] 20 53
```

3. Predictors nearZeroVar and correlation :No variables have a close to zero 
  variance ,some are correlated and will be removed.


```r
  require(caret)
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
  Nsv <- nearZeroVar(training2,saveMetrics=T)
    
  cor <- findCorrelation(cor(training2[,-53]))
  training2 <- training2[,-cor]
  testing2 <-  testing2[,-cor] 
  cor
```

```
## [1] 10  1  9  8 19 46 31
```

```r
  dim(training2)
```

```
## [1] 19622    46
```

```r
  dim(testing2)
```

```
## [1] 20 46
```

##  TRAINING DATASET AND CROSS-VALIDATION AND MODEL SELECTION : 
    
    The classification variable in the test set in  file pml-testing.csv ,
  is replaced by problem.id for grading purposes, to validate the model we need
  to partition the cleaned training set (my training2) in a training and test
  sets ( train and test),so we can train the model on train set and test the
  accuracy on the test set. A partition 60/40 is choosen (i am tryiny to reduce
  procesing time ,i missed two submissions because of that). 
  More exploration might be needed to further process the data for issues with
  calibration ,may be for another time, i will use the Random Forest algorithm
  for its flexibility and accuracy and also its popularity,and also to be able
  to compare my results to those of the authors of the original paper,who used
  Random Forest.
 

```r
   set.seed(100)
  
  inTrain <- createDataPartition(training2$classe, p= 0.60 ,list=F)
  train   <- training2[inTrain,]
  test    <- training2[-inTrain,]
```

 Training control setup :


```r
  require(randomForest)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
  ctrl <- trainControl( method = "repeatedcv", repeats=5 ,number=10 )
#  rfFit <- train(classe ~ . , data = train ,method= "rf",trControl=ctrl)
  print(rfFit)
```

```
## Random Forest 
## 
## 11776 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## 
## Summary of sample sizes: 10598, 10599, 10599, 10599, 10598, 10599, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1         1      0.003        0.004   
##   23    1         1      0.003        0.004   
##   45    1         1      0.004        0.006   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 23.
```

 Confusion matrix and expected out of sample error:


```r
  rfTrainPred <- predict(rfFit, newdata= train)
  rfTestPred <- predict(rfFit, newdata= test)
  rfTrainConfusion <- confusionMatrix(train$classe,rfTrainPred )
  rfTestConfusion <- confusionMatrix(test$classe,rfTestPred )
```

          
  The model accuracy on the training set (train) reached 99.7%  
  

```r
rfTrainConfusion  
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3345    1    1    0    1
##          B   12 2264    3    0    0
##          C    0    5 2046    3    0
##          D    0    0    9 1920    1
##          E    0    1    1    1 2162
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.998)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.997    0.993    0.998    0.999
## Specificity             1.000    0.998    0.999    0.999    1.000
## Pos Pred Value          0.999    0.993    0.996    0.995    0.999
## Neg Pred Value          0.999    0.999    0.999    1.000    1.000
## Prevalence              0.285    0.193    0.175    0.163    0.184
## Detection Rate          0.284    0.192    0.174    0.163    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.998    0.996    0.998    0.999
```

  The model accuracy on the testing set (test) reached 99.7% 

```r
  rfTestConfusion
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    3 1511    4    0    0
##          C    0    1 1363    4    0
##          D    0    0    4 1282    0
##          E    0    1    1    3 1437
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.996, 0.998)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.998    0.993    0.995    1.000
## Specificity             1.000    0.999    0.999    0.999    0.999
## Pos Pred Value          1.000    0.995    0.996    0.997    0.997
## Neg Pred Value          0.999    1.000    0.999    0.999    1.000
## Prevalence              0.285    0.193    0.175    0.164    0.183
## Detection Rate          0.284    0.193    0.174    0.163    0.183
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.998    0.996    0.997    1.000
```
##  Expected out of sample error:
  
  The model results shows that the accuracy on testing set (test) can reach 
  99.7%,so  we can predict an out of sample error rate on new data to be around    
  0.3%
  
#  Submission code for the class project assignment(test data classification):
  

```r
  testPred1 <- as.character(predict(rfFit, newdata= testing2))  
  testPred1
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers <- testPred1 
pml_write_files(answers)
```

