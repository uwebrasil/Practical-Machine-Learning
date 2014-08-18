Project File for Practical Machine Learning <br />at Coursera Johns Hopkins
========================================================
<h3>Summary</h3>

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
<br />
<ul><li>exactly according to the specification (Class A)
</li><li>throwing the elbows to the front (Class B)
</li><li>lifting the dumbbell only halfway (Class C)
</li><li> lowering the dumbbell only halfway (Class D)
</li><li>throwing the hips to the front (Class E)</li></ul>
Read more: http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz3AgifC0Uo
<br />
<p>
Due to extremely long running time with caret's train function, parallel processing
and randomForest's randomForest method are chosen(~1 min).<br /> 
</p>
<h3>Libraries, Options and Data Load</h3>
This script assumes the traingset and the testset in the same directory.
<br />
<pre>

```r
library(ggplot2);
library(caret)
```

```
## Loading required package: lattice
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(Hmisc)
```

```
## Loading required package: grid
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: Formula
## 
## Attaching package: 'Hmisc'
## 
## The following object is masked from 'package:randomForest':
## 
##     combine
## 
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
library(foreach)
library(doParallel)
```

```
## Loading required package: iterators
## Loading required package: parallel
```

```r
options(warn=1)
options(stringsAsFactors = FALSE)
testing <-read.csv("pml-testing.csv")
training <- read.csv("pml-training.csv")
```
</pre>
<ul>
 <li>Trainingset - pml-training.csv</li>
 <pre>

```
## [1] 19622   160
```
</pre>
 <li>Testset - pml-testing.csv</li>
<pre>

```
## [1]  20 160
```
</pre>
</ul> 
<h3>Cleaning Data and Select Features</h3>
A brief look at the data revealed many columns with mainly NA's and finally lead to select only the main acceleration columns.

```r
set.seed(123)
training <- training[,union(grep("^accel_", colnames(training)),grep("classe",colnames(training)) )] 
testing <- testing[,union(grep("^accel_", colnames(testing)),grep("classe",colnames(testing)) )] 
names(training)
```

```
##  [1] "accel_belt_x"     "accel_belt_y"     "accel_belt_z"    
##  [4] "accel_arm_x"      "accel_arm_y"      "accel_arm_z"     
##  [7] "accel_dumbbell_x" "accel_dumbbell_y" "accel_dumbbell_z"
## [10] "accel_forearm_x"  "accel_forearm_y"  "accel_forearm_z" 
## [13] "classe"
```
<h3>Data Partition</h3>
Split trainingset into trainingsample(80%) and testsample(20%)

```r
partition <- createDataPartition(y = training$classe, p = 0.8, list = FALSE)
sample1 <- training[partition, ]
psample1 <-  training[-partition, ]
```
<h3>Converting features</h3>
Classification-feature (classe) to factor, all others as numeric

```r
sample1$classe <-as.factor(sample1$classe)
sample1[, 1:6] <- sapply(sample1[, 1:6], as.numeric)
psample1$classe <-as.factor(psample1$classe)
psample1[, 1:6] <- sapply(psample1[, 1:6], as.numeric)
```
<h3>Model Build</h3>
Significant speedup using Parallel Processing.

```r
registerDoParallel()
rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
        randomForest(x=sample1[,1:12], y=sample1$classe, ntree=ntree)
}
```
<h3>Prediction and Errors </h3>
ConfusionMatrix, Out of Sample Error

```r
pred0 <- predict(rf,psample1)
confusionMatrix(pred0,psample1$classe)
```

```
## 
## Attaching package: 'e1071'
## 
## The following object is masked from 'package:Hmisc':
## 
##     impute
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1070   13   12   19    1
##          B    6  714   18    5   10
##          C   15   24  647   25    6
##          D   25    2    6  590    6
##          E    0    6    1    4  698
## 
## Overall Statistics
##                                         
##                Accuracy : 0.948         
##                  95% CI : (0.941, 0.955)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.934         
##  Mcnemar's Test P-Value : 0.00908       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.959    0.941    0.946    0.918    0.968
## Specificity             0.984    0.988    0.978    0.988    0.997
## Pos Pred Value          0.960    0.948    0.902    0.938    0.984
## Neg Pred Value          0.984    0.986    0.988    0.984    0.993
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.273    0.182    0.165    0.150    0.178
## Detection Prevalence    0.284    0.192    0.183    0.160    0.181
## Balanced Accuracy       0.971    0.964    0.962    0.953    0.982
```
<h3>Submit Answers</h3>

```r
predict(rf, testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  C  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
<h3>Conclusion</h3>
Because of an accuracy of around 96% it was expected nearly all of the answers to be correct.<br />
However there was an error found for answer 3. This can be corrected including more features.
