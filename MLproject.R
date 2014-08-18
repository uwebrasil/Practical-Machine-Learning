library(ggplot2);
library(caret)
library(randomForest)
library(Hmisc)
library(foreach)
library(doParallel)

options(warn=1)
options(stringsAsFactors = FALSE)

testing <-read.csv("pml-testing.csv")
training <- read.csv("pml-training.csv")

set.seed(123)
training <- training[,union(grep("^accel_", colnames(training)),grep("classe",colnames(training)) )] 
testing <- testing[,union(grep("^accel_", colnames(testing)),grep("problem_id",colnames(testing)) )] 

partition <- createDataPartition(y = training$classe, p = 0.8, list = FALSE)
sample1 <- training[partition, ]
psample1 <-  training[-partition, ]

sample1$classe <-as.factor(sample1$classe)
sample1[, 1:6] <- sapply(sample1[, 1:6], as.numeric)
psample1$classe <-as.factor(psample1$classe)
psample1[, 1:6] <- sapply(psample1[, 1:6], as.numeric)

registerDoParallel()

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
        randomForest(x=sample1[,1:12], y=sample1$classe, ntree=ntree)
}

pred0 <- predict(rf,psample1)
cm0<-confusionMatrix(pred0,psample1$classe)
answers <- predict(rf, testing)




