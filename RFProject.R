setwd("F:\\Complete Machine Learning & Data Science with R-2019\\Codes\\Resources\\Section 12")

library(xlsx)
library(plyr)
library(dplyr)
adult<-read.delim("adult-data.txt", header=FALSE, sep=",")
View(adult)
adult<-read.delim("adult-data.txt", header=FALSE, sep=",", na.strings = c(" ?"," ","NA"))
str(adult)

sum(is.na(adult))

colnames(adult)<-c("age",
                   "workclass",
                   "fnlwgt",
                   "education",
                   "education.num",
                   "marital.status",
                   "occupation",
                    "relationship",
                   "race",
                   "sex",
                   "capital.gain",
                   "capital.loss",
                   "hours.per.week",
                   "native.country",
                   "Sal.gthan50")

str(adult)

library(Amelia)
missmap(adult)

library(mice)
md.pattern(adult, plot=TRUE)

colnames(adult)[colSums(is.na(adult))>0]

#imputing missing values
init=mice(adult, maxit=0)
meth=init$method
predM=init$predictorMatrix
meth[c("age",
         "fnlwgt",
         "education",
         "education.num",
         "marital.status",
         "relationship",
         "race",
         "sex",
         "capital.gain",
         "capital.loss",
         "hours.per.week",
         "Sal.gthan50")]=""

meth[c("workclass","occupation","native.country")]="sample"

set.seed(103)
imputed = mice(adult, method=meth, predictorMatrix = predM, m=5)

imputed<-complete(imputed)
View(imputed)

adult.new<-imputed

sum(is.na(adult.new))
colnames(adult.new)[colSums(is.na(adult.new))>0]
str(adult.new)

View(adult.new)

str(adult.new)
adult.new$Sal.gthan50<-mapvalues(adult.new$Sal.gthan50, from=c(" >50K", " <=50K"), to=c(1,0))

adult.new$random<-runif(nrow(adult.new), 0, 1)
trainingData<-adult.new[which(adult.new$random<=0.8),]
testData<-adult.new[which(adult.new$random>0.8),]
View(trainingData)
View(testData)

trainingData$random<-NULL
testData$random<-NULL
adult.new$random<-NULL
#randomForest

library(randomForest)

bestmtry<-tuneRF(trainingData, trainingData$Sal.gthan50, stepFactor = 1.2, improve = 0.01, trace=T, plot = T)
bestmtry

adult.forest<-randomForest(Sal.gthan50~., data=trainingData)
adult.forest
# (44+4558)/(19763+44+4558+1717) = 0.176 i.e ~83% accuracy

varImpPlot(adult.forest) #variable importance

#retuning the model
adult.forest.retune<-randomForest(Sal.gthan50~capital.gain+relationship+age+fnlwgt+marital.status+occupation, data=trainingData)
adult.forest.retune 
# (1411+2509)/(18396+1411+2509+3766) = 0.15 i.e 85% accuracy
varImpPlot(adult.forest.retune)

#predict
pred.adult<-predict(adult.forest.retune, newdata = testData, type = "class")
pred.adult

library(caret)

confusionMatrix(table(pred.adult, testData$Sal.gthan50))
# (588+343)/(4570+588+343+978) = 0.14 i.e ~86% accuracy which is significantly good

library(ROCR)
library(ggplot2)
ROCRpred<- prediction(as.numeric(pred.adult), as.numeric(testData$Sal.gthan50))
ROCRperf<- performance(ROCRpred, 'tpr', 'fpr')
plot(ROCRperf, col=2, abline(a=0, b=1, lwd=2, lty=2, col="gray"))

auc <- performance(ROCRpred,"auc")

auc <- unlist(slot(auc, "y.values"))

minauc<-min(round(auc, digits = 2))
maxauc<-max(round(auc, digits = 2))
minauct <- paste(c("min(AUC) = "),minauc,sep="")
maxauct <- paste(c("max(AUC) = "),maxauc,sep="")

#auc is 0.78 which is significantly good