

############### data set information ######################
#The data is related with direct marketing campaigns of a Portuguese banking institution.
#The marketing campaigns were based on phone calls.
#Often, more than one contact to the same client was required, in order to access
#if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

## dependent variable: y - has the client subscribed a term deposit? (binary: 'yes','no')

##we will do following steps
#1.import the data set
#2.remove the unnecessary column
#3.renaming the data set column
#4.check the data types,dim,summary
#5.density plot before preprocessing
#6.checking the NULL values(replace with continuous:mean,median or discrite:mode)
#7.check the outliers values(replace with continuous:mean,median or discrite:mode)
#8.overall distribution for all features(histogramplots)
#9.label encoding features(categorecal into numerical)
#10.check the correlation matrix(ggcorrplot)
#11.scale the data
#12.split the data train and test
#13.apply the machine learning models
#14.check the accuracy of each model
#15.deploy the high accuracy algorithm 

#to remove the previous outputs and file in r studio
rm(list = ls()) 
#read the csv file
#bank=read.csv(file.choose())
bank=read.csv("customer_campaign.csv")
View(head(bank,10))

#renaming data set column
colnames(bank)[5]="default_credit"
colnames(bank)[7]="housing_loan"
colnames(bank)[8]="personal_loan"
colnames(bank)[9]="contact_type"
colnames(bank)[13]="current_compaign_contact_count"
colnames(bank)[14]="days_passed"
colnames(bank)[15]="previous_campaign_contact_count"
colnames(bank)[16]="previous_campaign_outcome"

#checking the NULL values
#data.frame(colSums(is.na(bankData)))

if(length(which(is.na(bank)==T))){
  print("missing values are found")
}else{
  print("no missing values are found")
}

# check the outliers
unique(boxplot(bank$age)$out)
unique(boxplot(bank$balance)$out)
unique(boxplot(bank$duration)$out)
boxplot(bank$age,bank$balance,bank$duration)

#data has approx 45000 data point with 17 features.
dim(bank)

#datatypes of column
data.frame(sapply(bank,class))

#summary of column
summary(bank)

#count analysis of categorical data
table(bank$job)
table(bank$marital)
table(bank$education)
table(bank$contact_type)
table(bank$month)
table(bank$previous_campaign_outcome)

#librarys
library(ggplot2)
library(rpart)
library(rpart.plot)
library(carData)
library(car)
library(class)
library(class)
library(lattice)
library(caTools)
library(caret)

#density plot before preprocessing
library(ggplot2)
ggplot(bank,aes(x=age))+geom_density(color="red",fill="lightgreen")
ggplot(bank,aes(x=duration))+geom_density(color="red",fill="lightgreen")

# outliers replace with mean

for (colName in c('age', 'duration','balance')) {
  high = quantile(bank[,colName])[4] + 1.5*IQR(bank[,colName])
  low = quantile(bank[,colName])[2] - 1.5*IQR(bank[,colName])
  for (index in c(1:nrow(bank))) {
    bank[,colName][index] = ifelse(bank[,colName][index] > high, high, bank[,colName][index])
    bank[,colName][index] = ifelse(bank[,colName][index] < low, low, bank[,colName][index])
  }
}


#after boxplot
boxplot(bank$age,bank$balance,bank$duration)



######Overall Distribution For All Features
library(patchwork)
library(tidyverse)

bank %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot() +
  geom_histogram(mapping = aes(x=value,fill=key), color="black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal() +
  theme(legend.position = 'none')

 
###Correlation Matrix

library(corrplot)
library(ggcorrplot)
Name=names(which(sapply(bank, is.numeric)))
corr=cor(bank[,Name], use = 'pairwise.complete.obs')
ggcorrplot(corr, lab = TRUE)

###########Response Variable
ggplot(bank, aes(y, fill = y)) +
  geom_bar() +
  theme(legend.position = 'none')
table(bank$y)

##density plot after preprocessing
library(ggplot2)
ggplot(bank,aes(x=age))+geom_density(color='red',fill='lightgreen')
ggplot(bank,aes(x=duration))+geom_density(color='red',fill='lightgreen')


#ggplot for input (education) and output(y)
ggplot(bank, aes(education)) + 
  geom_bar(aes(fill = y), position = "fill") +
  coord_flip()
      #OBSERVED:From above plot we can observed tertiary education peoples are more subscribed 
           #a term deposit(yes) compare to other educations.
#ggplot for input(job) and output(y)
ggplot(bank, aes(job)) + 
  geom_bar(aes(fill = y), position = "fill") +
  coord_flip()
        #OBSERVED:from above plot we can observed students are more subscribed a term deposit(yes)
                  #compare to other jobs.
#label encoding features

bank$default_credit =as.numeric(factor(bank$default_credit)) -1
bank$housing_loan =as.numeric(factor(bank$housing_loan)) -1
bank$personal_loan = as.numeric(factor(bank$personal_loan)) -1 
bank$y =as.numeric(factor(bank$y)) -1

library(fastDummies)
dummy=bank[c(2,3,4,9,11,16)]
bank1=fastDummies::dummy_cols(dummy,remove_first_dummy = FALSE)
bank1=bank1[c(-1,-2,-3,-4,-5,-6)]
bankData=cbind(bank,bank1)
bankData=bankData[,c(-2,-3,-4,-9,-11,-16)]
View(head(bankData))
dim(bankData)
colnames(bankData)[13]="job_bluecollar"
colnames(bankData)[18]="job_selfempoyed"


#normalization
library(caret)
preproc=preProcess(bankData[,c(1,3,7)], method=c("range"))

bankData=predict(preproc,bankData)

#split the data train and test
library(caTools)

bankData=bankData[sample(nrow(bankData)),]

split=sample.split(bankData$y,SplitRatio = 0.75)

training_set2<- subset(bankData, split == TRUE)

test_set2<- subset(bankData, split == FALSE)


##################### MACHINE LEARNING ALGORITMS #########################

############## 1.LOGISTIC REGRESSION ###########

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
library(caTools)
training=as.data.frame(bankData)
model=glm(y~.,data=training,family = "binomial")
pred=predict(model,training,type="response")
options(pre=-1)
# Confusion matrix and considering the threshold value as 0.5 
confusion<-table(pred>0.5,training$y)
confusion

# Model Accuracy 
Accuracy_glm<-sum(diag(confusion)/sum(confusion))
Accuracy_glm

############## 2.DECISION TREES (CART) ################
library(rpart)

fit=rpart(y~.,data=training_set2,method='class')
predicted=predict(fit,test_set2,type='class')
pred=table(test_set2$y,predicted)

confusionMatrix(pred)

accuracy_cart=sum(diag(pred))/sum(pred)
accuracy_cart

################### 3.NAIVE BAYES #################

#Naive Bayes
library(e1071)

training_set2$y = as.factor(training_set2$y)
test_set2$y =as.factor(test_set2$y)
bankData$y = as.factor(bankData$y)

model = naiveBayes(y~.,training_set2,na.action = na.pass)

predicted_y = predict(model, test_set2)

table <- table(test_set2$y,predicted_y)
table
library(caret)
confusionMatrix(table)

accuracy_navai=sum(diag(table))/sum(table)
accuracy_navai
################## 4.K-NEAREST NEIGHBORS (KNN) ##########
library(class)

knn_model <- knn(training_set2,test_set2,training_set2$y, k=3)
tab4 <- table(test_set2$y, knn_model)
tab4
knn_model2 <- knn(training_set2,test_set2,training_set2$y, k=10)
tab <- table(test_set2$y, knn_model)
tab

confusionMatrix(tab4)
accuracy_knn=sum(diag(tab4))/sum(tab4)
accuracy_knn

#################### 5.SUPPORT VECTOR MACHINE (SVM) ###############

library(e1071)

train.svm=svm(y~.,training_set2,kernel="polynomial",cost=0.01,scale=TRUE)

#  This is where we determine the model accuracy.  predict() applies the model to the test data.
test.svm=predict(train.svm,test_set2)

#table() shows the contingency table for the outcome of the prediction.  Each cell in a contingency table that counts
pred=table(test_set2$y,test.svm)
#confusion matrix
confusionMatrix(pred)
#model accuracy
accuracy_svm=sum(diag(pred)/sum(pred))
accuracy_svm

#################### 6.RANDOM FOREST ################
# Building a random forest model on training data 
library(randomForest)
model=randomForest(y~.,data=training_set2)

predict <- predict(model,test_set2)

pred=table(test_set2$y,predict)
#confusion matrix
confusionMatrix(pred)
#model accuracy
accuracy_raf=sum(diag(pred)/sum(pred))
accuracy_raf

######## MACHINE LEARNING MODELS ACCURACY ########

Accuracy_glm
accuracy_cart
accuracy_navai
accuracy_knn
accuracy_svm
accuracy_raf

############### OBERVATIONS #################

#### 1.ggplots for input and output variables
 #JOB:"student" are more subscribed(25%) compare to other jobs
 #EDUCATION:"tertiary" are more subscribed(16%) compare to other education

#### 2.machine learning models 
    #on appling knn algorithm the accuracy is 92.30% hence by comparing with 
     #other algorithms knn is best algorithm for this data set. 
        #deploy final model of knn model.

