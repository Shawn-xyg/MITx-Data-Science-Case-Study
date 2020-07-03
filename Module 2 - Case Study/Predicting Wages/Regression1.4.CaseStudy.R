# R code for Regression 1.4: Prediction Analysis

# Remove all vars
rm(list=ls())

# Set directory(this should be set to user's own directory)
setwd("E:/MIT Data Science and Big Data Analytics/Module 2/Case Study/Predicting Wages")

# Load data
load(file="pay.discrimination.Rdata")

# See variables in the dataset
class(data)
str(data)

# Dimensions of the dataset
dim(data)

# Table of means of each var
stats <- as.matrix(apply(data, 2, mean))

# Load xtable package to create latex tables
library(xtable)

# Assign colname to table stats
colnames(stats) = c("average")
xtable(stats)

####################  Linear and Quadratic specifications ##############################

# Wage linear regression
fmla1     <-  wage ~ female + sc+ cg+ mw + so + we + exp1 + exp2 + exp3

# Run Linear Specification and compute MSE and R^2
full.fit1 <-  lm(fmla1, data=data)
fit1      <-  summary(full.fit1)
R2.1      <-  fit1$r.squared
R2.adj1   <-  fit1$adj.r.squared
n1        <-  length(fit1$res)
p1        <-  fit1$df[1]
MSE.adj1  <-  (n1/(n1-p1))*mean(fit1$res^2)

# Linear regression: Quadratic specification
fmla2     <- wage ~  female + (sc+ cg+ mw + so + we + exp1 + exp2 + exp3)^2

# Run Quadratic Specification and compute MSE and R^2
full.fit2 <- lm(fmla2, data=data)
fit2      <- summary(full.fit2)
R2.2      <- fit2$r.squared
R2.adj2   <- fit2$adj.r.squared
n2        <- length(fit2$res)
p2        <- fit2$df[1]
MSE.adj2  <- (n2/(n2-p2))*mean(fit2$res^2)


# Summary of linear and quadratic specifications
table1     <- matrix(0, 2, 4)
table1[1,] <- c(p1, R2.1, R2.adj1, MSE.adj1)
table1[2,] <- c(p2, R2.2, R2.adj2, MSE.adj2)  

# Print Regresssion Results
colnames(table1) <- c("p", "R^2", "R^2 adj", "MSE adj")
rownames(table1) <- c("basic reg", "flex reg")


####################  Linear and Quadratic specifications with Sample Splitting ##############################

# set random number generator
set.seed(123)	

# split data into training and test sample
train      <- sample(1:nrow(data), nrow(data)/2) 	

# run linear specification and compute MSE and R^2 for the test sample
full.fit1  <- lm(fmla1, data=data[train,])
yhat.fit1  <- predict(full.fit1, newdata=data[-train,])	
y.test     <- data[-train,]$wage
MSE.fit1   <- summary(lm((y.test-yhat.fit1)^2~1))$coef[1]
R2.fit1    <- 1- MSE.fit1/var(y.test)

# split data into training and test sample
train      <- sample(1:nrow(data), nrow(data)/2) 	

# run quadratic specification and compute MSE and R^2 for the test sample
full.fit2  <- lm(fmla2, data=data[train,])
yhat.fit2  <- predict(full.fit2, newdata=data[-train,])	
y.test     <- data[-train,]$wage
MSE.fit2   <- summary(lm((y.test-yhat.fit2)^2~1))$coef[1]
R2.fit2    <- 1- MSE.fit2/var(y.test)

# Create result table
table2      <- matrix(0, 2, 3)
table2[1,]  <- c(p1, R2.fit1, MSE.fit1)
table2[2,]  <- c(p2, R2.fit2, MSE.fit2)  

# Give Columns and Row Names
colnames(table2)  <- c("p ", "R2 test", "MSE test")
rownames(table2)  <- c("basic reg", "flex reg")


# Print Results
print(table1,digits=4)
print(table2,digits=4)

