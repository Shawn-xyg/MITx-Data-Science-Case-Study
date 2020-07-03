##############################  variables ####################################  

# wage               : wage, lwage
# gender             : sex
# race               : white, black, hisp
# education          : shs, hsg, scl, clg
# region             : mw, so, we
# union membership   : union
# veteran status     : vet
# city               : cent, ncent
# family size        : fam1, fam2, fam3
# having children    : child
# foreign born       : fborn
# citizenship        : cit
# school attandence  : school
# pension            : pens
# firm size          : fsize10, fsize100
# health status      : health
# age                : age
# experience         : exp1, exp2, exp3, exp4
# occupation         : occ(factor with 456 levels), occ2(factor with 22 levels-aggregated)
# industry           : ind(factor with 257 levels), ind2(factor with 23 levels-aggregated)

###############################################################################    

# Clear workspace
rm(list=ls())

# Load Necessary Libraries
library(hdm) 
library(randomForest)
library(glmnet)
library(nnet)
options(warn=-1)
library(rpart)
library(nnet)
library(gbm)
library(rpart.plot)
library(xtable)

# Set Directory
setwd("/Users/VC/Dropbox/MITReg/CASES/case4")

# Load Data
load(file="wage2015.Rdata")


# Split Data into Training and Testing Sample
set.seed(1)
# Set of indices for training
training <- sample(nrow(data), nrow(data)*(1/2), replace=FALSE)

# training data
datause <- data[training,]

# test data
dataout <- data[-training,]

# Linear Control Variables. Use This Control Variables for Tree Based Machine Learning Methods.
x <- "sex+white+black+hisp+shs+hsg+scl+clg+mw+so+we+union+vet+cent+ncent+fam1+fam2+fam3+child+fborn+cit+school+pens+fsize10+fsize100+health+age+exp1+occ2+ind2"

# Quadratic Control Variables(Flexible Specification). Use This Control Variables for Linear methods
xL <- "(sex+white+black+hisp+shs+hsg+scl+clg+mw+so+we+union+vet+cent+ncent+fam1+fam2+fam3+child+fborn+cit+school+pens+fsize10+fsize100+health+age+exp1+exp2+exp3+exp4+occ2+ind2)^2"

# outcome variable log wage
y  <- "lwage"

# Linear Model: Quadratic (Large) Specification
formL <- as.formula(paste(y, "~", xL))

# Linear Model: Linear Specification
form  <- as.formula(paste(y, "~", x))




# "-1" : do not include constant in linear model
# x,y TRUE: return matrix of covariates/vector of outcomes
# Run these linear regression to use their outcome variables (fituse$y) and covariates(fituse$y) 
# a trick to extract x and y variables from a formula.
fituseL    <- lm(paste(y, "~", xL, "-1"), datause, x=TRUE, y=TRUE)  
fitoutL    <- lm(paste(y, "~", xL, "-1"), dataout, x=TRUE, y=TRUE)  
fituse     <- lm(paste(y, "~", x, "-1"), datause, x=TRUE, y=TRUE)  
fitout     <- lm(paste(y, "~", x, "-1"), dataout, x=TRUE, y=TRUE)  


########################################## Train Models ##########################################

# linear regression 
fit.lm      <- lm(form, datause)  
fit.lm2     <- lm(formL, datause)  

#lasso regression 
# alpha=1: first norm(lasso), alpha=0: second norm (ridge)
# alpha = 0.5 : both penalties (elastic net)

#post = FALSE: do not re-run least squares on selected columns

#Lasso
fit.rlasso  <- rlasso(form, datause, post=FALSE)

#Post Lasso
fit.rlasso2 <- rlasso(form, datause, post=TRUE)

#CV-Lasso
fit.lasso   <- cv.glmnet(fituse$x, fituse$y, family="gaussian", alpha=1)

#CV-Ridge
fit.ridge   <- cv.glmnet(fituse$x, fituse$y, family="gaussian", alpha=0)

#CV-Elastic Net
fit.elnet   <- cv.glmnet(fituse$x, fituse$y, family="gaussian", alpha=.5)

#Lasso Flexible
fit.rlassoL  <- rlasso(formL, datause, post=FALSE)

#Post-Lasso Flexible
fit.rlasso2L <- rlasso(formL, datause, post=TRUE)

#CV-Lasso Flexible
fit.lassoL   <- cv.glmnet(fituseL$x, fituseL$y, family="gaussian", alpha=1)

#CV-Ridge Flexible
fit.ridgeL   <- cv.glmnet(fituseL$x, fituseL$y, family="gaussian", alpha=0)

#CV-Elnet Flexible
fit.elnetL   <- cv.glmnet(fituseL$x, fituseL$y, family="gaussian", alpha=.5)

#Random Forest
fit.rf       <- randomForest(form, ntree=2000, nodesize=5, data=datause)

#Boosting Trees
fit.boost   <- gbm(form, data=datause, distribution= "gaussian", bag.fraction = .5, interaction.depth=2, n.trees=1000, shrinkage=.01)
best.boost  <- gbm.perf(fit.boost, plot.it = FALSE)

#Regression Trees
fit.trees   <- rpart(form, datause)
bestcp      <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]
fit.prunedtree <- prune(fit.trees,cp=bestcp)

#Neural Network
fit.nnet    <- nnet(form, datause, size=5,  maxit=1000, MaxNWts=100000, decay=0.01, linout = TRUE, trace=FALSE)   # simple neural net fit

########################################## Compute out-of-sample Predictions ##########################################

yhat.lm       <- predict(fit.lm, newdata=dataout)
yhat.lm2      <- predict(fit.lm2, newdata=dataout)
yhat.rlasso   <- predict(fit.rlasso, newdata=dataout)
yhat.rlasso2  <- predict(fit.rlasso2, newdata=dataout)
yhat.lasso    <- predict(fit.lasso, newx = fitout$x)
yhat.ridge    <- predict(fit.ridge, newx = fitout$x)
yhat.elnet    <- predict(fit.elnet, newx = fitout$x)
yhat.rlassoL  <- predict(fit.rlassoL, newdata=dataout)
yhat.rlasso2L <- predict(fit.rlasso2L, newdata=dataout)
yhat.lassoL   <- predict(fit.lassoL, newx = fitoutL$x)
yhat.ridgeL   <- predict(fit.ridgeL, newx = fitoutL$x)
yhat.elnetL   <- predict(fit.elnetL, newx = fitoutL$x)
yhat.rf       <- predict(fit.rf, newdata=dataout)
yhat.boost    <- predict(fit.boost, newdata=dataout, n.trees=best.boost)
yhat.pt       <- predict(fit.prunedtree,newdata=dataout)
yhat.nnet     <- predict(fit.nnet, newdata=dataout)

########################################## Compute Mean Squared Error for Each Model ##########################################

y.test       = dataout$lwage
MSE.lm       = summary(lm((y.test-yhat.lm)^2~1))$coef[1:2]
MSE.lm2      = summary(lm((y.test-yhat.lm2)^2~1))$coef[1:2]
MSE.rlasso   = summary(lm((y.test-yhat.rlasso)^2~1))$coef[1:2]
MSE.rlasso2  = summary(lm((y.test-yhat.rlasso2)^2~1))$coef[1:2]
MSE.lasso    = summary(lm((y.test-yhat.lasso)^2~1))$coef[1:2]
MSE.ridge    = summary(lm((y.test-yhat.ridge)^2~1))$coef[1:2]
MSE.elnet    = summary(lm((y.test-yhat.elnet)^2~1))$coef[1:2]
MSE.rlassoL  = summary(lm((y.test-yhat.rlassoL)^2~1))$coef[1:2]
MSE.rlasso2L = summary(lm((y.test-yhat.rlasso2L)^2~1))$coef[1:2]
MSE.lassoL   = summary(lm((y.test-yhat.lassoL)^2~1))$coef[1:2]
MSE.ridgeL   = summary(lm((y.test-yhat.ridgeL)^2~1))$coef[1:2]
MSE.elnetL   = summary(lm((y.test-yhat.elnetL)^2~1))$coef[1:2]
MSE.rf       = summary(lm((y.test-yhat.rf)^2~1))$coef[1:2]
MSE.boost    = summary(lm((y.test-yhat.boost)^2~1))$coef[1:2]
MSE.pt       = summary(lm((y.test-yhat.pt)^2~1))$coef[1:2]
MSE.nnet     = summary(lm((y.test-yhat.nnet)^2~1))$coef[1:2]	


########################################## Process Result into a Table ##########################################

table          <- matrix(0, 16, 3)
table[1,1:2]   <- MSE.lm
table[2,1:2]   <- MSE.lm2
table[3,1:2]   <- MSE.rlasso
table[4,1:2]   <- MSE.rlassoL
table[5,1:2]   <- MSE.rlasso2
table[6,1:2]   <- MSE.rlasso2L
table[7,1:2]   <- MSE.lasso
table[8,1:2]   <- MSE.lassoL
table[9,1:2]   <- MSE.ridge
table[10,1:2]  <- MSE.ridgeL
table[11,1:2]  <- MSE.elnet
table[12,1:2]  <- MSE.elnetL
table[13,1:2]  <- MSE.rf
table[14,1:2]  <- MSE.boost
table[15,1:2]  <- MSE.pt
table[16,1:2]  <- MSE.nnet

table[1,3]   <- 1-MSE.lm[1]/var(y.test)
table[2,3]   <- 1-MSE.lm2[1]/var(y.test)
table[3,3]   <- 1-MSE.rlasso[1]/var(y.test)
table[4,3]   <- 1-MSE.rlassoL[1]/var(y.test)
table[5,3]   <- 1-MSE.rlasso2[1]/var(y.test)
table[6,3]   <- 1-MSE.rlasso2L[1]/var(y.test)
table[7,3]   <- 1-MSE.lasso[1]/var(y.test)
table[8,3]   <- 1-MSE.lassoL[1]/var(y.test)
table[9,3]   <- 1-MSE.ridge[1]/var(y.test)
table[10,3]  <- 1-MSE.ridgeL[1]/var(y.test)
table[11,3]  <- 1-MSE.elnet[1]/var(y.test)
table[12,3]  <- 1-MSE.elnetL[1]/var(y.test)
table[13,3]  <- 1-MSE.rf[1]/var(y.test)
table[14,3]  <- 1-MSE.boost[1]/var(y.test)
table[15,3]  <- 1-MSE.pt[1]/var(y.test)
table[16,3]  <- 1-MSE.nnet[1]/var(y.test)


# Give column and row names
colnames(table)<- c("MSE", "S.E. for MSE", "R-squared")
rownames(table)<- c("Least Squares", "Least Squares(Flexible)", "Lasso", "Lasso(Flexible)", "Post-Lasso",  "Post-Lasso(Flexible)", 
                    "Cross-Validated lasso", "Cross-Validated lasso(Flexible)", "Cross-Validated ridge", "Cross-Validated ridge(Flexible)", "Cross-Validated elnet", "Cross-Validated elnet(Flexible)",  
                    "Random Forest","Boosted Trees", "Pruned Tree", "Neural Network")


#########################  Combinining Predictions/ Aggregations/ Ensemble Learning |Introduction  #################################

# Regress the outcome on predictions returned by each method : ordinary least squares
ens  <- lm(y.test~ yhat.lm+ yhat.rlasso+ yhat.elnet + yhat.rf+ yhat.pt +yhat.boost)
# Regress the outcome on predictions returned by each method : lasso, post=FALSE
ens2 <- rlasso(y.test~ yhat.lm+ yhat.rlasso+ yhat.elnet + yhat.rf+ yhat.pt + yhat.boost, post=FALSE)


# Mean Squared Error for Ensemble Learning
MSE.ens1  <- summary(lm((y.test-ens$fitted.values)^2~1))$coef[1:2]
MSE.ens2  <- summary(lm((y.test-predict(ens2))^2~1))$coef[1:2]	

# Table of Results for Ensemble Learning
table2<- matrix(0, 7, 2)

table2[1,1]  <- ens$coefficients[1]
table2[2,1]  <- ens$coefficients[2]
table2[3,1]  <- ens$coefficients[3]
table2[4,1]  <- ens$coefficients[4]
table2[5,1]  <- ens$coefficients[5]
table2[6,1]  <- ens$coefficients[6]
table2[7,1]  <- ens$coefficients[7]


table2[1,2]  <- ens2$coefficients[1]
table2[2,2]  <- ens2$coefficients[2]
table2[3,2]  <- ens2$coefficients[3]
table2[4,2]  <- ens2$coefficients[4]
table2[5,2]  <- ens2$coefficients[5]
table2[6,2]  <- ens2$coefficients[6]
table2[7,2]  <- ens2$coefficients[7]


# Give Column and Row Names
colnames(table2)<- c("Weight(OLS)", "Weight(Rlasso)")
rownames(table2)<- c("Constant","Basic OLS","Lasso","Cross-Validated elnet", "Random Forest", "Pruned Tree","Boosted Trees")

# Print Results
print(table, digits=3)
print(table2, digits=3)





