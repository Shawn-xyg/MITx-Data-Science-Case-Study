################################# Load Libraries #################################

library(foreign);
library(quantreg);
library(mnormt);
library(gbm);
library(glmnet);
library(MASS);
library(rpart);
library(sandwich);
library(hdm);
library(randomForest);
library(xtable)
library(nnet)
library(neuralnet)
library(caret)
library(matrixStats)
library(devtools)
library(plyr)

#################################  Loading functions and Data ########################

# Clear variables.
rm(list = ls())  

# Set Random Number Generator.
set.seed(1)

# Set Directory
setwd("/Users/VC/Dropbox/MITReg/CASES/case5")

# Load Functions from External Files.
source("Cond-comp.R")  
source("Functions.R")  

# Load Data
data <- read.csv("gun_clean.csv") 

# Table for Storing Results
table <- matrix(0,24,1)

#################################  Find Variable Names from Dataset ########################

varlist <- function (df=NULL,type=c("numeric","factor","character"), pattern="", exclude=NULL) {
  vars <- character(0)
  if (any(type %in% "numeric")) {
    vars <- c(vars,names(df)[sapply(df,is.numeric)])
  }
  if (any(type %in% "factor")) {
    vars <- c(vars,names(df)[sapply(df,is.factor)])
  }  
  if (any(type %in% "character")) {
    vars <- c(vars,names(df)[sapply(df,is.character)])
  }  
  vars[(!vars %in% exclude) & grepl(vars,pattern=pattern)]
}

################################# Create Variables ###############################


# Dummy Variables for Year and County Fixed Effects
fixed  <- grep("X_Jfips", names(data), value=TRUE, fixed=TRUE)
year   <- varlist(data, pattern="X_Tyear")

# Census Control Variables
census     <- NULL
census_var <- c("^AGE", "^BN", "^BP", "^BZ", "^ED", "^EL","^HI", "^HS", "^INC", "^LF", "^LN", "^PI", "^PO", "^PP", "^PV", "^SPR", "^VS")

for(i in 1:length(census_var)){
  
  census  <- append(census, varlist(data, pattern=census_var[i]))
  
}


# Treatment Variable
d     <- "logfssl"

# Outcome Variable
y     <- "logghomr"

# Other Control Variables
X1    <- c("logrobr", "logburg", "burg_missing", "robrate_missing")
X2    <- c("newblack", "newfhh", "newmove", "newdens", "newmal")



#################################  Partial out Fixed Effects ########################

# New Dataset for Partiled-out Variables
rdata    <- as.data.frame(data$CountyCode) 
colnames(rdata) <- "CountyCode"

# Variables to be Partialled-out
varlist <- c(y, d,X1, X2, census)


# Partial out Variables in varlist from year and county fixed effect
for(i in 1:length(varlist)){
  form <- as.formula(paste(varlist[i], "~", paste(paste(year,collapse="+"),  paste(fixed,collapse="+"), sep="+")))
  rdata[, varlist[i]] <- lm(form, data)$residuals
}

############################# Linear Regression #############################

form1 <- as.formula(paste(y, "~", d ))
form2 <- as.formula(paste(y, "~", paste(d, paste(X1,collapse="+"), paste(X2,collapse="+"), paste(census,collapse="+"),   sep="+")))

table[1:2,1] <- summary(lm(form1, rdata))$coef[2,1:2]
table[3:4,1] <- summary(lm(form2, rdata))$coef[2,1:2]


######################## Double Machine Learning Methods ########################

# Outcome Variable
y.name      <-  y;             

# Treatment Indicator
d.name      <- d;        

# Controls
x.name      <- paste(paste(X1,collapse="+"),  paste(X2,collapse="+"), paste(census,collapse="+"), sep="+") # use this for tree-based methods like forests and boosted trees

# Method names
method      <- c("RLasso","PostRLasso", "Forest", "Boosting", "Trees", "Lasso", "Ridge", "Elnet", "Nnet") 

# A Function that Returns Coefficients Estimated by Double Machine Learning Methods
#argument list:
#    1 : data
#    2 : outcome variable
#    3 : treatment variable
#    4 : control variables for tree-based methods
#    5 : control variables for linear models(flexible specification) - 4 and 5 are the same for this example
#    6 : method names
#    7 : number of split
#    8 : partially linear model

res <- DoubleML(rdata, y.name, d.name, x.name, x.name, method=method, 2 ,"plinear")

table[5:6,1]   <-res[,1]
table[7:8,1]   <-res[,2]
table[9:10,1]  <-res[,3]
table[11:12,1] <-res[,4]
table[13:14,1] <-res[,5]
table[15:16,1] <-res[,6]
table[17:18,1] <-res[,7]
table[19:20,1] <-res[,8]
table[21:22,1] <-res[,9]
table[23:24,1] <-res[,10]

################################# Print Results #################################

colnames(table) <- c("Gun")
rownames(table) <- c("Baseline1", "se(Baseline1)", "Baseline2", "sd(Baseline2)", "RLasso","sd(RLasso)", "PostRLasso", "sd(PostRLasso)",  "Forest", "sd(Forest)",
                     "Boosting", "sd(Boosting)", "Trees", "sd(Trees)", "Lasso", "sd(Lasso)", "Ridge", "sd(Ridge)", "Elnet", "sd(Elnet)", "Nnet", "sd(Nnet)", "Best", "sd(Best)")


print(table, digit=3)

