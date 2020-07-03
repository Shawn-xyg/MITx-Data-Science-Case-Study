# R code for Regression 1.6: Gender Waage Gap

# Load hdm library
library(hdm)

# Load Dataset and see variables and the number of observations.
load(file="growth.Rdata")
dim(growth)	
str(growth)	

# Get variable names
varnames= colnames(growth)	

# Set Directory
setwd("/Users/VC/Dropbox/MITReg/CASES/case2")

# Extract the names of control and treatment variables from varnames
xnames     <- varnames[-c(1,2,3)]     # names of X variables	
dandxnames <- varnames[-c(1,2)]       # names of D and X variables	

# create formulas by pasting names (this saves typing times)	
fmla      <-  as.formula(paste("Outcome ~ ", paste(dandxnames, collapse= "+")))	
full.fit  <-  lm(fmla, data=growth)	
fmla.y    <-  as.formula(paste("Outcome ~ ", paste(xnames, collapse= "+")))	
fmla.d    <-  as.formula(paste("gdpsh465~ ", paste(xnames, collapse= "+")))	


# partial d and y by linear regression
rY       <- rlasso(fmla.y, data =growth)$res	
rD       <- rlasso(fmla.d, data =growth)$res	


# regress partialed out Y on partialed out D
partial.fit.lasso <- lm(rY~rD-1)	

# create table to store results
table      <- matrix(0, 2, 2)	
table[1,]  <- summary(full.fit)$coef["gdpsh465",1:2]	
table[2,]  <- summary(partial.fit.lasso)$coef[1,1:2]	

# give column and row names
colnames(table) <- names(summary(full.fit)$coef["gdpsh465",])[1:2]	
rownames(table) <- c("Least Squares", "Partialling-out via lasso")	

# print results
print(table, digits=2)

