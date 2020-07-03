############### Part I: MC Algorithms: Rlasso, Tree, Neuralnet, Nnet, Boosting, Random Forest, Lava ################### 

# Create lasso residuals
lassoF <- function(datause, dataout, form_x, form_y, logit=FALSE, alp){
  
  form            <- as.formula(paste(form_y, "~", form_x));
  
  if(logit==TRUE){
    fit           <- lm(form,  x = TRUE, y = TRUE, data=datause);
    lasso         <- cv.glmnet(fit$x[ ,-1], fit$y, family="binomial", alpha=alp)
  }
  
  if(logit==FALSE){
    fit           <- lm(form,  x = TRUE, y = TRUE, data=datause);
    lasso         <- cv.glmnet(fit$x[ ,-1], fit$y, alpha=alp)
  }

  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause);   
  yhatuse         <- predict(lasso, newx=fit.p$x[,-1], s="lambda.1se")
  if(logit==TRUE){yhatuse         <- predict(lasso, newx=fit.p$x[,-1], s="lambda.1se", type="response")}
  resuse          <- fit.p$y - yhatuse
  xuse            <- fit.p$x
 
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout); 
  yhatout         <- predict(lasso, newx=fit.p$x[,-1], s="lambda.1se")
  if(logit==TRUE){ yhatout         <- predict(lasso, newx=fit.p$x[,-1], s="lambda.1se", type="response")}  
  resout          <- fit.p$y - yhatout
  xout            <- fit.p$x
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=lasso, yout=fit.p$y));
  
}


rlassoF <- function(datause, dataout, form_x, form_y, post, logit=FALSE){
  
  form            <- as.formula(paste(form_y, "~", form_x));
  
  if(logit==FALSE){
    lasso         <- rlasso(form, data=datause, post = post, intercept=TRUE)
  }
  
  if(logit==TRUE){
    lasso         <- rlassologit(form, data=datause, post = post, intercept=TRUE)
  }
  
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 
  yhatuse         <- predict(lasso, newdata=fit.p$x, type = "response")
  resuse          <- fit.p$y - predict(lasso, newdata=fit.p$x, type = "response")
  xuse            <- fit.p$x
  
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout); 
  yhatout         <- predict(lasso, newdata=fit.p$x, type = "response")  
  resout          <- fit.p$y - predict(lasso, newdata=fit.p$x, type = "response")
  xout            <- fit.p$x
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=lasso, yout=fit.p$y));
  
}

#Tree residual
tree <- function(datause, dataout, form_x, form_y){
  
  form           <- as.formula(paste(form_y, "~", form_x));
  # create tree
  trees          <- rpart(form, data=datause)
  # tree with minimal xerror among prun
  bestcp         <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]
  # best pruned tree
  ptree          <- prune(trees,cp=bestcp)
  
  #
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=datause); 
  yhatuse        <- predict(ptree, newdata=datause)
  resuse         <- fit.p$y - yhatuse
  xuse           <- fit.p$x
  
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout); 
  yhatout         <- predict(ptree, newdata=dataout)
  resout          <- fit.p$y - yhatout
  xout            <- fit.p$x
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=ptree));
  
}
# Neural net residual
nnetF <- function(datause, dataout, form_x, form_y, clas=FALSE){
  
  linout=FALSE
  if(clas==TRUE){ linout=FALSE}
  maxs <- apply(datause, 2, max) 
  mins <- apply(datause, 2, min)
  
  datause <- as.data.frame(scale(datause, center = mins, scale = maxs - mins))
  dataout <- as.data.frame(scale(dataout, center = mins, scale = maxs - mins))
  
  form           <- as.formula(paste(form_y, "~", form_x))
  
  nn             <- nnet(form, data=datause, size=8,  maxit=1000, decay=0.01, MaxNWts=10000, linout = linout, trace=FALSE)
  k              <- which(colnames(dataout)==form_y)
  
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=datause); 
  yhatuse        <- predict(nn, datause)*(maxs[k]-mins[k])+mins[k]
  resuse         <- fit.p$y*((maxs[k]-mins[k])+mins[k]) - yhatuse
  xuse           <- fit.p$x
  
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=dataout); 
  yhatout        <- predict(nn, dataout)*(maxs[k]-mins[k])+mins[k]
  resout         <- fit.p$y*(maxs[k]-mins[k])+mins[k] - yhatout
  xout           <- fit.p$x
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=nn, min=mins, max=maxs,k=k));
}

# Gradient boosting machine residual
boost <- function(datause, dataout, form_x, form_y, bag.fraction = .5, interaction.depth=2, n.trees=1000, shrinkage=.01, distribution='gaussian'){
  
  form           <- as.formula(paste(form_y, "~", form_x));
  boostfit       <- gbm(form,  distribution=distribution, data=datause, bag.fraction = bag.fraction, interaction.depth=interaction.depth, n.trees=n.trees, shrinkage=shrinkage ,verbose = FALSE,cv.folds=10)
  best           <- gbm.perf(boostfit,plot.it=FALSE,method="cv")
  
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=datause); 
  yhatuse        <- predict(boostfit, n.trees=best)
  resuse         <- fit.p$y - yhatuse
  xuse           <- fit.p$x
  
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=dataout); 
  yhatout        <- predict(boostfit, n.trees=best, newdata=dataout,  type="response")
  resout         <- fit.p$y - yhatout
  xout           <- fit.p$x
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=boostfit, best=best));
}

# Random forest residual
RF <- function(datause, dataout,  form_x, form_y, x=NA, y=NA, xout=NA, yout=NA, nodesize, ntree, reg=TRUE, tune=FALSE){
  
  yhatout <- NA
  reuse   <- NA
  yhatuse <- NA
  resout  <- NA
  
  
  if(is.na(x)){
    form            <- as.formula(paste(form_y, "~", form_x));
    
    if(tune==FALSE){
      forest       <- randomForest(form, nodesize=nodesize, ntree=ntree,  na.action=na.omit, data=datause)
    }
    if(tune==TRUE){
      fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 
      forest_t        <- tuneRF(x=fit.p$x, y=fit.p$y, mtryStart=floor(sqrt(ncol(fit.p$x))), stepFactor=1.5, improve=0.05, nodesize=5, ntree=ntree, doBest=TRUE, plot=FALSE, trace=FALSE)
      min             <- forest_t$mtry
      forest          <- randomForest(form, nodesize=nodesize, mtry=min, ntree=ntree,  na.action=na.omit, data=datause)
    }
    
    fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 
    yhatuse         <- as.numeric(forest$predicted)
    resuse          <- as.numeric(fit.p$y) -  yhatuse
    fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout);    
    if(reg==TRUE)  {yhatout         <- predict(forest, dataout, type="response")}
    if(reg==FALSE) {yhatout         <- predict(forest, dataout, type="prob")[,2]}
    
    resout          <- (as.numeric(fit.p$y)) - as.numeric(yhatout)
  }
  
  if(!is.na(x)){    
    forest          <- randomForest(x=x, y=y, nodesize=nodesize, ntree=ntree,  na.action=na.omit)
    yhatuse         <- as.numeric(forest$predicted)   
    resuse          <- y - yhatuse 
    
    if(!is.na(xout)){
      
      if(reg==TRUE)  {yhatout         <- predict(forest, newdata=xout, type="response")}
      if(reg==FALSE) {yhatout         <- predict(forest, newdata=xout, type="prob")[,2]}
      resuse          <- yout - as.numeric(yhatout)
    }  
  }
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, model = forest));
}


lava <- function(datause, dataout, form_x, form_y, post){
  
  form    <- as.formula(paste(form_y, "~", form_x));
  fit     <- lm(form,  x = TRUE, y = TRUE, data=datause);
  X       <- fit$x[,-1]
  Y       <- fit$y
  n       <- nrow(X)
  p       <- ncol(X)
  
  #   grid2  <- seq(from = 0.00001, to = 1, by = 0.05)
  #   grid <- seq(from = 0.001, to = 2000, by = 100)
  grid2  <- c(seq(from = 0.001, to = 0.1, by = 0.06), seq(from = 0.2, to = 1, by = 0.5), seq(from = 2, to = 10, by = 5), 100000000)
  grid <- c(seq(from = 0.001, to = 1000, by = 500), seq(from = 2000, to = 4000, by = 2000), 10000000)
  

  
  error  <- matrix(0,length(grid),length(grid2))
  folds <- cut(seq(1,nrow(X)),breaks=5,labels=FALSE)
  
  for(l in 1:5){
    
    testIndexes <- which(folds==l,arr.ind=TRUE)
    Xuse     <- X[-testIndexes, ]
    Xout     <- X[testIndexes, ]
    Yuse     <- Y[-testIndexes]
    Yout     <- Y[testIndexes]
    
    mean    <- colMeans(Xuse)
    std     <- apply(Xuse, 2, sd)
    meanY   <- mean(Yuse)
    
    Xuse    <-(Xuse - matrix(1,nrow(Xuse),1)%*% mean) / matrix(1,nrow(Xuse),1)%*% std
    Yuse    <-(Yuse - meanY)
    
    ite     <- 30
    
    for(j in 1:length(grid2)){
      for(i in 1:length(grid)){
        obj <- matrix(100000000,ite,1)
        delta <- matrix(0,p,1)
        for(t in 2:ite){
          
          beta    <- solve(t(Xuse) %*% Xuse + nrow(Xuse)*grid2[j]  * diag(p)) %*% t(Xuse) %*% ( Yuse - Xuse %*% delta)
          delta   <- glmnet(Xuse, Yuse - Xuse %*% beta , family = "gaussian", lambda = grid[i], intercept=FALSE,standardize=FALSE)$beta ;
          theta   <- delta + beta
          
          obj[t,1]  <- sum((Yuse-Xuse %*% theta)^2)/length(Yuse) + grid2[j]*sum(beta^2) + grid[i]*sum(abs(delta))
          if((obj[t-1,1]-obj[t,1])<0.0001){  break  }
        }
        
        if(post==TRUE){
          select <- which((delta!=0))
          if(length(select)>0){
            deltaP <- lm(as.matrix(Yuse - Xuse %*% beta) ~ as.matrix(Xuse[,select])-1)$coef
            deltaP[is.na(deltaP)] <- 0
            delta[select]  <- deltaP
            theta  <- delta + beta  
          }  
        }
        theta      <- theta/std
        XoutN      <- (Xout - matrix(1,nrow(Xout),1)%*% mean) 
        error[i,j] <- mean((Yout - meanY - XoutN %*%(theta))^2) + error[i,j]
      }
    }
  }  
  index <- which(error==min(error), arr.ind=TRUE)
  
  if(length(index)>2){
    lambda1 <- grid[index[1,1]]
    lambda2 <- grid2[index[1,2]]
  }
  
  if(length(index)==2){  
    lambda1 <- grid[index[1]]
    lambda2 <- grid2[index[2]]
  }
  
  mean    <- colMeans(X)
  std     <- apply(X, 2, sd)
  meanY   <- mean(Y)
  
  XN    <-(X - matrix(1,nrow(X),1)%*% mean) / matrix(1,nrow(X),1)%*% std
  YN    <-(Y - meanY)
  
  obj <- matrix(1000000,50,1)
  delta <- matrix(0,p,1)
  for(t in 2:30){
    
    beta    <- solve(t(XN) %*% XN + nrow(XN)*lambda2  * diag(p)) %*% t(XN) %*% (YN - XN %*% delta)
    delta   <- glmnet(XN,  YN - XN %*% beta , family = "gaussian", lambda = lambda1, intercept=FALSE,standardize=FALSE)$beta  ;
    theta   <- delta + beta
    
    obj[t,1]  <- sum((YN-XN %*% theta)^2)/length(Y) + grid2[j]*sum(beta^2) + grid[i]*sum(abs(delta))
    if((obj[t-1,1]-obj[t,1])<0.00001){break}
  }
  
  if(post==TRUE){
    
    select <- which((delta!=0))
    if(length(select)>0){
      deltaP <- lm(as.matrix(YN - XN %*% beta) ~as.matrix(XN[,select])-1)$coef
      deltaP[is.na(deltaP)] <- 0
      delta[select]  <- deltaP
      theta  <- delta + beta  
    }    
  }
  
  theta <- theta/std
  
  form    <- as.formula(paste(form_y, "~", form_x));
  fit     <- lm(form,  x = TRUE, y = TRUE, data=dataout);
  yhatout <- ((fit$x[,-1] - matrix(1,nrow(fit$x[,-1]),1)%*% mean)  %*% theta) + meanY
  resout  <- fit$y - yhatout 
  
  form    <- as.formula(paste(form_y, "~", form_x));
  fit     <- lm(form,  x = TRUE, y = TRUE, data=datause);
  yhatuse <- ((fit$x[,-1] - matrix(1,nrow(fit$x[,-1]),1)%*% mean)  %*% theta) + meanY
  resuse  <- fit$y - yhatuse
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, model=theta, mean=mean, std=std, meanY=meanY));
  
}

############# Auxilary Functions  ########################################################;

# Check if v is a binary (0,1) outcome, accouting for NAs
checkBinary = function(v){
    x <- unique(v)
    length(x) - sum(is.na(x)) == 2L && all(x[1:2] == 0:1)
}

# Returns MSE and misclassification rate for binary vars
error <- function(yhat,y){
  
  err         <- sqrt(mean((yhat-y)^2))
  mis         <- sum(abs(as.numeric(yhat > .5)-(as.numeric(y))))/length(y)   
  
  return(list(err = err, mis=mis));
  
}

formC <- function(form_y,form_x, data){
  
  form            <- as.formula(paste(form_y, "~", form_x));    
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=data); 
  
  return(list(x = fit.p$x, y=fit.p$y));
}

# Average treatment effect
ATE <- function(y, d, my_d1x, my_d0x, md_x)
{
  return( mean( (d * (y - my_d1x) / md_x) -  ((1 - d) * (y - my_d0x) / (1 - md_x)) + my_d1x - my_d0x ) );
}


# St.error of Average treatment effect
SE.ATE <- function(y, d, my_d1x, my_d0x, md_x)
{
  return( sd( (d * (y - my_d1x) / md_x) -  ((1 - d) * (y - my_d0x) / (1 - md_x)) + my_d1x - my_d0x )/sqrt(length(y)) );
}
